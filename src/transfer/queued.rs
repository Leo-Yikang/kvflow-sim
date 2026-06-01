use crate::error::{KvFlowError, Result};

use super::{TransferEstimate, TransferPath};

/// A transfer model that adds queueing delay for remote paths by tracking
/// NIC busy state.
///
/// Local transfers (GPU↔GPU, CPU↔GPU) are assumed to have dedicated
/// interconnects and do not queue.  Remote transfers share a NIC that
/// serialises concurrent transfers.
#[derive(Debug, Clone)]
pub struct QueuedTransferModel {
    pub local_gpu_bps: u64,
    pub local_cpu_bps: u64,
    pub remote_memory_bps: u64,
    pub remote_ssd_bps: u64,
    pub local_gpu_base_ns: u64,
    pub local_cpu_base_ns: u64,
    pub remote_memory_base_ns: u64,
    pub remote_ssd_base_ns: u64,
    /// Time until which the shared remote NIC is busy.
    pub nic_busy_until_ns: u64,
}

impl QueuedTransferModel {
    pub fn rdma_400g() -> Self {
        Self {
            local_gpu_bps: 900_000_000_000,
            local_cpu_bps: 200_000_000_000,
            remote_memory_bps: 400_000_000_000,
            remote_ssd_bps: 64_000_000_000,
            local_gpu_base_ns: 500,
            local_cpu_base_ns: 2_000,
            remote_memory_base_ns: 8_000,
            remote_ssd_base_ns: 80_000,
            nic_busy_until_ns: 0,
        }
    }

    pub fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> Result<TransferEstimate> {
        let (bandwidth_bps, base_latency_ns) = match path {
            TransferPath::LocalGpuToGpu => (self.local_gpu_bps, self.local_gpu_base_ns),
            TransferPath::LocalCpuToGpu => (self.local_cpu_bps, self.local_cpu_base_ns),
            TransferPath::RemoteMemoryToGpu => (self.remote_memory_bps, self.remote_memory_base_ns),
            TransferPath::RemoteSsdToGpu => (self.remote_ssd_bps, self.remote_ssd_base_ns),
        };

        if bandwidth_bps == 0 {
            return Err(KvFlowError::InvalidTransferModel(
                "bandwidth must be positive".to_string(),
            ));
        }

        let serialization_ns = serialization_ns(bytes, bandwidth_bps);

        // Remote paths may queue behind the shared NIC.
        let start_ns = if is_remote(path) {
            let start = now_ns.max(self.nic_busy_until_ns);
            self.nic_busy_until_ns = start.saturating_add(serialization_ns);
            start
        } else {
            now_ns
        };

        let finish_ns = start_ns
            .saturating_add(base_latency_ns)
            .saturating_add(serialization_ns);

        Ok(TransferEstimate {
            path,
            bytes,
            start_ns,
            finish_ns,
            base_latency_ns,
            serialization_ns,
            bandwidth_bps,
        })
    }

    /// Stateless duration estimate. Identical to `AnalyticalTransferModel`'s
    /// duration: queueing only delays the *start* time, not the transfer's
    /// own duration. Crucially this MUST NOT advance `nic_busy_until_ns` —
    /// placement scoring is called once per prefill completion and the
    /// runner then immediately calls `estimate` for the next real fetch,
    /// which would otherwise see a polluted start time.
    pub fn estimate_duration(&self, path: TransferPath, bytes: u64) -> Result<u64> {
        let (bandwidth_bps, base_latency_ns) = match path {
            TransferPath::LocalGpuToGpu => (self.local_gpu_bps, self.local_gpu_base_ns),
            TransferPath::LocalCpuToGpu => (self.local_cpu_bps, self.local_cpu_base_ns),
            TransferPath::RemoteMemoryToGpu => (self.remote_memory_bps, self.remote_memory_base_ns),
            TransferPath::RemoteSsdToGpu => (self.remote_ssd_bps, self.remote_ssd_base_ns),
        };

        if bandwidth_bps == 0 {
            return Err(KvFlowError::InvalidTransferModel(
                "bandwidth must be positive".to_string(),
            ));
        }

        let serialization_ns = serialization_ns(bytes, bandwidth_bps);
        Ok(base_latency_ns.saturating_add(serialization_ns))
    }
}

fn is_remote(path: TransferPath) -> bool {
    matches!(
        path,
        TransferPath::RemoteMemoryToGpu | TransferPath::RemoteSsdToGpu
    )
}

fn serialization_ns(bytes: u64, bandwidth_bps: u64) -> u64 {
    let ns = (bytes as u128)
        .saturating_mul(8)
        .saturating_mul(1_000_000_000)
        / bandwidth_bps as u128;
    ns.min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_transfers_queue_when_nic_busy() {
        let mut model = QueuedTransferModel::rdma_400g();
        let est1 = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, 400_000_000)
            .unwrap();
        // First transfer starts immediately.
        assert_eq!(est1.start_ns, 0);

        let est2 = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, 400_000_000)
            .unwrap();
        // Second transfer queues behind the first.
        assert_eq!(est2.start_ns, est1.serialization_ns);
        assert!(est2.finish_ns > est1.finish_ns);
    }

    #[test]
    fn local_transfers_do_not_queue() {
        let mut model = QueuedTransferModel::rdma_400g();
        let est1 = model
            .estimate(0, TransferPath::LocalCpuToGpu, 400_000_000)
            .unwrap();
        let est2 = model
            .estimate(0, TransferPath::LocalCpuToGpu, 400_000_000)
            .unwrap();
        // Both start at 0 because local paths do not share the remote NIC.
        assert_eq!(est1.start_ns, 0);
        assert_eq!(est2.start_ns, 0);
    }

    #[test]
    fn estimate_duration_does_not_advance_nic_busy_state() {
        // Regression: NetworkAwarePlacement calls a transfer model many
        // times per placement to score candidate tiers. The stateful
        // `estimate` would advance `nic_busy_until_ns` and skew the next
        // real fetch's start time. `estimate_duration` must be stateless.
        let model = QueuedTransferModel::rdma_400g();
        assert_eq!(model.nic_busy_until_ns, 0);

        // Score all three remote-relevant tiers with a non-trivial payload.
        let d_cpu = model
            .estimate_duration(TransferPath::LocalCpuToGpu, 64 * 1024)
            .unwrap();
        let d_mem = model
            .estimate_duration(TransferPath::RemoteMemoryToGpu, 64 * 1024)
            .unwrap();
        let d_ssd = model
            .estimate_duration(TransferPath::RemoteSsdToGpu, 64 * 1024)
            .unwrap();

        // All three return a finite duration (the serialization +
        // base-latency, not infinity).
        assert!(d_cpu > 0);
        assert!(d_mem > 0);
        assert!(d_ssd > 0);

        // Critical: the state must be unchanged. Any advance here would
        // mean the placement's "what-if" call is leaking into subsequent
        // real fetches.
        assert_eq!(
            model.nic_busy_until_ns, 0,
            "estimate_duration must not mutate nic_busy_until_ns"
        );
    }
}
