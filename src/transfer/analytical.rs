use crate::error::{KvFlowError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferPath {
    LocalGpuToGpu,
    LocalCpuToGpu,
    RemoteMemoryToGpu,
    RemoteSsdToGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferEstimate {
    pub path: TransferPath,
    pub bytes: u64,
    pub start_ns: u64,
    pub finish_ns: u64,
    pub base_latency_ns: u64,
    pub serialization_ns: u64,
    pub bandwidth_bps: u64,
}

impl TransferEstimate {
    pub fn duration_ns(&self) -> u64 {
        self.finish_ns.saturating_sub(self.start_ns)
    }
}

#[derive(Debug, Clone)]
pub struct AnalyticalTransferModel {
    pub local_gpu_bps: u64,
    pub local_cpu_bps: u64,
    pub remote_memory_bps: u64,
    pub remote_ssd_bps: u64,
    pub local_gpu_base_ns: u64,
    pub local_cpu_base_ns: u64,
    pub remote_memory_base_ns: u64,
    pub remote_ssd_base_ns: u64,
}

impl AnalyticalTransferModel {
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
        }
    }

    pub fn estimate(
        &self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> Result<TransferEstimate> {
        let (bandwidth_bps, base_latency_ns) = self.bandwidth_and_latency(path)?;
        let serialization_ns = serialization_ns(bytes, bandwidth_bps);
        let finish_ns = now_ns
            .saturating_add(base_latency_ns)
            .saturating_add(serialization_ns);

        Ok(TransferEstimate {
            path,
            bytes,
            start_ns: now_ns,
            finish_ns,
            base_latency_ns,
            serialization_ns,
            bandwidth_bps,
        })
    }

    /// Stateless duration estimate. Mirrors `estimate` but takes `&self` and
    /// does not return a full `TransferEstimate` (only the total duration).
    /// Useful for placement-time "what-if" scoring.
    pub fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> Result<u64> {
        let (bandwidth_bps, base_latency_ns) = self.bandwidth_and_latency(path)?;
        let serialization_ns = serialization_ns(bytes, bandwidth_bps);
        Ok(base_latency_ns.saturating_add(serialization_ns))
    }

    fn bandwidth_and_latency(
        &self,
        path: TransferPath,
    ) -> Result<(u64, u64)> {
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
        Ok((bandwidth_bps, base_latency_ns))
    }
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
    fn estimate_remote_memory_transfer() {
        let model = AnalyticalTransferModel::rdma_400g();
        let est = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, 400_000_000)
            .unwrap();

        assert_eq!(est.serialization_ns, 8_000_000);
        assert_eq!(est.duration_ns(), 8_008_000);
    }

    #[test]
    fn large_kv_transfer_does_not_overflow_u64_intermediate() {
        let model = AnalyticalTransferModel::rdma_400g();
        let four_gib = 4 * 1024 * 1024 * 1024_u64;
        let est = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, four_gib)
            .unwrap();

        assert_eq!(est.serialization_ns, 85_899_345);
        assert_eq!(est.duration_ns(), 85_907_345);
    }

    #[test]
    fn estimate_duration_matches_estimate_duration_ns() {
        // `estimate_duration` is the stateless counterpart of `estimate`;
        // the returned duration must equal `estimate(...).duration_ns()`
        // so utility-based placement scoring is consistent with the
        // real-fetch accounting.
        let model = AnalyticalTransferModel::rdma_400g();
        for (path, bytes) in [
            (TransferPath::LocalGpuToGpu, 1_000_000_u64),
            (TransferPath::LocalCpuToGpu, 4_000_000),
            (TransferPath::RemoteMemoryToGpu, 64_000_000),
            (TransferPath::RemoteSsdToGpu, 256_000_000),
        ] {
            let from_estimate = model.estimate(0, path, bytes).unwrap().duration_ns();
            let from_duration = model.estimate_duration(path, bytes).unwrap();
            assert_eq!(
                from_estimate, from_duration,
                "estimate_duration ({}) should match estimate.duration_ns ({}) for {:?}/{}",
                from_duration, from_estimate, path, bytes
            );
        }
    }

    #[test]
    fn estimate_duration_rejects_zero_bandwidth() {
        let mut model = AnalyticalTransferModel::rdma_400g();
        model.remote_ssd_bps = 0;
        let err = model
            .estimate_duration(TransferPath::RemoteSsdToGpu, 1024)
            .unwrap_err();
        match err {
            crate::error::KvFlowError::InvalidTransferModel(_) => {}
            other => panic!("expected InvalidTransferModel, got {:?}", other),
        }
    }
}
