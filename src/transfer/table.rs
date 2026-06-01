use std::collections::HashMap;

use crate::error::Result;

use super::{AnalyticalTransferModel, TransferEstimate, TransferPath};

/// A transfer model backed by a calibration table (bytes, path) -> latency.
/// Falls back to an analytical model for missing entries.
#[derive(Debug, Clone)]
pub struct TableCalibratedTransferModel {
    table: HashMap<(TransferPath, u64), u64>,
    fallback: AnalyticalTransferModel,
}

impl TableCalibratedTransferModel {
    pub fn new(fallback: AnalyticalTransferModel) -> Self {
        Self {
            table: HashMap::new(),
            fallback,
        }
    }

    pub fn add_entry(&mut self, path: TransferPath, bytes: u64, latency_ns: u64) {
        self.table.insert((path, bytes), latency_ns);
    }

    pub fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> Result<TransferEstimate> {
        if let Some(&latency_ns) = self.table.get(&(path, bytes)) {
            let start_ns = now_ns;
            let finish_ns = now_ns.saturating_add(latency_ns);
            Ok(TransferEstimate {
                path,
                bytes,
                start_ns,
                finish_ns,
                base_latency_ns: latency_ns,
                serialization_ns: 0,
                bandwidth_bps: 0,
            })
        } else {
            self.fallback.estimate(now_ns, path, bytes)
        }
    }

    /// Stateless duration estimate. Mirrors `estimate` but takes `&self`
    /// and skips the fallback's potential side effects (the fallback is an
    /// `AnalyticalTransferModel` today, which is already stateless, but we
    /// route through `estimate_duration` so future fallback changes cannot
    /// leak state here).
    pub fn estimate_duration(&self, path: TransferPath, bytes: u64) -> Result<u64> {
        if let Some(&latency_ns) = self.table.get(&(path, bytes)) {
            Ok(latency_ns)
        } else {
            self.fallback.estimate_duration(path, bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_hit_uses_custom_latency() {
        let fallback = AnalyticalTransferModel::rdma_400g();
        let mut model = TableCalibratedTransferModel::new(fallback);
        model.add_entry(TransferPath::RemoteMemoryToGpu, 1_000_000, 123_456);

        let est = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, 1_000_000)
            .unwrap();
        assert_eq!(est.finish_ns, 123_456);
    }

    #[test]
    fn table_miss_falls_back() {
        let fallback = AnalyticalTransferModel::rdma_400g();
        let mut model = TableCalibratedTransferModel::new(fallback);

        let est = model
            .estimate(0, TransferPath::RemoteMemoryToGpu, 400_000_000)
            .unwrap();
        assert!(est.finish_ns > 0);
    }
}
