mod analytical;
mod queued;
mod table;
mod units;

pub use analytical::{AnalyticalTransferModel, TransferEstimate, TransferPath};
pub use queued::QueuedTransferModel;
pub use table::TableCalibratedTransferModel;
pub use units::{format_bytes, format_ns};

/// Abstract transfer model so that runners can swap analytical, queued,
/// or table-calibrated implementations without code changes.
pub trait TransferModel {
    /// Stateful estimate: may advance internal queueing state (e.g.
    /// `QueuedTransferModel::nic_busy_until_ns`). Use for real fetches.
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<TransferEstimate>;

    /// Stateless estimate of the transfer duration in nanoseconds.
    ///
    /// Implementations MUST NOT advance any internal state (NIC queue,
    /// calibration cache, etc.). This is the right entry point for
    /// "what-if" scoring during placement, where calling `estimate` would
    /// pollute the model's state for subsequent real fetches.
    ///
    /// The duration is invariant in `now_ns` for both analytical and
    /// queued models, so `now_ns` is intentionally not a parameter.
    fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<u64>;
}

impl TransferModel for AnalyticalTransferModel {
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<TransferEstimate> {
        AnalyticalTransferModel::estimate(self, now_ns, path, bytes)
    }

    fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<u64> {
        AnalyticalTransferModel::estimate_duration(self, path, bytes)
    }
}

impl TransferModel for QueuedTransferModel {
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<TransferEstimate> {
        QueuedTransferModel::estimate(self, now_ns, path, bytes)
    }

    fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<u64> {
        QueuedTransferModel::estimate_duration(self, path, bytes)
    }
}

impl TransferModel for TableCalibratedTransferModel {
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<TransferEstimate> {
        TableCalibratedTransferModel::estimate(self, now_ns, path, bytes)
    }

    fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<u64> {
        TableCalibratedTransferModel::estimate_duration(self, path, bytes)
    }
}
