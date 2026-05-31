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
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> crate::Result<TransferEstimate>;
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
}
