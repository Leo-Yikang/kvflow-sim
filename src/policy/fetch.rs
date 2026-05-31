use crate::cache::CacheLocation;
use crate::cluster::NodeId;
use crate::model::ComputeModel;
use crate::transfer::AnalyticalTransferModel;

/// Decision when a reusable KV is found but not on a local GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchDecision {
    /// Fetch the KV to a local GPU; includes estimated transfer time.
    Fetch { duration_ns: u64 },
    /// Recompute the prefix instead of fetching.
    Recompute,
}

/// Decides whether to fetch a remote/off-CPU KV or recompute it.
pub trait FetchPolicy {
    fn decide(
        &self,
        session_id: u64,
        prefix_tokens: u32,
        bytes: u64,
        current_location: CacheLocation,
        target_node: NodeId,
        compute: &dyn ComputeModel,
        transfer: &AnalyticalTransferModel,
    ) -> crate::Result<FetchDecision>;
}
