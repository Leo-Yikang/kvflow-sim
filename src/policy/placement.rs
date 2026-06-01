use crate::cache::{CacheLocation, CacheState};
use crate::cluster::ClusterTopology;
use crate::model::ComputeModel;
use crate::transfer::TransferModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictedKv {
    pub kv_id: u64,
    pub bytes: u64,
}

/// Decides where a newly-computed KV object should be stored.
#[allow(clippy::too_many_arguments)]
pub trait PlacementPolicy {
    /// Called after prefill completes.
    ///
    /// `compute` and `transfer` are passed in so utility-based policies can
    /// score the cost of placing the KV at each tier. Policies that do not
    /// care about per-tier utility may simply ignore them.
    ///
    /// Returns the chosen `CacheLocation` and any objects evicted to make room.
    fn place(
        &mut self,
        kv_id: u64,
        session_id: u64,
        prefix_tokens: u32,
        bytes: u64,
        now_ns: u64,
        cluster: &ClusterTopology,
        cache: &mut CacheState,
        compute: &dyn ComputeModel,
        transfer: &mut dyn TransferModel,
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)>;
}
