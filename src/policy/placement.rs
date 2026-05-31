use crate::cache::{CacheLocation, CacheState};
use crate::cluster::ClusterTopology;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictedKv {
    pub kv_id: u64,
    pub bytes: u64,
}

/// Decides where a newly-computed KV object should be stored.
pub trait PlacementPolicy {
    /// Called after prefill completes.
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
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)>;
}
