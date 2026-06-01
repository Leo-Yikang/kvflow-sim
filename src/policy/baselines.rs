use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId};

use super::{EvictedKv, PlacementPolicy};

/// Always place KV on remote memory (no local caching).
#[derive(Debug, Clone)]
pub struct RemoteLru {
    pub target_node: NodeId,
}

impl PlacementPolicy for RemoteLru {
    fn place(
        &mut self,
        kv_id: u64,
        session_id: u64,
        prefix_tokens: u32,
        bytes: u64,
        now_ns: u64,
        _cluster: &ClusterTopology,
        cache: &mut CacheState,
        _compute: &dyn crate::model::ComputeModel,
        _transfer: &mut dyn crate::transfer::TransferModel,
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)> {
        let mut evicted = Vec::new();
        let loc = CacheLocation::RemoteMemory {
            node_id: self.target_node,
        };

        while cache.available_space(self.target_node, TierKind::RemoteMemory) < bytes {
            let candidates: Vec<_> = cache
                .objects_in_tier(self.target_node, TierKind::RemoteMemory)
                .into_iter()
                .cloned()
                .collect();
            if candidates.is_empty() {
                break;
            }
            let victim = candidates
                .iter()
                .min_by_key(|o| o.last_access_ns)
                .cloned()
                .unwrap();
            cache.remove(victim.kv_id);
            evicted.push(EvictedKv {
                kv_id: victim.kv_id,
                bytes: victim.bytes,
            });
        }

        let obj = KvObject {
            kv_id,
            session_id,
            model_id: String::new(),
            prefix_tokens,
            bytes,
            location: loc,
            last_access_ns: now_ns,
            ref_count: 0,
        };
        cache.insert(obj)?;
        Ok((loc, evicted))
    }
}

/// Place small KV objects on GPU, large ones on CPU (size-aware).
#[derive(Debug, Clone)]
pub struct SizeAware {
    pub target_node: NodeId,
    /// Objects larger than this threshold (bytes) go to CPU.
    pub gpu_threshold_bytes: u64,
}

impl PlacementPolicy for SizeAware {
    fn place(
        &mut self,
        kv_id: u64,
        session_id: u64,
        prefix_tokens: u32,
        bytes: u64,
        now_ns: u64,
        _cluster: &ClusterTopology,
        cache: &mut CacheState,
        _compute: &dyn crate::model::ComputeModel,
        _transfer: &mut dyn crate::transfer::TransferModel,
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)> {
        let mut evicted = Vec::new();
        let tier = if bytes <= self.gpu_threshold_bytes {
            TierKind::Gpu
        } else {
            TierKind::Cpu
        };

        let loc = match tier {
            TierKind::Gpu => CacheLocation::Gpu {
                node_id: self.target_node,
                gpu_id: 0,
            },
            _ => CacheLocation::Cpu {
                node_id: self.target_node,
            },
        };

        while cache.available_space(self.target_node, tier) < bytes {
            let candidates: Vec<_> = cache
                .objects_in_tier(self.target_node, tier)
                .into_iter()
                .cloned()
                .collect();
            if candidates.is_empty() {
                break;
            }
            let victim = candidates
                .iter()
                .min_by_key(|o| o.last_access_ns)
                .cloned()
                .unwrap();
            cache.remove(victim.kv_id);
            evicted.push(EvictedKv {
                kv_id: victim.kv_id,
                bytes: victim.bytes,
            });
        }

        let obj = KvObject {
            kv_id,
            session_id,
            model_id: String::new(),
            prefix_tokens,
            bytes,
            location: loc,
            last_access_ns: now_ns,
            ref_count: 0,
        };
        cache.insert(obj)?;
        Ok((loc, evicted))
    }
}
