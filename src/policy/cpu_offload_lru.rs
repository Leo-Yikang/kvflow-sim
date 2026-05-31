use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId};
use crate::error::KvFlowError;

use super::{EvictedKv, PlacementPolicy};

/// Store KV on GPU if possible; otherwise offload to CPU.  Eviction uses LRU
/// within each tier independently.
#[derive(Debug, Clone)]
pub struct CpuOffloadLru {
    pub target_node: NodeId,
}

impl PlacementPolicy for CpuOffloadLru {
    fn place(
        &mut self,
        kv_id: u64,
        session_id: u64,
        prefix_tokens: u32,
        bytes: u64,
        now_ns: u64,
        cluster: &ClusterTopology,
        cache: &mut CacheState,
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)> {
        let mut evicted = Vec::new();

        // Try GPU first.
        if let Some(loc) = try_place_in_tier(cluster, cache, self.target_node, TierKind::Gpu, bytes)
        {
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
            return Ok((loc, evicted));
        }

        // Evict from GPU LRU to make room.
        while cache.available_space(self.target_node, TierKind::Gpu) < bytes {
            let candidates: Vec<_> = cache
                .objects_in_tier(self.target_node, TierKind::Gpu)
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

        if let Some(loc) = try_place_in_tier(cluster, cache, self.target_node, TierKind::Gpu, bytes)
        {
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
            return Ok((loc, evicted));
        }

        // Fallback to CPU.
        if let Some(loc) = try_place_in_tier(cluster, cache, self.target_node, TierKind::Cpu, bytes)
        {
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
            return Ok((loc, evicted));
        }

        // Evict from CPU LRU to make room.
        while cache.available_space(self.target_node, TierKind::Cpu) < bytes {
            let candidates: Vec<_> = cache
                .objects_in_tier(self.target_node, TierKind::Cpu)
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

        if let Some(loc) = try_place_in_tier(cluster, cache, self.target_node, TierKind::Cpu, bytes)
        {
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
            return Ok((loc, evicted));
        }

        Err(KvFlowError::InvalidModelProfile(
            "CpuOffloadLru: no space on GPU or CPU for KV object".to_string(),
        ))
    }
}

fn try_place_in_tier(
    cluster: &ClusterTopology,
    cache: &CacheState,
    node_id: NodeId,
    tier: TierKind,
    bytes: u64,
) -> Option<CacheLocation> {
    if cache.available_space(node_id, tier) < bytes {
        return None;
    }
    match tier {
        TierKind::Gpu => {
            let node = cluster.node(node_id)?;
            node.gpus.first().map(|g| CacheLocation::Gpu {
                node_id,
                gpu_id: g.gpu_id,
            })
        }
        TierKind::Cpu => Some(CacheLocation::Cpu { node_id }),
        TierKind::LocalSsd => Some(CacheLocation::LocalSsd { node_id }),
        TierKind::RemoteMemory => Some(CacheLocation::RemoteMemory { node_id }),
        TierKind::RemoteSsd => Some(CacheLocation::RemoteSsd { node_id }),
    }
}
