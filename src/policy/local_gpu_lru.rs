use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId};
use crate::error::KvFlowError;

use super::{EvictedKv, PlacementPolicy};

/// Store KV on the local GPU; evict LRU objects within the GPU tier when
/// necessary.  If the object is larger than total GPU capacity, falls back to
/// CPU.
#[derive(Debug, Clone)]
pub struct LocalGpuLru {
    pub target_node: NodeId,
}

impl PlacementPolicy for LocalGpuLru {
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
        if let Some(gpu_idx) = find_gpu_with_space(cluster, cache, self.target_node, bytes) {
            let loc = CacheLocation::Gpu {
                node_id: self.target_node,
                gpu_id: gpu_idx,
            };
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

        // Evict LRU from GPU tier until we have enough space.
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

        if cache.available_space(self.target_node, TierKind::Gpu) >= bytes {
            if let Some(gpu_idx) = find_gpu_with_space(cluster, cache, self.target_node, bytes) {
                let loc = CacheLocation::Gpu {
                    node_id: self.target_node,
                    gpu_id: gpu_idx,
                };
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
        }

        // Fallback: place on CPU if GPU cannot hold it.
        if cache.available_space(self.target_node, TierKind::Cpu) >= bytes {
            let loc = CacheLocation::Cpu {
                node_id: self.target_node,
            };
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
            "LocalGpuLru: no space on GPU or CPU for KV object".to_string(),
        ))
    }
}

fn find_gpu_with_space(
    cluster: &ClusterTopology,
    cache: &CacheState,
    node_id: NodeId,
    bytes: u64,
) -> Option<u32> {
    let node = cluster.node(node_id)?;
    // Simplification: treat all GPU HBM on a node as a single pool.
    // Return the first GPU index if the node-level GPU tier has space.
    if cache.available_space(node_id, TierKind::Gpu) >= bytes {
        node.gpus.first().map(|g| g.gpu_id)
    } else {
        None
    }
}
