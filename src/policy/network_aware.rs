use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId};
use crate::error::KvFlowError;
use crate::model::ComputeModel;
use crate::transfer::{AnalyticalTransferModel, TransferPath};

use super::{EvictedKv, PlacementPolicy};

/// Network-aware placement: compute a utility score for each tier and pick the
/// best.  Utility = recompute_cost_saved - transfer_cost - cache_pressure.
#[derive(Debug, Clone)]
pub struct NetworkAwarePlacement {
    pub target_node: NodeId,
}

impl NetworkAwarePlacement {
    #[allow(dead_code)]
    fn utility_for_tier(
        &self,
        tier: TierKind,
        bytes: u64,
        prefix_tokens: u32,
        _node_id: NodeId,
        compute: &dyn ComputeModel,
        transfer: &AnalyticalTransferModel,
    ) -> i128 {
        let recompute_cost = compute.prefill_time_ns(prefix_tokens) as i128;

        let fetch_cost: i128 = match tier {
            TierKind::Gpu => 0, // local GPU, no transfer
            TierKind::Cpu => transfer
                .estimate(0, TransferPath::LocalCpuToGpu, bytes)
                .map(|e| e.duration_ns() as i128)
                .unwrap_or(i128::MAX),
            TierKind::RemoteMemory => transfer
                .estimate(0, TransferPath::RemoteMemoryToGpu, bytes)
                .map(|e| e.duration_ns() as i128)
                .unwrap_or(i128::MAX),
            TierKind::RemoteSsd => transfer
                .estimate(0, TransferPath::RemoteSsdToGpu, bytes)
                .map(|e| e.duration_ns() as i128)
                .unwrap_or(i128::MAX),
            TierKind::LocalSsd => i128::MAX, // not modelled explicitly
        };

        // Pressure penalty: discourage remote tiers under contention.
        let pressure_penalty: i128 = match tier {
            TierKind::Gpu => 0,
            TierKind::Cpu => bytes as i128 / 100,
            TierKind::RemoteMemory => bytes as i128 / 10,
            _ => bytes as i128,
        };

        recompute_cost - fetch_cost - pressure_penalty
    }
}

impl PlacementPolicy for NetworkAwarePlacement {
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
        // Default to LocalGpuLru-like behaviour for now.
        // Full utility scoring requires ComputeModel + TransferModel in the
        // trait signature, which we avoid here to keep the trait simple.
        // A production version would extend the trait or use a context object.

        let mut evicted = Vec::new();

        // Try GPU first.
        if cache.available_space(self.target_node, TierKind::Gpu) >= bytes {
            let node = cluster.node(self.target_node).ok_or_else(|| {
                KvFlowError::InvalidModelProfile(format!("node {} not found", self.target_node))
            })?;
            let gpu_id = node.gpus.first().map(|g| g.gpu_id).unwrap_or(0);
            let loc = CacheLocation::Gpu {
                node_id: self.target_node,
                gpu_id,
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

        // Evict from GPU LRU.
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
            let node = cluster.node(self.target_node).ok_or_else(|| {
                KvFlowError::InvalidModelProfile(format!("node {} not found", self.target_node))
            })?;
            let gpu_id = node.gpus.first().map(|g| g.gpu_id).unwrap_or(0);
            let loc = CacheLocation::Gpu {
                node_id: self.target_node,
                gpu_id,
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

        // Fallback to CPU.
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
            "NetworkAwarePlacement: no space on GPU or CPU".to_string(),
        ))
    }
}
