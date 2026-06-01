use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId};
use crate::error::KvFlowError;
use crate::model::ComputeModel;
use crate::transfer::{TransferModel, TransferPath};

use super::{EvictedKv, PlacementPolicy};

/// Network-aware placement: score each candidate tier by
/// `utility = recompute_cost - fetch_cost - pressure_penalty` and place the
/// KV in the highest-utility tier that has room (after LRU eviction within
/// the tier).
///
/// Higher utility = better. Tier ordering is recomputed every call, so this
/// policy adapts to the runner's current `ComputeModel` and `TransferModel`.
#[derive(Debug, Clone)]
pub struct NetworkAwarePlacement {
    pub target_node: NodeId,
}

impl NetworkAwarePlacement {
    /// Score a candidate tier.
    ///
    /// Uses `TransferModel::estimate_duration` (not `estimate`) so the call
    /// is stateless. `QueuedTransferModel::estimate` would otherwise advance
    /// `nic_busy_until_ns`, polluting later real fetches' start times.
    fn utility_for_tier(
        &self,
        tier: TierKind,
        bytes: u64,
        prefix_tokens: u32,
        compute: &dyn ComputeModel,
        transfer: &dyn TransferModel,
    ) -> i128 {
        let recompute_cost = compute.prefill_time_ns(prefix_tokens) as i128;

        let fetch_cost: i128 = match tier {
            TierKind::Gpu => 0, // local GPU, no transfer
            TierKind::Cpu => transfer
                .estimate_duration(TransferPath::LocalCpuToGpu, bytes)
                .map(|d| d as i128)
                .unwrap_or(i128::MAX),
            TierKind::RemoteMemory => transfer
                .estimate_duration(TransferPath::RemoteMemoryToGpu, bytes)
                .map(|d| d as i128)
                .unwrap_or(i128::MAX),
            TierKind::RemoteSsd => transfer
                .estimate_duration(TransferPath::RemoteSsdToGpu, bytes)
                .map(|d| d as i128)
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

    /// Build a `CacheLocation` for the given tier on `target_node`.
    /// Returns `None` for tiers we do not place into.
    fn location_for(
        target_node: NodeId,
        tier: TierKind,
        cluster: &ClusterTopology,
    ) -> Option<CacheLocation> {
        match tier {
            TierKind::Gpu => {
                let node = cluster.node(target_node)?;
                let gpu_id = node.gpus.first()?.gpu_id;
                Some(CacheLocation::Gpu {
                    node_id: target_node,
                    gpu_id,
                })
            }
            TierKind::Cpu => Some(CacheLocation::Cpu {
                node_id: target_node,
            }),
            TierKind::RemoteMemory => Some(CacheLocation::RemoteMemory {
                node_id: target_node,
            }),
            TierKind::RemoteSsd => Some(CacheLocation::RemoteSsd {
                node_id: target_node,
            }),
            TierKind::LocalSsd => None, // not modelled explicitly
        }
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
        compute: &dyn ComputeModel,
        transfer: &mut dyn TransferModel,
    ) -> crate::Result<(CacheLocation, Vec<EvictedKv>)> {
        // Score and order candidate tiers (highest utility first).
        // LocalSsd is excluded — it is not a placement target in the current
        // cache model.
        let mut scored: Vec<(TierKind, i128)> = [
            TierKind::Gpu,
            TierKind::Cpu,
            TierKind::RemoteMemory,
            TierKind::RemoteSsd,
        ]
        .iter()
        .map(|&tier| {
            (
                tier,
                self.utility_for_tier(tier, bytes, prefix_tokens, compute, transfer),
            )
        })
        .collect();
        // Stable sort so that ties keep the order: Gpu > Cpu > RemoteMemory > RemoteSsd.
        scored.sort_by_key(|(_, u)| std::cmp::Reverse(*u));

        let mut total_evicted: Vec<EvictedKv> = Vec::new();

        for (tier, _utility) in scored {
            let loc = match Self::location_for(self.target_node, tier, cluster) {
                Some(l) => l,
                None => continue,
            };

            // Capacity short-circuit: if the tier's *total* capacity is
            // smaller than `bytes`, no amount of eviction can make this KV
            // fit. Skip the tier without destroying its contents. This
            // also matters when a higher-utility tier cannot fit the object
            // at all (e.g. oversized KV, GPU tier is small) — we'd rather
            // try the next tier than empty this one.
            if cache.capacity(self.target_node, tier) < bytes {
                continue;
            }

            // Evict LRU from this tier until we have enough space.
            while cache.available_space(self.target_node, tier) < bytes {
                let victims: Vec<_> = cache
                    .objects_in_tier(self.target_node, tier)
                    .into_iter()
                    .cloned()
                    .collect();
                let victim = match victims.iter().min_by_key(|o| o.last_access_ns) {
                    Some(v) => v.clone(),
                    None => break,
                };
                cache.remove(victim.kv_id);
                total_evicted.push(EvictedKv {
                    kv_id: victim.kv_id,
                    bytes: victim.bytes,
                });
            }

            if cache.available_space(self.target_node, tier) >= bytes {
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
                return Ok((loc, total_evicted));
            }
            // Otherwise: every evictable object is gone but we still
            // cannot fit `bytes` (should not happen given the short-circuit
            // above, but be defensive). Try the next-best tier; the
            // evictions we performed are permanent.
        }

        Err(KvFlowError::InvalidModelProfile(format!(
            "NetworkAwarePlacement: no space in any tier on node {} for {} bytes",
            self.target_node, bytes
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{ClusterTopology, GpuResource, ServingNode, WorkerRole};
    use crate::model::LinearComputeModel;
    use crate::transfer::AnalyticalTransferModel;

    fn cluster_with_tiny_gpu() -> ClusterTopology {
        // 1 MiB of GPU HBM — forces offload for any non-trivial KV.
        let gpus = vec![GpuResource::new(0, 1 << 20, WorkerRole::Unified)];
        ClusterTopology::new(vec![ServingNode::new(
            0,
            0,
            gpus,
            1,
            1_000_000_000,      // 1 GiB CPU
            10_000_000_000_000, // 10 TiB local SSD (capacity only)
        )])
    }

    fn place_kv(
        policy: &mut NetworkAwarePlacement,
        cluster: &ClusterTopology,
        cache: &mut CacheState,
        kv_id: u64,
        bytes: u64,
        compute: &dyn ComputeModel,
        transfer: &mut dyn TransferModel,
    ) -> CacheLocation {
        let (loc, _evicted) = policy
            .place(kv_id, 0, 1024, bytes, 0, cluster, cache, compute, transfer)
            .expect("placement should succeed");
        loc
    }

    #[test]
    fn prefers_gpu_when_gpu_has_space() {
        let cluster = cluster_with_tiny_gpu();
        let mut cache = CacheState::new();
        for tier in [
            TierKind::Gpu,
            TierKind::Cpu,
            TierKind::RemoteMemory,
            TierKind::RemoteSsd,
        ] {
            cache.set_capacity(0, tier, 1 << 30);
        }
        let mut policy = NetworkAwarePlacement { target_node: 0 };
        let mut transfer = AnalyticalTransferModel::rdma_400g();
        let compute = LinearComputeModel::conservative_8b();

        // 64 KiB fits comfortably in the 1 MiB GPU tier.
        let loc = place_kv(
            &mut policy,
            &cluster,
            &mut cache,
            1,
            64 * 1024,
            &compute,
            &mut transfer,
        );
        assert!(loc.is_gpu(), "expected GPU placement, got {:?}", loc);
    }

    #[test]
    fn offloads_to_cpu_when_gpu_has_no_capacity() {
        // The discriminating test: GPU has zero capacity (or zero evictable
        // objects), so the policy must fall through to a non-GPU tier. With
        // RDMA-400G, CPU transfer is cheap, so CPU wins over remote due to
        // the pressure penalty.
        let cluster = cluster_with_tiny_gpu();
        let mut cache = CacheState::new();
        // GPU capacity 0 → no object can fit there.
        cache.set_capacity(0, TierKind::Gpu, 0);
        cache.set_capacity(0, TierKind::Cpu, 1 << 30);
        cache.set_capacity(0, TierKind::RemoteMemory, 1 << 30);
        cache.set_capacity(0, TierKind::RemoteSsd, 1 << 30);

        let mut policy = NetworkAwarePlacement { target_node: 0 };
        let mut transfer = AnalyticalTransferModel::rdma_400g();
        let compute = LinearComputeModel::conservative_8b();

        let loc = place_kv(
            &mut policy,
            &cluster,
            &mut cache,
            1,
            64 * 1024,
            &compute,
            &mut transfer,
        );
        assert!(
            matches!(loc, CacheLocation::Cpu { .. }),
            "expected CPU placement when GPU has no capacity, got {:?}",
            loc
        );
    }

    #[test]
    fn falls_through_to_remote_when_gpu_and_cpu_unavailable() {
        // If both GPU and CPU have no capacity, the policy should still
        // succeed by placing on remote memory.
        let cluster = cluster_with_tiny_gpu();
        let mut cache = CacheState::new();
        cache.set_capacity(0, TierKind::Gpu, 0);
        cache.set_capacity(0, TierKind::Cpu, 0);
        cache.set_capacity(0, TierKind::RemoteMemory, 1 << 30);
        cache.set_capacity(0, TierKind::RemoteSsd, 1 << 30);

        let mut policy = NetworkAwarePlacement { target_node: 0 };
        let mut transfer = AnalyticalTransferModel::rdma_400g();
        let compute = LinearComputeModel::conservative_8b();

        let loc = place_kv(
            &mut policy,
            &cluster,
            &mut cache,
            1,
            64 * 1024,
            &compute,
            &mut transfer,
        );
        assert!(
            matches!(loc, CacheLocation::RemoteMemory { .. }),
            "expected RemoteMemory placement, got {:?}",
            loc
        );
    }

    #[test]
    fn scores_tiers_via_utility() {
        // Direct unit test of the scoring: a small recompute cost should make
        // the GPU the most attractive tier; a long recompute should make even
        // remote tiers attractive.
        let policy = NetworkAwarePlacement { target_node: 0 };
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();

        // Tiny recompute (4 tokens): GPU wins by a wide margin.
        let gpu_small = policy.utility_for_tier(TierKind::Gpu, 1024, 4, &compute, &transfer);
        let cpu_small = policy.utility_for_tier(TierKind::Cpu, 1024, 4, &compute, &transfer);
        assert!(
            gpu_small > cpu_small,
            "GPU utility ({}) should beat CPU utility ({}) for tiny recompute",
            gpu_small,
            cpu_small,
        );

        // Larger KV + longer recompute: the relative gap narrows but GPU
        // should still be the best (since fetch cost grows with bytes too).
        let gpu_large =
            policy.utility_for_tier(TierKind::Gpu, 64 * 1024 * 1024, 4096, &compute, &transfer);
        let cpu_large =
            policy.utility_for_tier(TierKind::Cpu, 64 * 1024 * 1024, 4096, &compute, &transfer);
        assert!(gpu_large > cpu_large);
    }

    #[test]
    fn short_circuits_tier_whose_total_capacity_cannot_fit_object() {
        // Regression test for the LRU short-circuit: when a tier's total
        // capacity is smaller than the KV being placed, no amount of
        // eviction can make it fit, so the policy must skip the tier
        // *without* destroying its contents. The pre-populated CPU
        // objects must survive the call.
        let cluster = cluster_with_tiny_gpu();
        let mut cache = CacheState::new();
        cache.set_capacity(0, TierKind::Gpu, 0);
        cache.set_capacity(0, TierKind::Cpu, 1_000_000); // 1 MiB total
        cache.set_capacity(0, TierKind::RemoteMemory, 1 << 30);
        cache.set_capacity(0, TierKind::RemoteSsd, 1 << 30);

        // Pre-populate CPU with 3 objects totalling 768 KiB.
        for i in 0u64..3 {
            let obj = KvObject {
                kv_id: i,
                session_id: 100 + i,
                model_id: String::new(),
                prefix_tokens: 100 + i as u32,
                bytes: 256_000,
                location: CacheLocation::Cpu { node_id: 0 },
                last_access_ns: i,
                ref_count: 0,
            };
            cache.insert(obj).unwrap();
        }

        let mut policy = NetworkAwarePlacement { target_node: 0 };
        let mut transfer = AnalyticalTransferModel::rdma_400g();
        let compute = LinearComputeModel::conservative_8b();

        // Try to place a 2 MiB object — larger than the entire CPU tier.
        let (loc, _evicted) = policy
            .place(
                999,
                0,
                1024,
                2_000_000,
                0,
                &cluster,
                &mut cache,
                &compute,
                &mut transfer,
            )
            .expect("placement should fall through to a larger tier");

        // Pre-populated CPU objects must still be there (short-circuit
        // prevented destructive eviction).
        assert!(
            cache.lookup(100, 100).is_some(),
            "session 100 evicted from CPU"
        );
        assert!(
            cache.lookup(101, 101).is_some(),
            "session 101 evicted from CPU"
        );
        assert!(
            cache.lookup(102, 102).is_some(),
            "session 102 evicted from CPU"
        );

        // Placement went to remote (the only tier that could hold 2 MiB).
        assert!(
            matches!(loc, CacheLocation::RemoteMemory { .. }),
            "expected remote placement, got {:?}",
            loc
        );
    }
}
