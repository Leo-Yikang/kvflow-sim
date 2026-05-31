use std::collections::{HashMap, VecDeque};

use crate::cache::{CacheLocation, CacheState, KvObject, TierKind};
use crate::cluster::{ClusterTopology, NodeId, WorkerRole};
use crate::core::Simulator;
use crate::model::{ComputeModel, ModelProfile};
use crate::policy::PlacementPolicy;
use crate::trace::LlmRequest;
use crate::transfer::TransferModel;

use super::event::ServingEvent;
use super::inflight::InFlightRequest;
use super::metrics::ServingSummary;
use super::runner::{RequestResult, ServingConfig};
use super::scheduler::{start_decode, start_prefill};

/// Per-tier hit counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheHitStats {
    pub hits_gpu: u64,
    pub hits_cpu: u64,
    pub hits_remote: u64,
    pub misses: u64,
}

/// Event-driven serving runner with KV-cache awareness.
///
/// On arrival the runner looks up previously-computed KV for the session.
/// - GPU hit  -> prefill only new prompt tokens, then decode.
/// - CPU hit  -> fetch to GPU, prefill new prompt tokens, then decode.
/// - Remote   -> fetch to GPU, prefill new prompt tokens, then decode.
/// - Miss     -> run full prefill, then store the new KV via `placement`.
#[derive(Debug)]
pub struct CacheAwareRunner<C, P, T> {
    sim: Simulator<ServingEvent>,
    cluster: ClusterTopology,
    compute: C,
    transfer: T,
    profile: ModelProfile,
    cache: CacheState,
    placement: P,
    config: ServingConfig,
    in_flight: HashMap<u64, InFlightRequest>,
    prefill_queue: VecDeque<u64>,
    decode_queue: VecDeque<u64>,
    results: Vec<RequestResult>,
    hits: CacheHitStats,
    evicted_bytes: u64,
    recomputed_tokens: u64,
    fetched_bytes: u64,
}

impl<C: ComputeModel, P: PlacementPolicy, T: TransferModel> CacheAwareRunner<C, P, T> {
    pub fn new(
        config: ServingConfig,
        cluster: ClusterTopology,
        compute: C,
        transfer: T,
        profile: ModelProfile,
        placement: P,
    ) -> Self {
        let mut cache = CacheState::new();
        // Initialise cache capacities from cluster topology.
        for node in &cluster.nodes {
            cache.set_capacity(node.node_id, TierKind::Gpu, node.total_hbm_bytes());
            cache.set_capacity(node.node_id, TierKind::Cpu, node.cpu_mem_bytes);
            cache.set_capacity(node.node_id, TierKind::LocalSsd, node.local_ssd_bytes);
            cache.set_capacity(node.node_id, TierKind::RemoteMemory, node.cpu_mem_bytes);
            cache.set_capacity(node.node_id, TierKind::RemoteSsd, node.local_ssd_bytes);
        }

        let config = ServingConfig {
            prefill_workers: config.prefill_workers.max(1),
            decode_workers: config.decode_workers.max(1),
            decode_batch_size: config.decode_batch_size.max(1),
        };

        Self {
            sim: Simulator::new(),
            cluster,
            compute,
            transfer,
            profile,
            cache,
            placement,
            config,
            in_flight: HashMap::new(),
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            results: Vec::new(),
            hits: CacheHitStats::default(),
            evicted_bytes: 0,
            recomputed_tokens: 0,
            fetched_bytes: 0,
        }
    }

    pub fn run(&mut self, requests: &[LlmRequest]) -> Vec<RequestResult> {
        self.clear_state();

        let mut sorted: Vec<LlmRequest> = requests.to_vec();
        sorted.sort_by_key(|r| (r.arrival_ns, r.request_id));

        for req in &sorted {
            self.sim.schedule(
                req.arrival_ns,
                ServingEvent::RequestArrival {
                    request_id: req.request_id,
                },
            );
            self.in_flight
                .insert(req.request_id, InFlightRequest::new(req.clone()));
        }

        while let Some((now_ns, event)) = self.sim.step() {
            self.handle_event(now_ns, event);
        }

        self.results.sort_by_key(|r| (r.arrival_ns, r.request_id));
        std::mem::take(&mut self.results)
    }

    pub fn run_summary(
        &mut self,
        requests: &[LlmRequest],
    ) -> Option<(ServingSummary, CacheHitStats)> {
        let results = self.run(requests);
        let summary = ServingSummary::from_results_with_slos(&results, requests)?;
        Some((summary, self.hits))
    }

    fn clear_state(&mut self) {
        self.sim = Simulator::new();
        self.in_flight.clear();
        self.prefill_queue.clear();
        self.decode_queue.clear();
        self.results.clear();
        self.hits = CacheHitStats::default();
        self.evicted_bytes = 0;
        self.recomputed_tokens = 0;
        self.fetched_bytes = 0;
        // Note: we keep the cache contents across runs (warm cache).
    }

    fn handle_event(&mut self, now_ns: u64, event: ServingEvent) {
        match event {
            ServingEvent::RequestArrival { request_id } => {
                self.on_arrival(now_ns, request_id);
            }
            ServingEvent::PrefillDone {
                request_id,
                node_id,
                gpu_idx,
            } => {
                self.on_prefill_done(now_ns, request_id, node_id, gpu_idx);
            }
            ServingEvent::FetchDone { request_id } => {
                self.on_fetch_done(now_ns, request_id);
            }
            ServingEvent::DecodeDone {
                request_id,
                node_id,
                gpu_idx,
            } => {
                self.on_decode_done(now_ns, request_id, node_id, gpu_idx);
            }
        }
    }

    fn on_arrival(&mut self, now_ns: u64, request_id: u64) {
        let inflight = match self.in_flight.get(&request_id) {
            Some(r) => r,
            None => return,
        };
        let reused = inflight.request.reused_prefix_tokens();

        if reused > 0 {
            let kv_opt = self
                .cache
                .lookup(inflight.request.session_id, reused)
                .cloned();
            if let Some(ref kv) = kv_opt {
                match kv.location {
                    CacheLocation::Gpu { .. } => {
                        self.hits.hits_gpu += 1;
                        self.cache.update_access(kv.kv_id, now_ns);
                        self.start_or_queue_remaining_prefill(now_ns, request_id);
                        return;
                    }
                    CacheLocation::Cpu { .. } => {
                        self.hits.hits_cpu += 1;
                        self.handle_fetch(now_ns, request_id, kv);
                        return;
                    }
                    CacheLocation::RemoteMemory { .. } | CacheLocation::RemoteSsd { .. } => {
                        self.hits.hits_remote += 1;
                        self.handle_fetch(now_ns, request_id, kv);
                        return;
                    }
                    _ => {}
                }
            }
        }

        // Cache miss (or zero reuse).
        self.hits.misses += 1;
        self.recomputed_tokens += inflight.request.prompt_tokens as u64;

        if let Some(inflight) = self.in_flight.get_mut(&request_id) {
            inflight.prefill_tokens = inflight.request.prompt_tokens;
        }
        if let Some((node_id, gpu_idx)) =
            self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Prefill)
        {
            let inflight = self.in_flight.get_mut(&request_id).unwrap();
            start_prefill(
                &mut self.sim,
                &mut self.cluster,
                &self.compute,
                inflight,
                node_id,
                gpu_idx,
                now_ns,
            );
        } else {
            self.prefill_queue.push_back(request_id);
        }
    }

    fn handle_fetch(&mut self, now_ns: u64, request_id: u64, kv: &KvObject) {
        let path = match kv.location {
            CacheLocation::Cpu { .. } => crate::transfer::TransferPath::LocalCpuToGpu,
            CacheLocation::RemoteMemory { .. } => crate::transfer::TransferPath::RemoteMemoryToGpu,
            CacheLocation::RemoteSsd { .. } => crate::transfer::TransferPath::RemoteSsdToGpu,
            _ => {
                // Should not happen for CPU/remote.
                self.decode_queue.push_back(request_id);
                return;
            }
        };

        let estimate = self
            .transfer
            .estimate(now_ns, path, kv.bytes)
            .unwrap_or_else(|err| {
                panic!(
                    "failed to estimate KV fetch for request {} path {:?}: {}",
                    request_id, path, err
                )
            });

        self.fetched_bytes += kv.bytes;
        self.sim
            .schedule(estimate.finish_ns, ServingEvent::FetchDone { request_id });
    }

    fn start_or_queue_remaining_prefill(&mut self, now_ns: u64, request_id: u64) {
        let Some(inflight) = self.in_flight.get_mut(&request_id) else {
            return;
        };
        let tokens_to_prefill = inflight.request.new_prompt_tokens;
        if tokens_to_prefill == 0 {
            if let Some((d_node, d_gpu)) =
                self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Decode)
            {
                start_decode(
                    &mut self.sim,
                    &mut self.cluster,
                    &self.compute,
                    &self.config,
                    inflight,
                    d_node,
                    d_gpu,
                    now_ns,
                );
            } else {
                self.decode_queue.push_back(request_id);
            }
            return;
        }

        inflight.prefill_tokens = tokens_to_prefill;
        self.recomputed_tokens += tokens_to_prefill as u64;

        if let Some((node_id, gpu_idx)) =
            self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Prefill)
        {
            start_prefill(
                &mut self.sim,
                &mut self.cluster,
                &self.compute,
                inflight,
                node_id,
                gpu_idx,
                now_ns,
            );
        } else {
            self.prefill_queue.push_back(request_id);
        }
    }

    fn on_prefill_done(&mut self, now_ns: u64, request_id: u64, node_id: NodeId, gpu_idx: usize) {
        // Release prefill GPU.
        if let Some(node) = self.cluster.node_mut(node_id) {
            node.gpus[gpu_idx].busy_until_ns = now_ns;
        }

        // Store the newly-computed KV in cache.
        if let Some(inflight) = self.in_flight.get(&request_id) {
            let prefix_tokens = inflight.request.prompt_tokens;
            if prefix_tokens > 0 {
                let bytes = self.profile.kv_bytes(prefix_tokens);
                let kv_id = self.cache.alloc_kv_id();
                let (_loc, evicted) = self
                    .placement
                    .place(
                        kv_id,
                        inflight.request.session_id,
                        prefix_tokens,
                        bytes,
                        now_ns,
                        &self.cluster,
                        &mut self.cache,
                    )
                    .unwrap_or_else(|err| {
                        panic!("failed to place KV for request {}: {}", request_id, err)
                    });
                for evicted in &evicted {
                    self.evicted_bytes = self.evicted_bytes.saturating_add(evicted.bytes);
                }
            }
        }

        // Try to start decode for the finished request.
        if let Some(inflight) = self.in_flight.get_mut(&request_id) {
            if let Some((d_node, d_gpu)) =
                self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Decode)
            {
                start_decode(
                    &mut self.sim,
                    &mut self.cluster,
                    &self.compute,
                    &self.config,
                    inflight,
                    d_node,
                    d_gpu,
                    now_ns,
                );
            } else {
                self.decode_queue.push_back(request_id);
            }
        }

        // Unified GPUs may serve either role, so check both queues.
        self.try_assign_prefill(now_ns);
        self.try_assign_decode(now_ns);
    }

    fn on_fetch_done(&mut self, now_ns: u64, request_id: u64) {
        self.start_or_queue_remaining_prefill(now_ns, request_id);
        self.try_assign_decode(now_ns);
        self.try_assign_prefill(now_ns);
    }

    fn on_decode_done(&mut self, now_ns: u64, request_id: u64, node_id: NodeId, gpu_idx: usize) {
        if let Some(node) = self.cluster.node_mut(node_id) {
            node.gpus[gpu_idx].busy_until_ns = now_ns;
        }

        if let Some(inflight) = self.in_flight.remove(&request_id) {
            self.results.push(inflight.into_result());
        }

        // Unified GPUs may serve either role, so check both queues.
        self.try_assign_decode(now_ns);
        self.try_assign_prefill(now_ns);
    }

    fn try_assign_prefill(&mut self, now_ns: u64) {
        while let Some(&req_id) = self.prefill_queue.front() {
            let assignment = self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Prefill);
            if let Some((node_id, gpu_idx)) = assignment {
                self.prefill_queue.pop_front();
                if let Some(inflight) = self.in_flight.get_mut(&req_id) {
                    start_prefill(
                        &mut self.sim,
                        &mut self.cluster,
                        &self.compute,
                        inflight,
                        node_id,
                        gpu_idx,
                        now_ns,
                    );
                }
            } else {
                break;
            }
        }
    }

    fn try_assign_decode(&mut self, now_ns: u64) {
        while let Some(&req_id) = self.decode_queue.front() {
            let assignment = self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Decode);
            if let Some((node_id, gpu_idx)) = assignment {
                self.decode_queue.pop_front();
                if let Some(inflight) = self.in_flight.get_mut(&req_id) {
                    start_decode(
                        &mut self.sim,
                        &mut self.cluster,
                        &self.compute,
                        &self.config,
                        inflight,
                        node_id,
                        gpu_idx,
                        now_ns,
                    );
                }
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{GpuResource, ServingNode, WorkerRole};
    use crate::model::{LinearComputeModel, profiles};
    use crate::policy::{LocalGpuLru, RemoteLru};
    use crate::trace::{SyntheticTraceConfig, generate_synthetic_trace};
    use crate::transfer::AnalyticalTransferModel;

    fn single_node_cluster(prefill_gpus: usize, decode_gpus: usize) -> ClusterTopology {
        let mut gpus = Vec::new();
        for i in 0..prefill_gpus {
            gpus.push(GpuResource::new(
                i as u32,
                80_000_000_000,
                WorkerRole::Prefill,
            ));
        }
        for i in 0..decode_gpus {
            gpus.push(GpuResource::new(
                (prefill_gpus + i) as u32,
                80_000_000_000,
                WorkerRole::Decode,
            ));
        }
        ClusterTopology::new(vec![ServingNode::new(
            0,
            0,
            gpus,
            1,
            1_000_000_000_000,
            10_000_000_000_000,
        )])
    }

    #[test]
    fn cache_aware_with_large_gpu_has_hits() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 2,
            turns_per_session: 3,
            output_tokens: 4,
            initial_prompt_tokens: 512,
            tokens_added_per_turn: 128,
            ..Default::default()
        });
        let profile = profiles::llama_8b_bf16_gqa();
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();
        let cluster = single_node_cluster(1, 1);

        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster,
            compute,
            transfer,
            profile.clone(),
            LocalGpuLru { target_node: 0 },
        );

        let (summary, hits) = runner.run_summary(&requests).unwrap();

        // With a large GPU cache, later turns in a session should hit.
        assert!(hits.hits_gpu > 0 || hits.hits_cpu > 0 || hits.hits_remote > 0);
        assert_eq!(summary.completed_requests, requests.len());
    }

    #[test]
    fn cache_hit_prefills_only_new_prompt_tokens() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 1,
            turns_per_session: 2,
            inter_arrival_ns: 100_000_000,
            initial_prompt_tokens: 100,
            tokens_added_per_turn: 20,
            output_tokens: 4,
            ..Default::default()
        });
        let profile = profiles::llama_8b_bf16_gqa();
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();
        let cluster = single_node_cluster(1, 1);

        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster,
            compute,
            transfer,
            profile,
            LocalGpuLru { target_node: 0 },
        );

        let results = runner.run(&requests);
        let second = results.iter().find(|r| r.request_id == 1).unwrap();
        let expected_prefill_ns = LinearComputeModel::conservative_8b().prefill_time_ns(20);

        assert_eq!(
            second
                .prefill_done_ns
                .saturating_sub(second.prefill_start_ns),
            expected_prefill_ns
        );
    }

    #[test]
    fn remote_lru_places_and_hits_remote_memory() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 1,
            turns_per_session: 2,
            inter_arrival_ns: 100_000_000,
            initial_prompt_tokens: 512,
            tokens_added_per_turn: 128,
            output_tokens: 4,
            ..Default::default()
        });
        let profile = profiles::llama_8b_bf16_gqa();
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();
        let cluster = single_node_cluster(1, 1);

        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster,
            compute,
            transfer,
            profile,
            RemoteLru { target_node: 0 },
        );

        let (summary, hits) = runner.run_summary(&requests).unwrap();

        assert_eq!(summary.completed_requests, requests.len());
        assert_eq!(hits.hits_remote, 1);
    }

    #[test]
    fn cache_aware_with_tiny_gpu_evicts() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 4,
            turns_per_session: 2,
            output_tokens: 4,
            initial_prompt_tokens: 4096,
            tokens_added_per_turn: 512,
            ..Default::default()
        });
        let profile = profiles::llama_8b_bf16_gqa();
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();

        // Tiny GPU: 1 MiB — forces eviction / CPU offload.
        let gpus = vec![GpuResource::new(0, 1_048_576, WorkerRole::Unified)];
        let cluster = ClusterTopology::new(vec![ServingNode::new(
            0,
            0,
            gpus,
            1,
            1_000_000_000_000,
            10_000_000_000_000,
        )]);

        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster,
            compute,
            transfer,
            profile.clone(),
            LocalGpuLru { target_node: 0 },
        );

        let results = runner.run(&requests);
        assert_eq!(
            results.len(),
            requests.len(),
            "expected {} results, got {}",
            requests.len(),
            results.len()
        );
        let summary = ServingSummary::from_results(&results).unwrap();
        assert_eq!(summary.completed_requests, requests.len());
    }
}
