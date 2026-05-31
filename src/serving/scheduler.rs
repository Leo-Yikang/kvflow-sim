use std::collections::{HashMap, VecDeque};

use crate::cluster::{ClusterTopology, NodeId, WorkerRole};
use crate::core::Simulator;
use crate::model::ComputeModel;
use crate::trace::LlmRequest;

use super::event::ServingEvent;
use super::inflight::InFlightRequest;
use super::metrics::ServingSummary;
use super::runner::{RequestResult, ServingConfig};

/// An event-driven serving runner that models prefill/decode workers as
/// discrete resources with FCFS queues.
///
/// This is functionally equivalent to `NoReuseRunner` but built on an
/// explicit event loop, making it easier to extend with KV-cache lookup,
/// transfer delays, and policy decisions.
#[derive(Debug)]
pub struct EventDrivenRunner<C> {
    sim: Simulator<ServingEvent>,
    cluster: ClusterTopology,
    compute: C,
    config: ServingConfig,
    in_flight: HashMap<u64, InFlightRequest>,
    prefill_queue: VecDeque<u64>,
    decode_queue: VecDeque<u64>,
    results: Vec<RequestResult>,
}

impl<C: ComputeModel> EventDrivenRunner<C> {
    pub fn new(config: ServingConfig, cluster: ClusterTopology, compute: C) -> Self {
        let config = ServingConfig {
            prefill_workers: config.prefill_workers.max(1),
            decode_workers: config.decode_workers.max(1),
            decode_batch_size: config.decode_batch_size.max(1),
        };
        Self {
            sim: Simulator::new(),
            cluster,
            compute,
            config,
            in_flight: HashMap::new(),
            prefill_queue: VecDeque::new(),
            decode_queue: VecDeque::new(),
            results: Vec::new(),
        }
    }

    /// Run a batch of requests through the simulator and return per-request
    /// results ordered by arrival time.
    pub fn run(&mut self, requests: &[LlmRequest]) -> Vec<RequestResult> {
        self.clear_state();

        // Sort by arrival time and schedule all arrivals.
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

        // Event loop.
        while let Some((now_ns, event)) = self.sim.step() {
            self.handle_event(now_ns, event);
        }

        // Results are produced in completion order; re-sort by arrival.
        self.results.sort_by_key(|r| (r.arrival_ns, r.request_id));
        std::mem::take(&mut self.results)
    }

    /// Run the simulator and produce a summary, including SLO violations when
    /// the original requests carry SLO thresholds.
    pub fn run_summary(&mut self, requests: &[LlmRequest]) -> Option<ServingSummary> {
        let results = self.run(requests);
        ServingSummary::from_results_with_slos(&results, requests)
    }

    fn clear_state(&mut self) {
        self.sim = Simulator::new();
        self.in_flight.clear();
        self.prefill_queue.clear();
        self.decode_queue.clear();
        self.results.clear();
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
            ServingEvent::FetchDone { .. } => {
                panic!("EventDrivenRunner does not support KV fetch events; use CacheAwareRunner");
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
        if let Some((node_id, gpu_idx)) =
            self.cluster.find_idle_gpu_for(now_ns, WorkerRole::Prefill)
        {
            if let Some(inflight) = self.in_flight.get_mut(&request_id) {
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
            self.prefill_queue.push_back(request_id);
        }
    }

    fn on_prefill_done(&mut self, now_ns: u64, request_id: u64, node_id: NodeId, gpu_idx: usize) {
        // Release the prefill GPU.
        if let Some(node) = self.cluster.node_mut(node_id) {
            node.gpus[gpu_idx].busy_until_ns = now_ns;
        }

        // Try to start decode for the request that just finished prefill.
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

    fn on_decode_done(&mut self, now_ns: u64, request_id: u64, node_id: NodeId, gpu_idx: usize) {
        // Release the decode GPU.
        if let Some(node) = self.cluster.node_mut(node_id) {
            node.gpus[gpu_idx].busy_until_ns = now_ns;
        }

        // Finalize the request.
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

pub(crate) fn start_prefill<C: ComputeModel>(
    sim: &mut Simulator<ServingEvent>,
    cluster: &mut ClusterTopology,
    compute: &C,
    inflight: &mut InFlightRequest,
    node_id: NodeId,
    gpu_idx: usize,
    now_ns: u64,
) {
    let prompt_tokens = if inflight.prefill_tokens == 0 {
        inflight.request.prompt_tokens
    } else {
        inflight.prefill_tokens
    };
    let duration = compute.prefill_time_ns(prompt_tokens);
    let done_ns = now_ns.saturating_add(duration);

    inflight.prefill_start_ns = Some(now_ns);
    inflight.prefill_done_ns = Some(done_ns);
    inflight.prefill_node = Some(node_id);
    inflight.prefill_gpu = Some(gpu_idx);

    if let Some(node) = cluster.node_mut(node_id) {
        node.gpus[gpu_idx].busy_until_ns = done_ns;
    }

    sim.schedule(
        done_ns,
        ServingEvent::PrefillDone {
            request_id: inflight.request.request_id,
            node_id,
            gpu_idx,
        },
    );
}

pub(crate) fn start_decode<C: ComputeModel>(
    sim: &mut Simulator<ServingEvent>,
    cluster: &mut ClusterTopology,
    compute: &C,
    config: &ServingConfig,
    inflight: &mut InFlightRequest,
    node_id: NodeId,
    gpu_idx: usize,
    now_ns: u64,
) {
    let step_ns =
        compute.decode_step_time_ns(config.decode_batch_size, inflight.request.prompt_tokens);
    let total_decode_ns = step_ns.saturating_mul(inflight.request.output_tokens as u64);
    let finish_ns = now_ns.saturating_add(total_decode_ns);
    let first_token_ns = now_ns.saturating_add(step_ns);

    inflight.decode_start_ns = Some(now_ns);
    inflight.first_token_ns = Some(first_token_ns);
    inflight.finish_ns = Some(finish_ns);
    inflight.decode_node = Some(node_id);
    inflight.decode_gpu = Some(gpu_idx);

    if let Some(node) = cluster.node_mut(node_id) {
        node.gpus[gpu_idx].busy_until_ns = finish_ns;
    }

    sim.schedule(
        finish_ns,
        ServingEvent::DecodeDone {
            request_id: inflight.request.request_id,
            node_id,
            gpu_idx,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{GpuResource, ServingNode, WorkerRole};
    use crate::model::LinearComputeModel;
    use crate::trace::{SyntheticTraceConfig, generate_synthetic_trace};

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
    fn event_driven_matches_no_reuse_for_small_trace() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 2,
            turns_per_session: 2,
            output_tokens: 4,
            ..Default::default()
        });
        let compute = LinearComputeModel::conservative_8b();
        let config = ServingConfig {
            prefill_workers: 1,
            decode_workers: 1,
            decode_batch_size: 1,
        };

        // Run with the old runner.
        let old_results =
            super::super::runner::NoReuseRunner::new(config.clone(), compute).run(&requests);

        // Run with the event-driven runner.
        let cluster = single_node_cluster(1, 1);
        let new_results = EventDrivenRunner::new(config, cluster, compute).run(&requests);

        assert_eq!(old_results.len(), new_results.len());
        for (old, new) in old_results.iter().zip(new_results.iter()) {
            assert_eq!(old.request_id, new.request_id);
            assert_eq!(old.finish_ns, new.finish_ns);
            assert_eq!(old.ttft_ns(), new.ttft_ns());
            assert_eq!(old.jct_ns(), new.jct_ns());
        }
    }

    #[test]
    fn event_driven_produces_summary() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 4,
            turns_per_session: 2,
            output_tokens: 8,
            ..Default::default()
        });
        let cluster = single_node_cluster(2, 2);
        let mut runner = EventDrivenRunner::new(
            ServingConfig {
                prefill_workers: 2,
                decode_workers: 2,
                decode_batch_size: 1,
            },
            cluster,
            LinearComputeModel::conservative_8b(),
        );

        let summary = runner.run_summary(&requests).unwrap();
        assert_eq!(summary.completed_requests, requests.len());
        assert!(summary.ttft.p50_ns > 0);
        assert!(summary.jct.p99_ns >= summary.ttft.p99_ns);
    }
}
