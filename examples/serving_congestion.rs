use kvflow_sim::cluster::{ClusterTopology, GpuResource, ServingNode, WorkerRole};
use kvflow_sim::model::{LinearComputeModel, profiles};
use kvflow_sim::policy::LocalGpuLru;
use kvflow_sim::serving::{CacheAwareRunner, ServingConfig};
use kvflow_sim::trace::{SyntheticTraceConfig, generate_synthetic_trace};
use kvflow_sim::transfer::format_ns;

fn main() {
    let profile = profiles::llama_8b_bf16_gqa();
    let compute = LinearComputeModel::conservative_8b();

    let cluster = ClusterTopology::new(vec![ServingNode::new(
        0,
        0,
        vec![GpuResource::new(0, 80_000_000_000, WorkerRole::Unified)],
        1,
        1_000_000_000_000,
        10_000_000_000_000,
    )]);

    let requests = generate_synthetic_trace(&SyntheticTraceConfig {
        sessions: 8,
        turns_per_session: 4,
        inter_arrival_ns: 200_000,
        initial_prompt_tokens: 4096,
        tokens_added_per_turn: 1024,
        output_tokens: 32,
        ..Default::default()
    });

    println!("model,completed,p50_ttft,p99_ttft,p50_tbt,p99_tbt");

    // Analytical model (no queueing).
    {
        let transfer = kvflow_sim::transfer::AnalyticalTransferModel::rdma_400g();
        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster.clone(),
            compute,
            transfer,
            profile.clone(),
            LocalGpuLru { target_node: 0 },
        );
        let (summary, _hits) = runner.run_summary(&requests).unwrap();
        println!(
            "Analytical,{},{},{},{},{}",
            summary.completed_requests,
            format_ns(summary.ttft.p50_ns),
            format_ns(summary.ttft.p99_ns),
            format_ns(summary.tbt.p50_ns),
            format_ns(summary.tbt.p99_ns),
        );
    }

    // Queued model (NIC serialises remote fetches).
    {
        let transfer = kvflow_sim::transfer::QueuedTransferModel::rdma_400g();
        let mut runner = CacheAwareRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            cluster.clone(),
            compute,
            transfer,
            profile.clone(),
            LocalGpuLru { target_node: 0 },
        );
        let (summary, _hits) = runner.run_summary(&requests).unwrap();
        println!(
            "Queued,{},{},{},{},{}",
            summary.completed_requests,
            format_ns(summary.ttft.p50_ns),
            format_ns(summary.ttft.p99_ns),
            format_ns(summary.tbt.p50_ns),
            format_ns(summary.tbt.p99_ns),
        );
    }
}
