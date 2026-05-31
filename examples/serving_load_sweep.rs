use kvflow_sim::cluster::{ClusterTopology, GpuResource, ServingNode, WorkerRole};
use kvflow_sim::model::LinearComputeModel;
use kvflow_sim::serving::{EventDrivenRunner, ServingConfig};
use kvflow_sim::trace::{SyntheticTraceConfig, generate_synthetic_trace};
use kvflow_sim::transfer::format_ns;

fn main() {
    let cluster = ClusterTopology::new(vec![ServingNode::new(
        0,
        0,
        vec![
            GpuResource::new(0, 80_000_000_000, WorkerRole::Prefill),
            GpuResource::new(1, 80_000_000_000, WorkerRole::Prefill),
            GpuResource::new(2, 80_000_000_000, WorkerRole::Decode),
            GpuResource::new(3, 80_000_000_000, WorkerRole::Decode),
        ],
        1,
        1_000_000_000_000,
        10_000_000_000_000,
    )]);

    let compute = LinearComputeModel::conservative_8b();
    let config = ServingConfig {
        prefill_workers: 2,
        decode_workers: 2,
        decode_batch_size: 1,
    };

    println!("inter_arrival_us,completed,throughput_req_s,p50_ttft,p99_ttft,p50_tbt,p99_tbt");

    for inter_arrival_us in [50, 100, 200, 400, 800, 1_600, 3_200] {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 64,
            turns_per_session: 4,
            inter_arrival_ns: inter_arrival_us * 1_000,
            initial_prompt_tokens: 1024,
            tokens_added_per_turn: 256,
            output_tokens: 64,
            ..Default::default()
        });

        let mut runner = EventDrivenRunner::new(config.clone(), cluster.clone(), compute);
        let summary = runner.run_summary(&requests).unwrap();

        println!(
            "{},{},{:.2},{},{},{},{}",
            inter_arrival_us,
            summary.completed_requests,
            summary.throughput_req_s,
            format_ns(summary.ttft.p50_ns),
            format_ns(summary.ttft.p99_ns),
            format_ns(summary.tbt.p50_ns),
            format_ns(summary.tbt.p99_ns),
        );
    }
}
