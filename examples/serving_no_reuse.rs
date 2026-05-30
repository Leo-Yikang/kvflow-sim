use kvflow_sim::model::LinearComputeModel;
use kvflow_sim::serving::{NoReuseRunner, ServingConfig};
use kvflow_sim::trace::{SyntheticTraceConfig, generate_synthetic_trace};
use kvflow_sim::transfer::format_ns;

fn main() {
    let requests = generate_synthetic_trace(&SyntheticTraceConfig {
        sessions: 32,
        turns_per_session: 4,
        inter_arrival_ns: 500_000,
        initial_prompt_tokens: 1024,
        tokens_added_per_turn: 512,
        output_tokens: 64,
        ..Default::default()
    });

    let mut runner = NoReuseRunner::new(
        ServingConfig {
            prefill_workers: 4,
            decode_workers: 8,
            decode_batch_size: 1,
        },
        LinearComputeModel::conservative_8b(),
    );
    let summary = runner.run_summary(&requests).unwrap();

    println!("completed_requests: {}", summary.completed_requests);
    println!("total_output_tokens: {}", summary.total_output_tokens);
    println!("makespan: {}", format_ns(summary.makespan_ns));
    println!("throughput_req_s: {:.2}", summary.throughput_req_s);
    println!("throughput_token_s: {:.2}", summary.throughput_token_s);
    print_latency("ttft", summary.ttft);
    print_latency("tbt", summary.tbt);
    print_latency("jct", summary.jct);
}

fn print_latency(name: &str, stats: kvflow_sim::serving::LatencyStats) {
    println!(
        "{name}: p50={} p95={} p99={} mean={}",
        format_ns(stats.p50_ns),
        format_ns(stats.p95_ns),
        format_ns(stats.p99_ns),
        format_ns(stats.mean_ns as u64),
    );
}
