use std::env;

use kvflow_sim::trace::{SyntheticTraceConfig, TraceStats, generate_synthetic_trace, read_jsonl};

fn main() -> kvflow_sim::Result<()> {
    let requests = match env::args().nth(1) {
        Some(path) => read_jsonl(path)?,
        None => generate_synthetic_trace(&SyntheticTraceConfig::default()),
    };

    let Some(stats) = TraceStats::from_requests(&requests) else {
        println!("empty trace");
        return Ok(());
    };

    println!("requests: {}", stats.requests);
    println!("sessions: {}", stats.sessions);
    println!("turns: {}", stats.turns);
    println!("reuse_ratio_mean: {:.3}", stats.reuse_ratio_mean);
    print_field("prompt_tokens", stats.prompt_tokens);
    print_field("new_prompt_tokens", stats.new_prompt_tokens);
    print_field("reused_prefix_tokens", stats.reused_prefix_tokens);
    print_field("output_tokens", stats.output_tokens);

    Ok(())
}

fn print_field(name: &str, s: kvflow_sim::trace::FieldStats) {
    println!(
        "{name}: min={} p50={} p95={} p99={} max={} mean={:.1}",
        s.min, s.p50, s.p95, s.p99, s.max, s.mean
    );
}
