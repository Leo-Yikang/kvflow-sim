use kvflow_sim::decision::{FetchVsRecompute, KvDecision};
use kvflow_sim::model::{LinearComputeModel, profiles};
use kvflow_sim::transfer::{AnalyticalTransferModel, TransferPath, format_bytes, format_ns};

fn main() -> kvflow_sim::Result<()> {
    let profile = profiles::llama_8b_bf16_gqa();
    let compute = LinearComputeModel::conservative_8b();
    let mut transfer = AnalyticalTransferModel::rdma_400g();

    println!("bandwidth_gbps,reused_tokens,kv_bytes,recompute,fetch,saved,decision");
    for bandwidth_gbps in [50, 100, 200, 400, 800] {
        transfer.remote_memory_bps = bandwidth_gbps * 1_000_000_000;
        for reused_tokens in [512, 1024, 2048, 4096, 8192, 16384] {
            let result = FetchVsRecompute::evaluate(
                &profile,
                &compute,
                &transfer,
                TransferPath::RemoteMemoryToGpu,
                reused_tokens,
                reused_tokens,
            )?;
            let decision = match result.decision {
                KvDecision::Fetch { .. } => "fetch",
                KvDecision::RecomputePrefill => "recompute",
            };
            println!(
                "{},{},{},{},{},{},{}",
                bandwidth_gbps,
                reused_tokens,
                format_bytes(result.kv_bytes),
                format_ns(result.recompute_ns),
                format_ns(result.fetch.duration_ns()),
                format_ns(result.saved_ns().max(0) as u64),
                decision
            );
        }
    }

    Ok(())
}
