use kvflow_sim::model::{ComputeModel, LinearComputeModel, profiles};
use kvflow_sim::transfer::{AnalyticalTransferModel, TransferPath, format_bytes, format_ns};

fn main() -> kvflow_sim::Result<()> {
    let profile = profiles::llama_8b_bf16_gqa();
    let compute = LinearComputeModel::conservative_8b();
    let transfer = AnalyticalTransferModel::rdma_400g();
    let contexts = [512, 1024, 2048, 4096, 8192, 16384, 32768];

    println!("model={}", profile.model_id);
    println!(
        "tokens,kv_bytes,recompute_prefill,local_cpu_fetch,remote_rdma_fetch,remote_ssd_fetch,winner"
    );

    for tokens in contexts {
        let kv_bytes = profile.kv_bytes(tokens);
        let recompute = compute.prefill_time_ns(tokens);
        let local_cpu = transfer
            .estimate(0, TransferPath::LocalCpuToGpu, kv_bytes)?
            .duration_ns();
        let remote = transfer
            .estimate(0, TransferPath::RemoteMemoryToGpu, kv_bytes)?
            .duration_ns();
        let remote_ssd = transfer
            .estimate(0, TransferPath::RemoteSsdToGpu, kv_bytes)?
            .duration_ns();
        let winner = [
            ("recompute", recompute),
            ("local_cpu", local_cpu),
            ("remote_rdma", remote),
            ("remote_ssd", remote_ssd),
        ]
        .into_iter()
        .min_by_key(|(_, latency)| *latency)
        .map(|(name, _)| name)
        .unwrap();

        println!(
            "{},{},{},{},{},{},{}",
            tokens,
            format_bytes(kv_bytes),
            format_ns(recompute),
            format_ns(local_cpu),
            format_ns(remote),
            format_ns(remote_ssd),
            winner
        );
    }

    Ok(())
}
