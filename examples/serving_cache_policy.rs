use kvflow_sim::cluster::{ClusterTopology, GpuResource, ServingNode, WorkerRole};
use kvflow_sim::model::{LinearComputeModel, profiles};
use kvflow_sim::policy::{CpuOffloadLru, LocalGpuLru};
use kvflow_sim::serving::{CacheAwareRunner, ServingConfig};
use kvflow_sim::trace::{SyntheticTraceConfig, generate_synthetic_trace};
use kvflow_sim::transfer::AnalyticalTransferModel;
use kvflow_sim::transfer::format_ns;

fn main() {
    let profile = profiles::llama_8b_bf16_gqa();
    let compute = LinearComputeModel::conservative_8b();
    let transfer = AnalyticalTransferModel::rdma_400g();

    // Vary GPU cache capacity.
    let gpu_hbm_bytes = [1_000_000_000, 10_000_000_000, 80_000_000_000];

    println!("policy,gpu_hmb_gib,completed,hits_gpu,hits_cpu,misses,p50_ttft,p99_ttft");

    for hbm in gpu_hbm_bytes {
        let cluster = ClusterTopology::new(vec![ServingNode::new(
            0,
            0,
            vec![GpuResource::new(0, hbm, WorkerRole::Unified)],
            1,
            1_000_000_000_000,
            10_000_000_000_000,
        )]);

        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 16,
            turns_per_session: 4,
            inter_arrival_ns: 500_000,
            initial_prompt_tokens: 2048,
            tokens_added_per_turn: 512,
            output_tokens: 64,
            ..Default::default()
        });

        // NoReuse baseline (via EventDrivenRunner with the same cluster).
        {
            let mut runner = kvflow_sim::serving::EventDrivenRunner::new(
                ServingConfig {
                    prefill_workers: 1,
                    decode_workers: 1,
                    decode_batch_size: 1,
                },
                cluster.clone(),
                compute,
            );
            let summary = runner.run_summary(&requests).unwrap();
            println!(
                "NoReuse,{:.2},{},{},{},{},{},{}",
                hbm as f64 / (1024.0 * 1024.0 * 1024.0),
                summary.completed_requests,
                0,
                0,
                requests.len(),
                format_ns(summary.ttft.p50_ns),
                format_ns(summary.ttft.p99_ns),
            );
        }

        // LocalGpuLru.
        {
            let mut runner = CacheAwareRunner::new(
                ServingConfig {
                    prefill_workers: 1,
                    decode_workers: 1,
                    decode_batch_size: 1,
                },
                cluster.clone(),
                compute,
                transfer.clone(),
                profile.clone(),
                LocalGpuLru { target_node: 0 },
            );
            let (summary, hits) = runner.run_summary(&requests).unwrap();
            println!(
                "LocalGpuLru,{:.2},{},{},{},{},{},{}",
                hbm as f64 / (1024.0 * 1024.0 * 1024.0),
                summary.completed_requests,
                hits.hits_gpu,
                hits.hits_cpu,
                hits.misses,
                format_ns(summary.ttft.p50_ns),
                format_ns(summary.ttft.p99_ns),
            );
        }

        // CpuOffloadLru.
        {
            let mut runner = CacheAwareRunner::new(
                ServingConfig {
                    prefill_workers: 1,
                    decode_workers: 1,
                    decode_batch_size: 1,
                },
                cluster.clone(),
                compute,
                transfer.clone(),
                profile.clone(),
                CpuOffloadLru { target_node: 0 },
            );
            let (summary, hits) = runner.run_summary(&requests).unwrap();
            println!(
                "CpuOffloadLru,{:.2},{},{},{},{},{},{}",
                hbm as f64 / (1024.0 * 1024.0 * 1024.0),
                summary.completed_requests,
                hits.hits_gpu,
                hits.hits_cpu,
                hits.misses,
                format_ns(summary.ttft.p50_ns),
                format_ns(summary.ttft.p99_ns),
            );
        }
    }
}
