use super::runner::RequestResult;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatencyStats {
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

impl LatencyStats {
    pub fn from_values(values: &mut [u64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let sum: u128 = values.iter().map(|&v| v as u128).sum();
        Some(Self {
            min_ns: values[0],
            max_ns: *values.last().unwrap(),
            mean_ns: sum as f64 / values.len() as f64,
            p50_ns: percentile(values, 0.50),
            p95_ns: percentile(values, 0.95),
            p99_ns: percentile(values, 0.99),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServingSummary {
    pub completed_requests: usize,
    pub total_output_tokens: u64,
    pub makespan_ns: u64,
    pub throughput_req_s: f64,
    pub throughput_token_s: f64,
    pub ttft: LatencyStats,
    pub tbt: LatencyStats,
    pub jct: LatencyStats,
}

impl ServingSummary {
    pub fn from_results(results: &[RequestResult]) -> Option<Self> {
        if results.is_empty() {
            return None;
        }

        let start_ns = results.iter().map(|r| r.arrival_ns).min().unwrap();
        let finish_ns = results.iter().map(|r| r.finish_ns).max().unwrap();
        let makespan_ns = finish_ns.saturating_sub(start_ns).max(1);
        let total_output_tokens: u64 = results.iter().map(|r| r.output_tokens as u64).sum();
        let mut ttft: Vec<u64> = results.iter().map(RequestResult::ttft_ns).collect();
        let mut tbt: Vec<u64> = results.iter().map(RequestResult::mean_tbt_ns).collect();
        let mut jct: Vec<u64> = results.iter().map(RequestResult::jct_ns).collect();

        Some(Self {
            completed_requests: results.len(),
            total_output_tokens,
            makespan_ns,
            throughput_req_s: results.len() as f64 * 1e9 / makespan_ns as f64,
            throughput_token_s: total_output_tokens as f64 * 1e9 / makespan_ns as f64,
            ttft: LatencyStats::from_values(&mut ttft).unwrap(),
            tbt: LatencyStats::from_values(&mut tbt).unwrap(),
            jct: LatencyStats::from_values(&mut jct).unwrap(),
        })
    }
}

fn percentile(sorted_values: &[u64], q: f64) -> u64 {
    debug_assert!(!sorted_values.is_empty());
    let idx = ((sorted_values.len() - 1) as f64 * q).ceil() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}
