use std::collections::HashMap;

use crate::trace::LlmRequest;

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
    pub slo_violations_ttft: usize,
    pub slo_violations_tbt: usize,
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
            slo_violations_ttft: 0,
            slo_violations_tbt: 0,
        })
    }

    pub fn from_results_with_slos(
        results: &[RequestResult],
        requests: &[LlmRequest],
    ) -> Option<Self> {
        let mut summary = Self::from_results(results)?;
        summary.apply_slos(results, requests)?;
        Some(summary)
    }

    pub fn apply_slos(&mut self, results: &[RequestResult], requests: &[LlmRequest]) -> Option<()> {
        let results_by_id: HashMap<u64, &RequestResult> = results
            .iter()
            .map(|result| (result.request_id, result))
            .collect();

        self.slo_violations_ttft = 0;
        self.slo_violations_tbt = 0;

        for req in requests {
            let result = results_by_id.get(&req.request_id)?;
            if let Some(slo) = req.slo_ttft_ns {
                if result.ttft_ns() > slo {
                    self.slo_violations_ttft = self.slo_violations_ttft.saturating_add(1);
                }
            }
            if let Some(slo) = req.slo_tbt_ns {
                if result.mean_tbt_ns() > slo {
                    self.slo_violations_tbt = self.slo_violations_tbt.saturating_add(1);
                }
            }
        }

        Some(())
    }
}

fn percentile(sorted_values: &[u64], q: f64) -> u64 {
    debug_assert!(!sorted_values.is_empty());
    let idx = ((sorted_values.len() - 1) as f64 * q).ceil() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}
