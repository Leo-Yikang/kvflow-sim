use crate::model::ComputeModel;
use crate::serving::ServingSummary;
use crate::trace::LlmRequest;

#[derive(Debug, Clone, Copy)]
pub struct ServingConfig {
    pub prefill_workers: usize,
    pub decode_workers: usize,
    pub decode_batch_size: u32,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            prefill_workers: 4,
            decode_workers: 4,
            decode_batch_size: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestResult {
    pub request_id: u64,
    pub session_id: u64,
    pub turn_id: u32,
    pub arrival_ns: u64,
    pub prefill_start_ns: u64,
    pub prefill_done_ns: u64,
    pub decode_start_ns: u64,
    pub first_token_ns: u64,
    pub finish_ns: u64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
}

impl RequestResult {
    pub fn ttft_ns(&self) -> u64 {
        self.first_token_ns.saturating_sub(self.arrival_ns)
    }

    pub fn jct_ns(&self) -> u64 {
        self.finish_ns.saturating_sub(self.arrival_ns)
    }

    pub fn mean_tbt_ns(&self) -> u64 {
        if self.output_tokens <= 1 {
            return 0;
        }
        self.finish_ns
            .saturating_sub(self.first_token_ns)
            .checked_div(self.output_tokens.saturating_sub(1) as u64)
            .unwrap_or(0)
    }
}

/// Baseline runner: every request recomputes full prefill and then decodes.
///
/// This is intentionally simple. It gives future KV-cache policies a stable
/// baseline for TTFT, TBT, JCT, and throughput.
pub struct NoReuseRunner<C> {
    config: ServingConfig,
    compute: C,
    prefill_busy_until: Vec<u64>,
    decode_busy_until: Vec<u64>,
}

impl<C: ComputeModel> NoReuseRunner<C> {
    pub fn new(config: ServingConfig, compute: C) -> Self {
        let prefill_workers = config.prefill_workers.max(1);
        let decode_workers = config.decode_workers.max(1);
        Self {
            config: ServingConfig {
                prefill_workers,
                decode_workers,
                decode_batch_size: config.decode_batch_size.max(1),
            },
            compute,
            prefill_busy_until: vec![0; prefill_workers],
            decode_busy_until: vec![0; decode_workers],
        }
    }

    pub fn run(&mut self, requests: &[LlmRequest]) -> Vec<RequestResult> {
        let mut sorted = requests.to_vec();
        sorted.sort_by_key(|r| (r.arrival_ns, r.request_id));
        sorted.iter().map(|req| self.run_one(req)).collect()
    }

    pub fn run_summary(&mut self, requests: &[LlmRequest]) -> Option<ServingSummary> {
        let results = self.run(requests);
        ServingSummary::from_results(&results)
    }

    fn run_one(&mut self, req: &LlmRequest) -> RequestResult {
        let prefill_idx = earliest_worker(&self.prefill_busy_until);
        let prefill_start_ns = req.arrival_ns.max(self.prefill_busy_until[prefill_idx]);
        let prefill_done_ns =
            prefill_start_ns.saturating_add(self.compute.prefill_time_ns(req.prompt_tokens));
        self.prefill_busy_until[prefill_idx] = prefill_done_ns;

        let decode_idx = earliest_worker(&self.decode_busy_until);
        let decode_start_ns = prefill_done_ns.max(self.decode_busy_until[decode_idx]);
        let decode_step_ns = self
            .compute
            .decode_step_time_ns(self.config.decode_batch_size, req.prompt_tokens);
        let first_token_ns = decode_start_ns.saturating_add(decode_step_ns);
        let finish_ns =
            decode_start_ns.saturating_add(decode_step_ns.saturating_mul(req.output_tokens as u64));
        self.decode_busy_until[decode_idx] = finish_ns;

        RequestResult {
            request_id: req.request_id,
            session_id: req.session_id,
            turn_id: req.turn_id,
            arrival_ns: req.arrival_ns,
            prefill_start_ns,
            prefill_done_ns,
            decode_start_ns,
            first_token_ns,
            finish_ns,
            prompt_tokens: req.prompt_tokens,
            output_tokens: req.output_tokens,
        }
    }
}

fn earliest_worker(workers: &[u64]) -> usize {
    workers
        .iter()
        .enumerate()
        .min_by_key(|(_, busy_until)| *busy_until)
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LinearComputeModel;
    use crate::trace::{SyntheticTraceConfig, generate_synthetic_trace};

    #[test]
    fn no_reuse_runner_completes_requests() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 2,
            turns_per_session: 2,
            output_tokens: 4,
            ..Default::default()
        });
        let mut runner = NoReuseRunner::new(
            ServingConfig {
                prefill_workers: 1,
                decode_workers: 1,
                decode_batch_size: 1,
            },
            LinearComputeModel::conservative_8b(),
        );

        let results = runner.run(&requests);
        let summary = ServingSummary::from_results(&results).unwrap();

        assert_eq!(results.len(), 4);
        assert_eq!(summary.completed_requests, 4);
        assert_eq!(summary.total_output_tokens, 16);
        assert!(summary.ttft.p50_ns > 0);
        assert!(summary.jct.p99_ns >= summary.ttft.p99_ns);
    }
}
