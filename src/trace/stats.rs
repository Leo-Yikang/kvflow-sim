use std::collections::BTreeSet;

use super::LlmRequest;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldStats {
    pub min: u32,
    pub max: u32,
    pub mean: f64,
    pub p50: u32,
    pub p95: u32,
    pub p99: u32,
}

impl FieldStats {
    pub fn from_values(values: &mut [u32]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        let sum: u64 = values.iter().map(|&v| v as u64).sum();
        Some(Self {
            min: values[0],
            max: *values.last().unwrap(),
            mean: sum as f64 / values.len() as f64,
            p50: percentile(values, 0.50),
            p95: percentile(values, 0.95),
            p99: percentile(values, 0.99),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraceStats {
    pub requests: usize,
    pub sessions: usize,
    pub turns: usize,
    pub prompt_tokens: FieldStats,
    pub new_prompt_tokens: FieldStats,
    pub reused_prefix_tokens: FieldStats,
    pub output_tokens: FieldStats,
    pub reuse_ratio_mean: f64,
}

impl TraceStats {
    pub fn from_requests(requests: &[LlmRequest]) -> Option<Self> {
        if requests.is_empty() {
            return None;
        }

        let mut sessions = BTreeSet::new();
        let mut turns = BTreeSet::new();
        let mut prompt_tokens = Vec::with_capacity(requests.len());
        let mut new_prompt_tokens = Vec::with_capacity(requests.len());
        let mut reused_prefix_tokens = Vec::with_capacity(requests.len());
        let mut output_tokens = Vec::with_capacity(requests.len());
        let mut reuse_ratio_sum = 0.0;

        for req in requests {
            sessions.insert(req.session_id);
            turns.insert((req.session_id, req.turn_id));
            prompt_tokens.push(req.prompt_tokens);
            new_prompt_tokens.push(req.new_prompt_tokens);
            let reused = req.reused_prefix_tokens();
            reused_prefix_tokens.push(reused);
            output_tokens.push(req.output_tokens);
            if req.prompt_tokens > 0 {
                reuse_ratio_sum += reused as f64 / req.prompt_tokens as f64;
            }
        }

        Some(Self {
            requests: requests.len(),
            sessions: sessions.len(),
            turns: turns.len(),
            prompt_tokens: FieldStats::from_values(&mut prompt_tokens).unwrap(),
            new_prompt_tokens: FieldStats::from_values(&mut new_prompt_tokens).unwrap(),
            reused_prefix_tokens: FieldStats::from_values(&mut reused_prefix_tokens).unwrap(),
            output_tokens: FieldStats::from_values(&mut output_tokens).unwrap(),
            reuse_ratio_mean: reuse_ratio_sum / requests.len() as f64,
        })
    }
}

fn percentile(sorted_values: &[u32], q: f64) -> u32 {
    debug_assert!(!sorted_values.is_empty());
    let idx = ((sorted_values.len() - 1) as f64 * q).ceil() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{SyntheticTraceConfig, generate_synthetic_trace};

    #[test]
    fn stats_capture_multiturn_reuse() {
        let requests = generate_synthetic_trace(&SyntheticTraceConfig {
            sessions: 2,
            turns_per_session: 3,
            initial_prompt_tokens: 100,
            tokens_added_per_turn: 50,
            output_tokens: 10,
            ..Default::default()
        });

        let stats = TraceStats::from_requests(&requests).unwrap();

        assert_eq!(stats.requests, 6);
        assert_eq!(stats.sessions, 2);
        assert_eq!(stats.prompt_tokens.max, 200);
        assert_eq!(stats.reused_prefix_tokens.min, 0);
        assert_eq!(stats.reused_prefix_tokens.max, 150);
        assert!(stats.reuse_ratio_mean > 0.0);
    }
}
