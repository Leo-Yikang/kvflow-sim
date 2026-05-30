use super::LlmRequest;

#[derive(Debug, Clone)]
pub struct SyntheticTraceConfig {
    pub sessions: u64,
    pub turns_per_session: u32,
    pub start_ns: u64,
    pub inter_arrival_ns: u64,
    pub initial_prompt_tokens: u32,
    pub tokens_added_per_turn: u32,
    pub output_tokens: u32,
    pub model_id: String,
}

impl Default for SyntheticTraceConfig {
    fn default() -> Self {
        Self {
            sessions: 16,
            turns_per_session: 4,
            start_ns: 0,
            inter_arrival_ns: 1_000_000,
            initial_prompt_tokens: 1024,
            tokens_added_per_turn: 256,
            output_tokens: 128,
            model_id: "llama-8b".to_string(),
        }
    }
}

pub fn generate_synthetic_trace(config: &SyntheticTraceConfig) -> Vec<LlmRequest> {
    let mut requests =
        Vec::with_capacity((config.sessions * config.turns_per_session as u64) as usize);
    let mut request_id = 0_u64;
    let mut arrival_ns = config.start_ns;

    for turn_id in 0..config.turns_per_session {
        for session_id in 0..config.sessions {
            let prompt_tokens = config
                .initial_prompt_tokens
                .saturating_add(turn_id.saturating_mul(config.tokens_added_per_turn));
            let new_prompt_tokens = if turn_id == 0 {
                prompt_tokens
            } else {
                config.tokens_added_per_turn
            };

            requests.push(LlmRequest {
                request_id,
                session_id,
                turn_id,
                arrival_ns,
                prompt_tokens,
                new_prompt_tokens,
                output_tokens: config.output_tokens,
                model_id: config.model_id.clone(),
                slo_ttft_ns: None,
                slo_tbt_ns: None,
            });

            request_id += 1;
            arrival_ns = arrival_ns.saturating_add(config.inter_arrival_ns);
        }
    }

    requests
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_trace_has_reused_prefix_after_first_turn() {
        let cfg = SyntheticTraceConfig {
            sessions: 1,
            turns_per_session: 3,
            initial_prompt_tokens: 100,
            tokens_added_per_turn: 20,
            ..Default::default()
        };

        let trace = generate_synthetic_trace(&cfg);

        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].reused_prefix_tokens(), 0);
        assert_eq!(trace[1].reused_prefix_tokens(), 100);
        assert_eq!(trace[2].reused_prefix_tokens(), 120);
    }
}
