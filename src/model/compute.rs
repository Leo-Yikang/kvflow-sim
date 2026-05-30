/// Compute latency model for prefill and decode.
pub trait ComputeModel {
    fn prefill_time_ns(&self, prompt_tokens: u32) -> u64;
    fn decode_step_time_ns(&self, active_sequences: u32, context_tokens: u32) -> u64;

    fn decode_total_time_ns(
        &self,
        active_sequences: u32,
        context_tokens: u32,
        output_tokens: u32,
    ) -> u64 {
        self.decode_step_time_ns(active_sequences, context_tokens)
            .saturating_mul(output_tokens as u64)
    }
}

/// A deliberately simple model for early break-even studies.
#[derive(Debug, Clone, Copy)]
pub struct LinearComputeModel {
    pub prefill_base_ns: u64,
    pub prefill_per_token_ns: u64,
    pub decode_base_ns: u64,
    pub decode_per_sequence_ns: u64,
    pub decode_per_1k_context_ns: u64,
}

impl LinearComputeModel {
    pub fn conservative_8b() -> Self {
        Self {
            prefill_base_ns: 100_000,
            prefill_per_token_ns: 3_500,
            decode_base_ns: 600_000,
            decode_per_sequence_ns: 35_000,
            decode_per_1k_context_ns: 12_000,
        }
    }
}

impl ComputeModel for LinearComputeModel {
    fn prefill_time_ns(&self, prompt_tokens: u32) -> u64 {
        self.prefill_base_ns.saturating_add(
            self.prefill_per_token_ns
                .saturating_mul(prompt_tokens as u64),
        )
    }

    fn decode_step_time_ns(&self, active_sequences: u32, context_tokens: u32) -> u64 {
        let context_k = context_tokens.div_ceil(1024) as u64;
        self.decode_base_ns
            .saturating_add(
                self.decode_per_sequence_ns
                    .saturating_mul(active_sequences as u64),
            )
            .saturating_add(self.decode_per_1k_context_ns.saturating_mul(context_k))
    }
}
