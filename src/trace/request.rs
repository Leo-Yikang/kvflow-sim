use serde::{Deserialize, Serialize};

pub type RequestId = u64;
pub type SessionId = u64;
pub type ModelId = String;

/// A single LLM serving request.
///
/// `prompt_tokens` is the full context visible to the request. `new_prompt_tokens`
/// is the increment since the previous turn in the same session, which is useful
/// for estimating potential KV reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmRequest {
    pub request_id: RequestId,
    pub session_id: SessionId,
    pub turn_id: u32,
    pub arrival_ns: u64,
    pub prompt_tokens: u32,
    pub new_prompt_tokens: u32,
    pub output_tokens: u32,
    pub model_id: ModelId,
    #[serde(default)]
    pub slo_ttft_ns: Option<u64>,
    #[serde(default)]
    pub slo_tbt_ns: Option<u64>,
}

impl LlmRequest {
    pub fn reused_prefix_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_sub(self.new_prompt_tokens)
    }
}
