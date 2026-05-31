use crate::cluster::NodeId;
use crate::trace::LlmRequest;

use super::runner::RequestResult;

/// State of a request while it is being processed by the simulator.
#[derive(Debug)]
pub(crate) struct InFlightRequest {
    pub request: LlmRequest,
    pub prefill_start_ns: Option<u64>,
    pub prefill_done_ns: Option<u64>,
    pub decode_start_ns: Option<u64>,
    pub first_token_ns: Option<u64>,
    pub finish_ns: Option<u64>,
    pub prefill_node: Option<NodeId>,
    pub prefill_gpu: Option<usize>,
    pub decode_node: Option<NodeId>,
    pub decode_gpu: Option<usize>,
    pub prefill_tokens: u32,
}

impl InFlightRequest {
    pub fn new(request: LlmRequest) -> Self {
        Self {
            request,
            prefill_start_ns: None,
            prefill_done_ns: None,
            decode_start_ns: None,
            first_token_ns: None,
            finish_ns: None,
            prefill_node: None,
            prefill_gpu: None,
            decode_node: None,
            decode_gpu: None,
            prefill_tokens: 0,
        }
    }

    pub fn into_result(self) -> RequestResult {
        RequestResult {
            request_id: self.request.request_id,
            session_id: self.request.session_id,
            turn_id: self.request.turn_id,
            arrival_ns: self.request.arrival_ns,
            prefill_start_ns: self.prefill_start_ns.unwrap_or(self.request.arrival_ns),
            prefill_done_ns: self.prefill_done_ns.unwrap_or(self.request.arrival_ns),
            decode_start_ns: self.decode_start_ns.unwrap_or(self.request.arrival_ns),
            first_token_ns: self.first_token_ns.unwrap_or(self.request.arrival_ns),
            finish_ns: self.finish_ns.unwrap_or(self.request.arrival_ns),
            prompt_tokens: self.request.prompt_tokens,
            output_tokens: self.request.output_tokens,
        }
    }
}
