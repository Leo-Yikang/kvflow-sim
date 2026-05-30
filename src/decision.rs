use crate::model::{ComputeModel, ModelProfile};
use crate::transfer::{AnalyticalTransferModel, TransferEstimate, TransferPath};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDecision {
    RecomputePrefill,
    Fetch { estimate: TransferEstimate },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FetchVsRecompute {
    pub prompt_tokens: u32,
    pub reused_prefix_tokens: u32,
    pub kv_bytes: u64,
    pub recompute_ns: u64,
    pub fetch: TransferEstimate,
    pub decision: KvDecision,
}

impl FetchVsRecompute {
    pub fn evaluate<C: ComputeModel>(
        profile: &ModelProfile,
        compute: &C,
        transfer: &AnalyticalTransferModel,
        path: TransferPath,
        prompt_tokens: u32,
        reused_prefix_tokens: u32,
    ) -> crate::Result<Self> {
        let reused_prefix_tokens = reused_prefix_tokens.min(prompt_tokens);
        let kv_bytes = profile.kv_bytes(reused_prefix_tokens);
        let recompute_ns = compute.prefill_time_ns(reused_prefix_tokens);
        let fetch = transfer.estimate(0, path, kv_bytes)?;
        let decision = if fetch.duration_ns() < recompute_ns {
            KvDecision::Fetch { estimate: fetch }
        } else {
            KvDecision::RecomputePrefill
        };

        Ok(Self {
            prompt_tokens,
            reused_prefix_tokens,
            kv_bytes,
            recompute_ns,
            fetch,
            decision,
        })
    }

    pub fn saved_ns(&self) -> i128 {
        self.recompute_ns as i128 - self.fetch.duration_ns() as i128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LinearComputeModel, profiles};
    use crate::transfer::TransferPath;

    #[test]
    fn fetch_wins_when_transfer_is_faster_than_recompute() {
        let profile = profiles::llama_8b_bf16_gqa();
        let compute = LinearComputeModel::conservative_8b();
        let transfer = AnalyticalTransferModel::rdma_400g();

        let result = FetchVsRecompute::evaluate(
            &profile,
            &compute,
            &transfer,
            TransferPath::RemoteMemoryToGpu,
            4096,
            4096,
        )
        .unwrap();

        assert!(matches!(result.decision, KvDecision::Fetch { .. }));
        assert!(result.saved_ns() > 0);
    }
}
