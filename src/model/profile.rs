use crate::error::{KvFlowError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub model_id: String,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub bytes_per_elem: u32,
    pub max_context_tokens: u32,
}

impl ModelProfile {
    pub fn new(
        model_id: impl Into<String>,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        bytes_per_elem: u32,
        max_context_tokens: u32,
    ) -> Result<Self> {
        let profile = Self {
            model_id: model_id.into(),
            num_layers,
            num_kv_heads,
            head_dim,
            bytes_per_elem,
            max_context_tokens,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() {
            return Err(KvFlowError::InvalidModelProfile(
                "model_id is empty".to_string(),
            ));
        }
        if self.num_layers == 0
            || self.num_kv_heads == 0
            || self.head_dim == 0
            || self.bytes_per_elem == 0
        {
            return Err(KvFlowError::InvalidModelProfile(
                "KV shape fields must be positive".to_string(),
            ));
        }
        Ok(())
    }

    pub fn kv_bytes_per_token(&self) -> u64 {
        2_u64
            * self.num_layers as u64
            * self.num_kv_heads as u64
            * self.head_dim as u64
            * self.bytes_per_elem as u64
    }

    pub fn kv_bytes(&self, tokens: u32) -> u64 {
        self.kv_bytes_per_token().saturating_mul(tokens as u64)
    }
}

pub mod profiles {
    use super::ModelProfile;

    pub fn llama_8b_bf16_gqa() -> ModelProfile {
        ModelProfile::new("llama-8b-bf16-gqa", 32, 8, 128, 2, 131_072).unwrap()
    }

    pub fn llama_70b_bf16_gqa() -> ModelProfile {
        ModelProfile::new("llama-70b-bf16-gqa", 80, 8, 128, 2, 131_072).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::profiles;

    #[test]
    fn llama_8b_kv_bytes_per_token_is_128_kib() {
        let profile = profiles::llama_8b_bf16_gqa();
        assert_eq!(profile.kv_bytes_per_token(), 128 * 1024);
        assert_eq!(profile.kv_bytes(4096), 512 * 1024 * 1024);
    }
}
