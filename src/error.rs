use thiserror::Error;

pub type Result<T> = std::result::Result<T, KvFlowError>;

#[derive(Debug, Error)]
pub enum KvFlowError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid trace line {line}: {message}")]
    InvalidTraceLine { line: usize, message: String },

    #[error("invalid model profile: {0}")]
    InvalidModelProfile(String),

    #[error("invalid transfer model: {0}")]
    InvalidTransferModel(String),
}
