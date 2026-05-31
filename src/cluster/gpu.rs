/// Identifier for a GPU within a node.
pub type GpuId = u32;

/// Role assigned to a GPU worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Handles both prefill and decode.
    Unified,
    /// Prefill-only worker.
    Prefill,
    /// Decode-only worker.
    Decode,
    /// Dedicated cache server (no compute).
    Cache,
}

/// A single GPU resource in the cluster.
#[derive(Debug, Clone)]
pub struct GpuResource {
    pub gpu_id: GpuId,
    /// HBM capacity in bytes.
    pub hbm_bytes: u64,
    /// Time until which this GPU is busy (in nanoseconds).
    pub busy_until_ns: u64,
    pub role: WorkerRole,
}

impl GpuResource {
    pub fn new(gpu_id: GpuId, hbm_bytes: u64, role: WorkerRole) -> Self {
        Self {
            gpu_id,
            hbm_bytes,
            busy_until_ns: 0,
            role,
        }
    }

    /// Check if the GPU is idle at `now_ns`.
    pub fn is_idle_at(&self, now_ns: u64) -> bool {
        self.busy_until_ns <= now_ns
    }
}
