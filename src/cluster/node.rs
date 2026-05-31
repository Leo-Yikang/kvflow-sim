use super::{GpuResource, WorkerRole};

/// Identifier for a serving node (physical machine).
pub type NodeId = u32;

/// A physical serving node with GPUs, NICs, CPU memory, and local SSD.
#[derive(Debug, Clone)]
pub struct ServingNode {
    pub node_id: NodeId,
    pub rack_id: u32,
    pub gpus: Vec<GpuResource>,
    /// Number of NICs available for remote transfer.
    pub nics: u32,
    /// CPU DRAM capacity in bytes.
    pub cpu_mem_bytes: u64,
    /// Local SSD capacity in bytes.
    pub local_ssd_bytes: u64,
    /// Time until which the node's NICs are busy.
    pub nic_busy_until_ns: u64,
}

impl ServingNode {
    pub fn new(
        node_id: NodeId,
        rack_id: u32,
        gpus: Vec<GpuResource>,
        nics: u32,
        cpu_mem_bytes: u64,
        local_ssd_bytes: u64,
    ) -> Self {
        Self {
            node_id,
            rack_id,
            gpus,
            nics: nics.max(1),
            cpu_mem_bytes,
            local_ssd_bytes,
            nic_busy_until_ns: 0,
        }
    }

    /// Find the earliest idle GPU matching the requested role (or Unified).
    pub fn earliest_gpu_for(&self, now_ns: u64, role: WorkerRole) -> Option<(usize, u64)> {
        self.gpus
            .iter()
            .enumerate()
            .filter(|(_, g)| g.role == role || g.role == WorkerRole::Unified)
            .map(|(idx, g)| (idx, g.busy_until_ns.max(now_ns)))
            .min_by_key(|(_, start)| *start)
    }

    /// Find an idle GPU matching the requested role.
    pub fn find_idle_gpu_for(&self, now_ns: u64, role: WorkerRole) -> Option<usize> {
        self.gpus
            .iter()
            .enumerate()
            .filter(|(_, g)| {
                g.is_idle_at(now_ns) && (g.role == role || g.role == WorkerRole::Unified)
            })
            .min_by_key(|(_, g)| g.busy_until_ns)
            .map(|(idx, _)| idx)
    }

    /// Total HBM capacity across all GPUs.
    pub fn total_hbm_bytes(&self) -> u64 {
        self.gpus.iter().map(|g| g.hbm_bytes).sum()
    }
}
