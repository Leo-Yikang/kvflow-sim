use crate::cluster::NodeId;

/// Unique identifier for a KV object.
pub type KvId = u64;

/// A KV cache object representing the key-value tensors for a prefix of tokens
/// within a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvObject {
    pub kv_id: KvId,
    pub session_id: u64,
    pub model_id: String,
    pub prefix_tokens: u32,
    pub bytes: u64,
    pub location: CacheLocation,
    pub last_access_ns: u64,
    pub ref_count: u32,
}

/// Physical location of a KV object in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLocation {
    Gpu { node_id: NodeId, gpu_id: u32 },
    Cpu { node_id: NodeId },
    LocalSsd { node_id: NodeId },
    RemoteMemory { node_id: NodeId },
    RemoteSsd { node_id: NodeId },
    Missing,
}

impl CacheLocation {
    /// The node that hosts this location, if any.
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            CacheLocation::Gpu { node_id, .. }
            | CacheLocation::Cpu { node_id }
            | CacheLocation::LocalSsd { node_id }
            | CacheLocation::RemoteMemory { node_id }
            | CacheLocation::RemoteSsd { node_id } => Some(*node_id),
            CacheLocation::Missing => None,
        }
    }

    /// Whether this location is GPU HBM on any node.
    pub fn is_gpu(&self) -> bool {
        matches!(self, CacheLocation::Gpu { .. })
    }

    /// Whether this location is local to the given node.
    pub fn is_local_to(&self, node_id: NodeId) -> bool {
        match self {
            CacheLocation::Gpu { node_id: nid, .. }
            | CacheLocation::Cpu { node_id: nid }
            | CacheLocation::LocalSsd { node_id: nid } => *nid == node_id,
            _ => false,
        }
    }
}
