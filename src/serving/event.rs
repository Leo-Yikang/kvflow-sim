use crate::cluster::NodeId;

/// Discrete events that drive the serving simulator.
#[derive(Debug, Clone)]
pub enum ServingEvent {
    /// A new request has arrived.
    RequestArrival { request_id: u64 },
    /// A prefill stage has finished.
    PrefillDone {
        request_id: u64,
        node_id: NodeId,
        gpu_idx: usize,
    },
    /// A KV fetch (from CPU/remote to GPU) has finished.
    FetchDone { request_id: u64 },
    /// A decode stage has finished.
    DecodeDone {
        request_id: u64,
        node_id: NodeId,
        gpu_idx: usize,
    },
}
