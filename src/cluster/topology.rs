use super::{NodeId, ServingNode};

/// Cluster-level topology and resource pool.
#[derive(Debug, Clone)]
pub struct ClusterTopology {
    pub nodes: Vec<ServingNode>,
}

impl ClusterTopology {
    pub fn new(nodes: Vec<ServingNode>) -> Self {
        Self { nodes }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn gpu_count(&self) -> usize {
        self.nodes.iter().map(|n| n.gpus.len()).sum()
    }

    /// Find the node with the earliest idle GPU for the given role.
    pub fn earliest_gpu_for(
        &self,
        now_ns: u64,
        role: super::WorkerRole,
    ) -> Option<(NodeId, usize, u64)> {
        let mut best: Option<(NodeId, usize, u64)> = None;
        for node in &self.nodes {
            if let Some((gpu_idx, start_ns)) = node.earliest_gpu_for(now_ns, role) {
                if best.map_or(true, |(_, _, best_start)| start_ns < best_start) {
                    best = Some((node.node_id, gpu_idx, start_ns));
                }
            }
        }
        best
    }

    /// Mutable access to a node by id.
    pub fn node_mut(&mut self, node_id: NodeId) -> Option<&mut ServingNode> {
        self.nodes.iter_mut().find(|n| n.node_id == node_id)
    }

    /// Immutable access to a node by id.
    pub fn node(&self, node_id: NodeId) -> Option<&ServingNode> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    /// Find an idle GPU anywhere in the cluster for the given role.
    pub fn find_idle_gpu_for(
        &self,
        now_ns: u64,
        role: super::WorkerRole,
    ) -> Option<(NodeId, usize)> {
        self.nodes
            .iter()
            .filter_map(|node| {
                node.find_idle_gpu_for(now_ns, role)
                    .map(|gpu_idx| (node.node_id, gpu_idx))
            })
            .min_by_key(|(node_id, gpu_idx)| {
                self.node(*node_id).unwrap().gpus[*gpu_idx].busy_until_ns
            })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{GpuResource, WorkerRole};
    use super::*;

    #[test]
    fn earliest_gpu_selects_idle_one() {
        let node = ServingNode::new(
            0,
            0,
            vec![
                GpuResource::new(0, 80_000_000_000, WorkerRole::Prefill),
                GpuResource::new(1, 80_000_000_000, WorkerRole::Decode),
            ],
            1,
            1_000_000_000_000,
            10_000_000_000_000,
        );
        let cluster = ClusterTopology::new(vec![node]);

        let (nid, gid, start) = cluster.earliest_gpu_for(0, WorkerRole::Prefill).unwrap();
        assert_eq!(nid, 0);
        assert_eq!(gid, 0);
        assert_eq!(start, 0);

        // Decode worker should not match Prefill request
        let decode = cluster.earliest_gpu_for(0, WorkerRole::Decode);
        assert!(decode.is_some());
        assert_eq!(decode.unwrap().1, 1);
    }
}
