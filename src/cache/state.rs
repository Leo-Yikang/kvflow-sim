use std::collections::HashMap;

use crate::cluster::NodeId;
use crate::error::{KvFlowError, Result};

use super::{CacheLocation, KvId, KvObject, TierKind};

/// Tracks all KV objects, their locations, and per-tier capacity usage.
#[derive(Debug, Clone)]
pub struct CacheState {
    objects: HashMap<KvId, KvObject>,
    /// Index from (session_id, prefix_tokens) -> kv_id for fast prefix lookup.
    by_session_prefix: HashMap<(u64, u32), KvId>,
    /// Capacity per (node_id, tier).
    capacities: HashMap<(NodeId, TierKind), u64>,
    /// Usage per (node_id, tier).
    usage: HashMap<(NodeId, TierKind), u64>,
    next_kv_id: u64,
}

impl Default for CacheState {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheState {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            by_session_prefix: HashMap::new(),
            capacities: HashMap::new(),
            usage: HashMap::new(),
            next_kv_id: 0,
        }
    }

    /// Set capacity for a node+tier.  Used during cluster initialisation.
    pub fn set_capacity(&mut self, node_id: NodeId, tier: TierKind, bytes: u64) {
        self.capacities.insert((node_id, tier), bytes);
        self.usage.entry((node_id, tier)).or_insert(0);
    }

    /// Allocate a fresh KV id.
    pub fn alloc_kv_id(&mut self) -> KvId {
        let id = self.next_kv_id;
        self.next_kv_id = self.next_kv_id.saturating_add(1);
        id
    }

    /// Look up a KV object by session and prefix length.
    pub fn lookup(&self, session_id: u64, prefix_tokens: u32) -> Option<&KvObject> {
        let kv_id = self.by_session_prefix.get(&(session_id, prefix_tokens))?;
        self.objects.get(kv_id)
    }

    /// Mutable lookup.
    pub fn lookup_mut(&mut self, session_id: u64, prefix_tokens: u32) -> Option<&mut KvObject> {
        let kv_id = self
            .by_session_prefix
            .get(&(session_id, prefix_tokens))
            .copied()?;
        self.objects.get_mut(&kv_id)
    }

    /// Available space on a node+tier.
    pub fn available_space(&self, node_id: NodeId, tier: TierKind) -> u64 {
        let cap = self.capacities.get(&(node_id, tier)).copied().unwrap_or(0);
        let used = self.usage.get(&(node_id, tier)).copied().unwrap_or(0);
        cap.saturating_sub(used)
    }

    /// Total capacity on a node+tier.
    pub fn capacity(&self, node_id: NodeId, tier: TierKind) -> u64 {
        self.capacities.get(&(node_id, tier)).copied().unwrap_or(0)
    }

    /// Current usage on a node+tier.
    pub fn usage(&self, node_id: NodeId, tier: TierKind) -> u64 {
        self.usage.get(&(node_id, tier)).copied().unwrap_or(0)
    }

    /// All objects residing on a given node+tier.
    pub fn objects_in_tier(&self, node_id: NodeId, tier: TierKind) -> Vec<&KvObject> {
        self.objects
            .values()
            .filter(|o| Self::location_matches_tier(&o.location, node_id, tier))
            .collect()
    }

    /// Remove a KV object and free its capacity.
    pub fn remove(&mut self, kv_id: KvId) -> Option<KvObject> {
        let obj = self.objects.remove(&kv_id)?;
        self.by_session_prefix
            .remove(&(obj.session_id, obj.prefix_tokens));
        if let Some(node_id) = obj.location.node_id() {
            if let Some(tier) = tier_of_location(&obj.location) {
                if let Some(used) = self.usage.get_mut(&(node_id, tier)) {
                    *used = used.saturating_sub(obj.bytes);
                }
            }
        }
        Some(obj)
    }

    /// Insert a new KV object.  Fails if the target tier is over capacity.
    pub fn insert(&mut self, obj: KvObject) -> Result<()> {
        let node_id = obj.location.node_id();
        let tier = tier_of_location(&obj.location);

        if let (Some(node_id), Some(tier)) = (node_id, tier) {
            let available = self.available_space(node_id, tier);
            if available < obj.bytes {
                return Err(KvFlowError::InvalidModelProfile(format!(
                    "cache overflow on node {} tier {}: need {} bytes, available {}",
                    node_id, tier, obj.bytes, available
                )));
            }
            *self.usage.entry((node_id, tier)).or_insert(0) += obj.bytes;
        }

        self.by_session_prefix
            .insert((obj.session_id, obj.prefix_tokens), obj.kv_id);
        self.objects.insert(obj.kv_id, obj);
        Ok(())
    }

    /// Update the last-access timestamp of an object.
    pub fn update_access(&mut self, kv_id: KvId, now_ns: u64) {
        if let Some(obj) = self.objects.get_mut(&kv_id) {
            obj.last_access_ns = now_ns;
        }
    }

    /// Move an object to a new location, updating capacity bookkeeping.
    pub fn relocate(
        &mut self,
        kv_id: KvId,
        new_location: CacheLocation,
        now_ns: u64,
    ) -> Result<()> {
        let mut obj = self.objects.remove(&kv_id).ok_or_else(|| {
            KvFlowError::InvalidModelProfile(format!("KV {} not found for relocation", kv_id))
        })?;

        // Free old capacity.
        if let (Some(old_node), Some(old_tier)) =
            (obj.location.node_id(), tier_of_location(&obj.location))
        {
            if let Some(used) = self.usage.get_mut(&(old_node, old_tier)) {
                *used = used.saturating_sub(obj.bytes);
            }
        }

        // Reserve new capacity.
        if let (Some(new_node), Some(new_tier)) =
            (new_location.node_id(), tier_of_location(&new_location))
        {
            let available = self.available_space(new_node, new_tier);
            if available < obj.bytes {
                // Rollback: restore old object state.
                self.objects.insert(kv_id, obj.clone());
                if let (Some(old_node), Some(old_tier)) =
                    (obj.location.node_id(), tier_of_location(&obj.location))
                {
                    *self.usage.entry((old_node, old_tier)).or_insert(0) += obj.bytes;
                }
                return Err(KvFlowError::InvalidModelProfile(format!(
                    "relocation overflow on node {} tier {}: need {} bytes, available {}",
                    new_node, new_tier, obj.bytes, available
                )));
            }
            *self.usage.entry((new_node, new_tier)).or_insert(0) += obj.bytes;
        }

        obj.location = new_location;
        obj.last_access_ns = now_ns;
        self.objects.insert(kv_id, obj);
        Ok(())
    }

    /// Total number of tracked objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    fn location_matches_tier(location: &CacheLocation, node_id: NodeId, tier: TierKind) -> bool {
        match (location, tier) {
            (CacheLocation::Gpu { node_id: nid, .. }, TierKind::Gpu) => *nid == node_id,
            (CacheLocation::Cpu { node_id: nid }, TierKind::Cpu) => *nid == node_id,
            (CacheLocation::LocalSsd { node_id: nid }, TierKind::LocalSsd) => *nid == node_id,
            (CacheLocation::RemoteMemory { node_id: nid }, TierKind::RemoteMemory) => {
                *nid == node_id
            }
            (CacheLocation::RemoteSsd { node_id: nid }, TierKind::RemoteSsd) => *nid == node_id,
            _ => false,
        }
    }
}

fn tier_of_location(location: &CacheLocation) -> Option<TierKind> {
    match location {
        CacheLocation::Gpu { .. } => Some(TierKind::Gpu),
        CacheLocation::Cpu { .. } => Some(TierKind::Cpu),
        CacheLocation::LocalSsd { .. } => Some(TierKind::LocalSsd),
        CacheLocation::RemoteMemory { .. } => Some(TierKind::RemoteMemory),
        CacheLocation::RemoteSsd { .. } => Some(TierKind::RemoteSsd),
        CacheLocation::Missing => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup_roundtrip() {
        let mut state = CacheState::new();
        state.set_capacity(0, TierKind::Gpu, 1_000_000_000);

        let obj = KvObject {
            kv_id: state.alloc_kv_id(),
            session_id: 7,
            model_id: "llama-8b".to_string(),
            prefix_tokens: 1024,
            bytes: 128_000_000,
            location: CacheLocation::Gpu {
                node_id: 0,
                gpu_id: 0,
            },
            last_access_ns: 100,
            ref_count: 0,
        };
        state.insert(obj).unwrap();

        let found = state.lookup(7, 1024).unwrap();
        assert_eq!(found.session_id, 7);
        assert_eq!(found.bytes, 128_000_000);
    }

    #[test]
    fn insert_fails_when_over_capacity() {
        let mut state = CacheState::new();
        state.set_capacity(0, TierKind::Gpu, 100);

        let obj = KvObject {
            kv_id: state.alloc_kv_id(),
            session_id: 1,
            model_id: "m".to_string(),
            prefix_tokens: 10,
            bytes: 200,
            location: CacheLocation::Gpu {
                node_id: 0,
                gpu_id: 0,
            },
            last_access_ns: 0,
            ref_count: 0,
        };
        assert!(state.insert(obj).is_err());
    }

    #[test]
    fn remove_frees_capacity() {
        let mut state = CacheState::new();
        state.set_capacity(0, TierKind::Gpu, 1_000);

        let obj = KvObject {
            kv_id: state.alloc_kv_id(),
            session_id: 1,
            model_id: "m".to_string(),
            prefix_tokens: 10,
            bytes: 500,
            location: CacheLocation::Gpu {
                node_id: 0,
                gpu_id: 0,
            },
            last_access_ns: 0,
            ref_count: 0,
        };
        state.insert(obj).unwrap();
        assert_eq!(state.usage(0, TierKind::Gpu), 500);

        state.remove(0);
        assert_eq!(state.usage(0, TierKind::Gpu), 0);
    }
}
