use super::{CacheState, KvObject};

/// Decides which object(s) to evict when a cache tier is full.
pub trait EvictionPolicy {
    /// From the provided candidates, pick one to evict.
    ///
    /// `candidates` are usually the objects currently in the target tier.
    /// The policy may use `state` to access global information (e.g. reuse
    /// probability, object size).
    fn select_for_eviction<'a>(
        &self,
        candidates: &'a [KvObject],
        state: &CacheState,
    ) -> Option<&'a KvObject>;
}

/// Evict the least-recently-used object.
#[derive(Debug, Clone, Copy)]
pub struct LruEviction;

impl EvictionPolicy for LruEviction {
    fn select_for_eviction<'a>(
        &self,
        candidates: &'a [KvObject],
        _state: &CacheState,
    ) -> Option<&'a KvObject> {
        candidates.iter().min_by_key(|o| o.last_access_ns)
    }
}

/// Evict the largest object (size-aware).
#[derive(Debug, Clone, Copy)]
pub struct LargestFirstEviction;

impl EvictionPolicy for LargestFirstEviction {
    fn select_for_eviction<'a>(
        &self,
        candidates: &'a [KvObject],
        _state: &CacheState,
    ) -> Option<&'a KvObject> {
        candidates.iter().max_by_key(|o| o.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheLocation;

    fn make_obj(kv_id: u64, bytes: u64, last_access_ns: u64) -> KvObject {
        KvObject {
            kv_id,
            session_id: kv_id,
            model_id: "m".to_string(),
            prefix_tokens: 10,
            bytes,
            location: CacheLocation::Gpu {
                node_id: 0,
                gpu_id: 0,
            },
            last_access_ns,
            ref_count: 0,
        }
    }

    #[test]
    fn lru_picks_oldest() {
        let state = CacheState::new();
        let candidates = vec![
            make_obj(0, 100, 500),
            make_obj(1, 100, 100),
            make_obj(2, 100, 300),
        ];
        let policy = LruEviction;
        let chosen = policy.select_for_eviction(&candidates, &state).unwrap();
        assert_eq!(chosen.kv_id, 1);
    }

    #[test]
    fn largest_first_picks_biggest() {
        let state = CacheState::new();
        let candidates = vec![
            make_obj(0, 100, 0),
            make_obj(1, 500, 0),
            make_obj(2, 200, 0),
        ];
        let policy = LargestFirstEviction;
        let chosen = policy.select_for_eviction(&candidates, &state).unwrap();
        assert_eq!(chosen.kv_id, 1);
    }
}
