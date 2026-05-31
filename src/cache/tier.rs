use std::fmt;

/// Classification of cache storage tiers for capacity tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierKind {
    Gpu,
    Cpu,
    LocalSsd,
    RemoteMemory,
    RemoteSsd,
}

impl TierKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            TierKind::Gpu => "gpu",
            TierKind::Cpu => "cpu",
            TierKind::LocalSsd => "local_ssd",
            TierKind::RemoteMemory => "remote_memory",
            TierKind::RemoteSsd => "remote_ssd",
        }
    }
}

impl fmt::Display for TierKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
