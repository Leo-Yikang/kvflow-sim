# Architecture Plan

## 1. Design Principle

KVFlow-Sim should be a request/object-level simulator.

The main simulation unit is not a packet. It is one of:

- LLM request
- prefill task
- decode step
- KV object
- KV transfer
- cache eviction

Network transport should be modeled as object-level transfer cost:

```text
transfer_time = base_latency + bytes / effective_bandwidth + queueing_delay
```

Packet-level simulation may later calibrate `base_latency`, `effective_bandwidth`, and tail factors.

## 2. Proposed Module Layout

```text
kvflow-sim/
  README.md
  docs/
    roadmap.md
    research_plan.md
    architecture.md
  src/
    core/
      event.rs
      queue.rs
      simulator.rs
    trace/
      request.rs
      jsonl.rs
      synthetic.rs
    model/
      profile.rs
      latency.rs
      kv_size.rs
    cluster/
      node.rs
      gpu.rs
      topology.rs
      resource.rs
    cache/
      object.rs
      tier.rs
      state.rs
      eviction.rs
    transfer/
      model.rs
      analytical.rs
      queued.rs
      table.rs
    policy/
      placement.rs
      eviction.rs
      transfer.rs
      baselines.rs
    serving/
      scheduler.rs
      runner.rs
      event.rs
      metrics.rs
    calibration/
      compute.rs
      network.rs
      storage.rs
    report/
      summary.rs
  examples/
    trace_stats.rs
    kv_size.rs
    fetch_vs_recompute.rs
    serving_no_reuse.rs
    serving_cache_policy.rs
```

## 3. Core Types

### 3.1 Request

```rust
pub struct LlmRequest {
    pub request_id: u64,
    pub session_id: u64,
    pub turn_id: u32,
    pub arrival_ns: u64,
    pub prompt_tokens: u32,
    pub new_prompt_tokens: u32,
    pub output_tokens: u32,
    pub model_id: ModelId,
    pub slo_ttft_ns: Option<u64>,
    pub slo_tbt_ns: Option<u64>,
}
```

### 3.2 Model Profile

```rust
pub struct ModelProfile {
    pub model_id: ModelId,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub bytes_per_elem: u32,
    pub max_context_tokens: u32,
    pub prefill_curve: LatencyCurve,
    pub decode_curve: LatencyCurve,
}
```

KV size formula:

```text
kv_bytes_per_token =
  2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
```

### 3.3 KV Object

```rust
pub struct KvObject {
    pub kv_id: KvId,
    pub session_id: u64,
    pub model_id: ModelId,
    pub prefix_tokens: u32,
    pub bytes: u64,
    pub location: CacheLocation,
    pub last_access_ns: u64,
    pub ref_count: u32,
}

pub enum CacheLocation {
    Gpu { node: NodeId, gpu: GpuId },
    Cpu { node: NodeId },
    LocalSsd { node: NodeId },
    RemoteMemory { node: NodeId },
    RemoteSsd { node: NodeId },
    Missing,
}
```

### 3.4 Cluster

```rust
pub struct ServingNode {
    pub node_id: NodeId,
    pub rack_id: RackId,
    pub gpus: Vec<GpuResource>,
    pub nics: Vec<NicResource>,
    pub cpu_mem_bytes: u64,
    pub local_ssd_bytes: u64,
}

pub struct GpuResource {
    pub gpu_id: GpuId,
    pub hbm_bytes: u64,
    pub busy_until_ns: u64,
    pub role: WorkerRole,
}

pub enum WorkerRole {
    Unified,
    Prefill,
    Decode,
    Cache,
}
```

### 3.5 Transfer

```rust
pub trait TransferModel {
    /// Stateful estimate. May advance internal queueing state
    /// (e.g. `QueuedTransferModel::nic_busy_until_ns`). Use for real fetches.
    fn estimate(
        &mut self,
        now_ns: u64,
        path: TransferPath,
        bytes: u64,
    ) -> Result<TransferEstimate>;

    /// Stateless duration estimate. MUST NOT advance any internal state.
    /// This is the right entry point for "what-if" scoring during
    /// placement — calling `estimate` would pollute the model's state
    /// (e.g. NIC busy time) for subsequent real fetches.
    fn estimate_duration(
        &self,
        path: TransferPath,
        bytes: u64,
    ) -> Result<u64>;
}

pub struct TransferEstimate {
    pub start_ns: u64,
    pub finish_ns: u64,
    pub base_latency_ns: u64,
    pub serialization_ns: u64,
    pub bandwidth_bps: u64,
    pub path: TransferPath,
    pub bytes: u64,
}

pub enum TransferPath {
    LocalGpuToGpu,
    LocalCpuToGpu,
    RemoteMemoryToGpu,
    RemoteSsdToGpu,
}
```

Implementations:

- `AnalyticalTransferModel` — fixed bandwidth and base latency per path.
  `estimate` and `estimate_duration` are equivalent (no internal state).
- `QueuedTransferModel` — remote paths share a NIC busy time. `estimate`
  advances `nic_busy_until_ns`; `estimate_duration` returns the
  base + serialization latency without touching the busy time.
- `TableCalibratedTransferModel` — table hit returns the stored
  latency; miss falls back to the analytical model. Both `estimate`
  and `estimate_duration` skip the table mutation.

## 4. Serving Events

```rust
pub enum ServingEventKind {
    RequestArrival { request_id: u64 },
    KvLookupDone { request_id: u64 },
    KvTransferStart { transfer_id: u64 },
    KvTransferDone { transfer_id: u64 },
    PrefillStart { request_id: u64, worker: WorkerId },
    PrefillDone { request_id: u64, worker: WorkerId },
    DecodeStart { request_id: u64, worker: WorkerId },
    DecodeStepDone { request_id: u64, token_idx: u32 },
    RequestDone { request_id: u64 },
    CacheEvict { kv_id: KvId },
}
```

## 5. Runner Flow

Baseline NoReuse:

```text
RequestArrival
  -> scheduler chooses prefill worker
  -> PrefillStart
  -> PrefillDone
  -> create KV object in GPU cache
  -> DecodeStart
  -> DecodeStepDone repeated output_tokens times
  -> RequestDone
```

With KV reuse:

```text
RequestArrival
  -> lookup previous session/prefix KV
  -> if hit:
       choose fetch or recompute
       maybe KvTransferStart -> KvTransferDone
     else:
       PrefillStart -> PrefillDone
  -> DecodeStart
  -> DecodeStepDone repeated output_tokens times
  -> RequestDone
```

## 6. Policy Boundaries

Placement policy:

```text
Where should a newly created KV object be stored?
```

Eviction policy:

```text
Which KV objects should be evicted to admit a new one?
```

Transfer policy:

```text
If a request can reuse remote KV, should it fetch, recompute, partially fetch, or wait?
```

Scheduler:

```text
Which prefill/decode worker should run the request?
```

These should remain separate so experiments can swap one policy at a time.

`NetworkAwarePlacement` scores each candidate tier by
`utility = recompute_cost - fetch_cost - pressure_penalty` and tries them
in descending order. Two correctness details:

- **Capacity short-circuit**: before attempting LRU eviction on a tier,
  if `cache.capacity(node, tier) < bytes` the policy skips the tier
  without destroying its contents (LRU eviction is destructive, and
  cannot recover from a tier that physically cannot hold the object).
- **Stateless transfer cost**: tier scoring calls
  `transfer.estimate_duration(...)`, not `transfer.estimate(...)`. The
  latter would advance `QueuedTransferModel::nic_busy_until_ns` and
  pollute the next real fetch's start time.

## 7. Metrics

`ServingSummary` should include:

```text
total_requests
completed_requests
throughput_req_s
throughput_token_s
ttft_p50/p95/p99
tbt_p50/p95/p99
jct_p50/p95/p99
slo_violation_rate
prefill_gpu_util
decode_gpu_util
network_util
kv_hit_rate_gpu
kv_hit_rate_cpu
kv_hit_rate_remote
remote_kv_bytes
recomputed_tokens
recompute_time_saved_ns
evicted_kv_bytes
prefetch_waste_bytes
```

`CacheAwareRunner` additionally returns a `CacheHitStats` struct:

```text
hits_gpu        // immediate GPU hit, no fetch required
hits_cpu        // deferred credit on FetchDone for a CPU hit
hits_remote     // deferred credit on FetchDone for a remote (memory/SSD) hit
misses          // true miss OR a fetch that degraded to a full prefill
fetch_errors    // transfer estimate failed; the hit was *not* credited
placement_errors // placement failed; the request still completed without caching
pending_fetches // in-flight cache-to-GPU transfers at the snapshot moment
```

`hits_cpu` and `hits_remote` are credited only on `FetchDone`; until
then a request is counted in `pending_fetches`. If the fetch fails
(transfer-estimate error or unexpected cache location), the request
moves from `pending_fetches` to `misses` and the hit is *not*
credited. A single request contributes to exactly one of
`{hits_remote, hits_cpu, misses}` — never two. `pending_fetches` must
converge to 0 by the end of the run.

## 8. Configuration

Prefer explicit config files:

```toml
[cluster]
nodes = 64
gpus_per_node = 8
nics_per_node = 1
racks = 8

[network]
intra_node_nvlink_bps = 900000000000
nic_bps = 400000000000
rack_local_base_latency_ns = 5000
rack_remote_base_latency_ns = 12000

[cache]
gpu_hbm_bytes = 80000000000
cpu_mem_bytes = 1000000000000

[model.llama_8b]
num_layers = 32
num_kv_heads = 8
head_dim = 128
bytes_per_elem = 2
```

## 9. Implementation Order

Recommended order:

1. Define types and config.
2. Implement trace reader and synthetic generator.
3. Implement KV size and model profile.
4. Implement static compute/transfer models.
5. Implement NoReuse serving runner.
6. Add GPU/CPU KV cache.
7. Add baseline policies.
8. Add network-aware policy.
9. Add calibration table support.
10. Add plotting/report scripts.

## 10. Main Constraint

Keep the simulator explainable.

When a result changes, it should be easy to attribute the change to:

- more cache hits
- less recompute
- more remote transfer
- network queueing
- GPU queueing
- cache eviction
- prefetch waste

This is more important than simulating every hardware detail.
