# KVFlow-Sim Roadmap

> Planning document. No implementation is assumed yet.

## Phase 0: Scope Lock

Goal: make the new project's abstraction clear before writing code.

Deliverables:

- `README.md` with project scope and non-goals.
- `docs/research_plan.md` with candidate paper directions.
- `docs/architecture.md` with module boundaries.
- A minimal terminology list:
  - request
  - session
  - turn
  - prefill
  - decode
  - KV object
  - cache tier
  - transfer
  - recompute
  - TTFT
  - TBT
  - SLO violation

Exit criteria:

- The project is no longer described as a network protocol simulator.
- Packet-level fabric simulation is explicitly scoped as calibration only.

## Phase 1: Trace and KV Size Analysis

Goal: understand workload shape and KV-cache size before building a full simulator.

Core components:

```text
src/trace/
  jsonl.rs
  synthetic.rs

src/model/
  profile.rs
  kv_size.rs

examples/
  trace_stats.rs
  kv_size.rs
  fetch_vs_recompute.rs
```

Minimum data model:

```text
LlmRequest:
  request_id
  session_id
  turn_id
  arrival_ns
  prompt_tokens
  new_prompt_tokens
  output_tokens
  model_id
```

Experiments:

- Prompt length distribution.
- Output length distribution.
- Multi-turn reuse potential.
- KV bytes per token for representative models.
- Break-even curve:

```text
remote KV fetch time < recompute prefill time
```

Exit criteria:

- Can load or generate an LLM request trace.
- Can compute KV-cache bytes for each request.
- Can produce the first fetch-vs-recompute table or plot.

## Phase 2: Request-Level Serving Runner

Goal: run a baseline LLM serving simulation without KV reuse.

Core events:

```text
RequestArrival
PrefillStart
PrefillDone
DecodeStart
DecodeStepDone
RequestDone
```

Core resources:

```text
GPU busy_until
prefill worker pool
decode worker pool
simple queue
```

Baseline:

- `NoReuse`: every request computes full prefill.

Metrics:

- TTFT
- TBT / TPOT
- JCT
- throughput in req/s and token/s
- GPU utilization
- queueing delay

Exit criteria:

- Can simulate request streams at different arrival rates.
- Can report P50/P95/P99 TTFT and TBT.
- Can distinguish prefill and decode resource pressure.

## Phase 3: KV Cache Hierarchy

Goal: make KV cache an explicit object in the simulator.

Core components:

```text
src/cache/
  object.rs
  tier.rs
  state.rs
  eviction.rs

src/policy/
  no_reuse.rs
  local_lru.rs
  remote_lru.rs
  size_aware.rs
```

Cache tiers:

```text
GPU HBM
CPU DRAM
Remote CPU DRAM
```

Optional later tiers:

```text
Local SSD
Remote SSD
Object store
```

Baseline policies:

- `NoReuse`
- `LocalGpuLru`
- `CpuOffloadLru`
- `RemoteLru`
- `SizeAware`

Metrics:

- KV hit rate by tier.
- Cache occupancy.
- Evicted bytes.
- Recomputed tokens.
- Recompute time saved.
- Remote bytes fetched.

Exit criteria:

- Can compare recompute against local and remote KV reuse.
- Can show when remote KV reuse helps or hurts.

## Phase 4: Data Movement and RDMA-Like Transfer

Goal: add object-level transfer models and network-aware policies.

Transfer models:

- `AnalyticalTransferModel`
- `QueuedResourceTransferModel`
- `TableCalibratedTransferModel`

Path types:

```text
GPU -> GPU same node via NVLink/NVSwitch
GPU -> CPU same node
CPU -> GPU same node
Remote CPU -> GPU via RDMA-like fabric
Remote SSD -> GPU
```

Network resources:

```text
NIC busy_until
rack uplink busy_until
fabric bottleneck busy_until
per-tier bandwidth and base latency
```

Candidate policies:

- `TopologyAware`
- `NetworkAware`
- `DeadlineAware`
- `FetchOrRecompute`

Exit criteria:

- Can model remote KV fetch queueing.
- Can show how network contention affects TTFT/TBT.
- Can evaluate fetch-vs-recompute under load.

## Phase 5: Fabric Calibration Bridge

Goal: use packet-level simulation or real measurements to calibrate object-level transfer costs.

Inputs:

- Packet-level simulator results.
- Real microbenchmarks if available.
- Literature parameters if hardware is unavailable.

Calibration outputs:

```text
transfer_latency_table:
  bytes
  concurrency
  locality
  protocol
  p50_latency
  p99_latency
  effective_bandwidth
```

Use cases:

- TCP vs RDMA-like transfer.
- Rack-local vs rack-remote transfer.
- Incast KV pull.
- Hotspot cache server.
- Multi-rail sensitivity.

Exit criteria:

- Serving simulator can read a calibration table.
- Large-scale experiments do not run packet-level simulation directly.

## Phase 6: Paper-Oriented Evaluation

Goal: produce a complete experimental story.

Workloads:

- ShareGPT-like multi-turn chat.
- LMSYS-like open chatbot traffic.
- Synthetic long-context workload.
- RAG-like repeated-prefix workload.
- Stress tests for cache hot spots and network congestion.

Figures:

- Request rate vs P99 TTFT.
- Request rate vs P99 TBT.
- Cache capacity vs recompute saved.
- Network bandwidth vs remote KV benefit.
- Context length vs fetch/recompute break-even.
- Multi-turn reuse ratio vs throughput.
- Rack locality vs tail latency.

Exit criteria:

- One clear policy contribution.
- Baselines are stable and explainable.
- Simulator assumptions are documented and calibrated.
