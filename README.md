# KVFlow-Sim

KVFlow-Sim is a trace-driven discrete-event simulator for studying KV-cache placement, reuse, and data movement in disaggregated LLM serving clusters.

This repository is planned as a clean successor to a packet-level fabric simulator. The new simulator should model LLM requests, sessions, prefill/decode stages, KV-cache objects, memory tiers, and object-level transfers. Packet-level RDMA or fabric simulation may be used later as a calibration backend, but it is not the main abstraction.

## Research Focus

The core question is:

> When LLM serving systems split work across GPUs, nodes, memory tiers, and cache servers, which KV-cache objects should be reused, moved, evicted, prefetched, or recomputed?

The simulator should help evaluate policies for:

- KV-cache placement across GPU HBM, CPU DRAM, local SSD, remote memory, and remote SSD.
- Fetch-vs-recompute decisions for multi-turn and long-context workloads.
- Prefill/decode disaggregation and KV transfer scheduling.
- Network-aware and deadline-aware remote KV fetch.
- The effect of RDMA-like transport on TTFT, TBT, and SLO violations.

## Non-Goals

KVFlow-Sim should not become a full packet-level RDMA/RoCE/InfiniBand simulator.

It should not primarily model:

- Per-packet ACK/NAK/PSN behavior.
- PFC pause frames or pause storms.
- Precise DCQCN/HPCC/Swift control loops.
- Switch buffer microarchitecture.
- Packet-level simulation at 10K-node scale.

Those belong in a separate fabric calibration tool.

## Initial Documents

- [Roadmap](docs/roadmap.md)
- [Research Plan](docs/research_plan.md)
- [Architecture](docs/architecture.md)

## Relationship to Fabric-Sim

The old packet-level simulator can still be useful as a calibration tool:

```text
packet-level fabric experiments
  -> effective bandwidth / tail latency tables
  -> KVFlow-Sim object-level TransferModel
  -> large-scale LLM serving experiments
```

KVFlow-Sim should remain request/object-level by default.
