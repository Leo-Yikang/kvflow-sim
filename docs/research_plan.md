# Research Plan

## 1. Motivation

LLM serving systems increasingly split work across stages and resources:

- Prefill and decode can run on different workers.
- KV cache can outgrow GPU HBM.
- Multi-turn conversations and long-context workloads create reuse opportunities.
- Remote memory and cache servers can reduce recomputation but add data movement.
- RDMA-like transport reduces transfer overhead, but network contention and tail latency still matter.

This creates a storage and memory systems problem:

> KV cache is no longer just temporary GPU memory. It becomes a distributed memory object whose placement and movement affect user-visible latency.

## 2. Main Research Question

When serving multi-turn and long-context LLM workloads, how should a serving system decide whether to reuse, transfer, prefetch, evict, or recompute KV cache?

## 3. Candidate Paper Directions

### Direction A: Network-Aware KV Placement

Hypothesis:

> Remote KV reuse is beneficial only when placement decisions consider topology, congestion, KV size, and recompute cost.

Baseline policies:

- No reuse.
- Local GPU LRU.
- CPU offload LRU.
- Remote LRU.
- Size-aware placement.
- Topology-aware placement.

Proposed policy:

- Network-aware placement and fetch.

Key idea:

```text
utility(KV) =
  recompute_cost_saved
  - expected_transfer_cost
  - expected_cache_pressure
  - expected_deadline_risk
```

Expected results:

- Lower P99 TTFT under high reuse workloads.
- Better SLO compliance under network contention.
- Less wasteful remote transfer than naive remote LRU.

### Direction B: Deadline-Aware KV Transfer

Hypothesis:

> KV transfer should be scheduled by serving-stage urgency, not only by bytes or cache recency.

Examples:

- A request close to TTFT deadline should get priority over speculative prefetch.
- A decode worker blocked on KV should be prioritized differently from background cache migration.
- Under congestion, recompute may beat remote fetch for some requests.

Proposed policy:

- Choose among:
  - remote fetch
  - recompute prefill
  - partial fetch
  - delayed prefetch
  - eviction

based on deadline slack and resource state.

Expected results:

- Lower SLO violation rate.
- Better P99 TBT in congested settings.
- Clear fetch-vs-recompute boundary.

### Direction C: Hierarchical KV Memory System

Hypothesis:

> KV cache needs a hierarchy-aware admission and eviction policy because KV objects have very different sizes, reuse probabilities, and recompute costs.

Tiers:

- GPU HBM
- CPU DRAM
- local SSD
- remote CPU DRAM
- remote SSD

Policy:

- Admit to a tier based on expected reuse, size, recompute cost, and transfer cost.
- Evict based on utility density, not recency only.

Expected results:

- Higher effective cache value per byte.
- Reduced GPU memory pressure.
- Better long-context serving throughput.

## 4. Recommended First Target

Start with Direction A:

> Network-aware KV placement for disaggregated LLM serving.

Why:

- It connects naturally to RDMA and large-cluster communication.
- It remains a storage/memory systems problem.
- It can be evaluated with small-scale calibration plus large-scale simulation.
- It does not require perfect packet-level RDMA modeling.

## 5. Research Story

The paper story can be:

1. LLM serving is becoming disaggregated.
2. KV cache is large and reusable, especially in multi-turn and long-context workloads.
3. Existing cache policies are mostly local or recency-based.
4. Remote KV reuse can help, but naive remote reuse hurts under congestion or poor topology.
5. We model KV cache as distributed memory objects with transfer and recompute costs.
6. We propose a network-aware policy.
7. We evaluate using trace-driven simulation calibrated by small-scale experiments or fabric simulation.

## 6. Workload Plan

Trace-driven workloads:

- ShareGPT-style multi-turn conversations.
- LMSYS-style chatbot traffic.

Synthetic workloads:

- Short prompt, short output.
- Long prompt, short output.
- Long-context multi-turn sessions.
- High prefix-reuse workload.
- RAG-like repeated document prefix workload.
- Hotspot session workload.

Arrival processes:

- Poisson arrivals.
- Bursty arrivals.
- Diurnal rate changes.
- Mixed tenant priorities.

## 7. Metrics

User-facing:

- TTFT
- TBT / TPOT
- JCT
- P50/P95/P99 latency
- SLO violation rate

System:

- req/s
- token/s
- prefill GPU utilization
- decode GPU utilization
- network utilization
- NIC queueing delay
- cache tier occupancy

KV-specific:

- KV hit rate by tier
- remote KV bytes
- recompute tokens avoided
- recompute time saved
- evicted KV bytes
- prefetch waste
- fetch-vs-recompute decision distribution

## 8. Baselines

Required:

- NoReuse
- LocalGpuLRU
- CpuOffloadLRU
- RemoteLRU
- SizeAware
- Oracle or near-oracle offline policy

Optional:

- TopologyAware
- DeadlineAware
- Random placement
- AlwaysFetch
- AlwaysRecompute

## 9. Calibration Plan

Minimum calibration:

- KV bytes per token from model config.
- Prefill latency vs prompt length.
- Decode step latency vs batch size and context length.
- CPU/GPU copy bandwidth.
- Network transfer bandwidth and base latency.

Possible sources:

- vLLM or SGLang microbenchmark on available GPUs.
- Existing packet-level fabric simulator.
- Public hardware specs and paper parameters.
- Cloud experiments if credits are available.

Important principle:

> Use calibration to set object-level model parameters. Do not run large-scale serving experiments packet-by-packet.

## 10. First Three Experiments

### Experiment 1: KV Size and Break-Even

Question:

> For different models and context lengths, when is fetching remote KV faster than recomputing prefill?

Output:

- Table or plot of context length vs latency.
- Lines for local CPU, remote CPU over RDMA-like fabric, remote SSD, and recompute.

### Experiment 2: Remote Reuse Under Load

Question:

> As request rate increases, when does naive remote KV reuse become harmful?

Output:

- Request rate vs P99 TTFT.
- Remote bytes and network queueing delay.
- Comparison of NoReuse, RemoteLRU, and NetworkAware.

### Experiment 3: Multi-Turn Reuse

Question:

> How much does multi-turn session locality help, and where should KV be kept?

Output:

- Reuse ratio vs throughput.
- GPU cache capacity vs SLO violation.
- Tier hit rates.

## 11. Risks

Risk: simulator credibility.

Mitigation:

- Keep the model simple and explainable.
- Calibrate key latency/bandwidth parameters.
- Run sensitivity analysis.

Risk: workload representativeness.

Mitigation:

- Use both trace-driven and synthetic workloads.
- Show results across short, long, and multi-turn settings.

Risk: policy is too obvious.

Mitigation:

- Make deadline/network contention central.
- Compare against strong size-aware/topology-aware baselines.
- Include cases where remote reuse is harmful.

Risk: too much engineering.

Mitigation:

- Implement the smallest runner that can answer fetch-vs-recompute and placement questions.
- Avoid packet-level integration until the serving simulator works.
