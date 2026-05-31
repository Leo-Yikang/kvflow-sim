# KVFlow-Sim

KVFlow-Sim 是一个面向 LLM 推理服务的、由 trace 驱动的离散事件模拟器。它用于研究在解耦式服务集群中，KV cache 的放置、复用、迁移、淘汰以及对象级数据传输如何影响端到端性能。

本项目的抽象层级是“请求 / 对象级”，不是 packet-level 网络协议模拟。模拟器关注 LLM 请求、会话、多轮对话、prefill/decode 阶段、KV 对象、缓存层级和对象级传输成本。未来可以使用 packet-level RDMA 或 fabric 模拟作为校准后端，但它不是 KVFlow-Sim 的主抽象。

## 研究重点

核心问题是：

> 当 LLM 推理服务把工作拆分到不同 GPU、节点、内存层级和缓存服务器上时，系统应该复用、传输、淘汰、预取还是重新计算哪些 KV cache 对象？

KVFlow-Sim 希望帮助评估以下策略：

- KV cache 在 GPU HBM、CPU DRAM、本地 SSD、远端内存、远端 SSD 等层级之间的放置。
- 多轮对话和长上下文负载下的 fetch-vs-recompute 决策。
- prefill/decode 解耦后的 KV 传输调度。
- 感知网络拓扑、拥塞和 deadline 的远端 KV 拉取。
- RDMA-like 传输对 TTFT、TBT 和 SLO violation 的影响。

## 当前实现

仓库目前已经包含一组用于早期实验的基础模块：

- `trace`：LLM 请求数据结构、JSONL trace 读写、合成 trace 生成、trace 统计。
- `model`：模型 profile、KV cache 大小计算、简单线性 compute latency 模型。
- `transfer`：对象级 analytical transfer model，覆盖本地 GPU、本地 CPU、远端内存和远端 SSD 到 GPU 的传输估计。
- `decision`：fetch-vs-recompute 决策辅助逻辑。
- `serving`：无 KV 复用的 baseline runner，以及 TTFT、TBT、JCT、吞吐等统计。
- `examples`：trace 统计、KV 大小表、fetch-vs-recompute、带宽敏感性和 NoReuse serving baseline 示例。

## 快速运行

先运行测试：

```bash
cargo test
```

运行合成 trace 的统计示例：

```bash
cargo run --example trace_stats
```

计算代表性模型在不同上下文长度下的 KV cache 大小：

```bash
cargo run --example kv_size
```

比较远端 KV fetch 和重新计算 prefill 的代价：

```bash
cargo run --example fetch_vs_recompute
cargo run --example fetch_sensitivity
```

运行不复用 KV cache 的服务 baseline：

```bash
cargo run --example serving_no_reuse
```

`trace_stats` 也可以读取 JSONL trace：

```bash
cargo run --example trace_stats -- path/to/trace.jsonl
```

每一行 JSON 应对应一个 `LlmRequest`，主要字段包括：

```json
{
  "request_id": 1,
  "session_id": 7,
  "turn_id": 0,
  "arrival_ns": 10,
  "prompt_tokens": 1024,
  "new_prompt_tokens": 1024,
  "output_tokens": 128,
  "model_id": "llama-8b",
  "slo_ttft_ns": null,
  "slo_tbt_ns": 50000000
}
```

其中 `prompt_tokens` 表示本次请求可见的完整上下文长度，`new_prompt_tokens` 表示相对同一 session 上一轮新增的 prompt token 数量，可用于估计可复用的 prefix KV。

## 非目标

KVFlow-Sim 不应演变成完整的 packet-level RDMA/RoCE/InfiniBand 模拟器。

它不会以以下内容作为主要建模对象：

- per-packet ACK/NAK/PSN 行为。
- PFC pause frame 或 pause storm。
- 精确的 DCQCN/HPCC/Swift 控制环路。
- 交换机 buffer 微架构。
- 10K 节点规模下的 packet-level 事件模拟。

这些内容更适合放在单独的 fabric calibration 工具中。

## 文档

- [Roadmap](docs/roadmap.md)
- [Research Plan](docs/research_plan.md)
- [Architecture](docs/architecture.md)

## 与 Fabric-Sim 的关系

旧的 packet-level 模拟器仍可作为校准工具使用：

```text
packet-level fabric experiments
  -> effective bandwidth / tail latency tables
  -> KVFlow-Sim object-level TransferModel
  -> large-scale LLM serving experiments
```

KVFlow-Sim 默认应保持在 request/object-level。这样可以在较大规模上解释 KV cache 策略、资源排队、远端传输和重新计算之间的权衡，而不是把主要复杂度放在每个网络包的细节上。
