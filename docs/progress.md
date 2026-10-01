# Memory Path Engine — 当前进展分析

> 对照 [vision.md](vision.md) 三阶段路线图与 [benchmark-strategy.md](benchmark-strategy.md) 三层评测模型。  
> 本文件是可交付的现状快照；**不取代**也不修改历史 plan 文件。

**快照：** Product **M5**（可插拔 dense embedding + LongMemEval KPI 对比）进行中 / 合入后更新。

## 1. 定位与成熟度

| 维度 | 现状 |
| --- | --- |
| 定位 | 从研究原型转向 **本地记忆产品**（可回放路径） |
| 版本 | `pyproject.toml` → `0.5.1` |
| 产品入口 | `mpe` CLI、`.mpe/` SQLite palace、`mpe mcp` / Docker `mpe-mcp`、hooks |
| 研究内核 | typed graph、`MemoryPath`、Palace v1、Stage 6 类脑机制 |
| Embedding | 默认 `ngram`（无新硬依赖）；可选 `fastembed` / `sentence-transformers` |

一句话：

> **公开召回做门面（Layer A）+ CLI/MCP 做闭环 + 路径/图做差异（Layer B）**。

## 2. 技术架构（两轨 + 产品面）

| 轨道 | 位置 | 状态 |
| --- | --- | --- |
| v0 图检索栈 | `schema` / `store` / `retrieve` / `scoring` / `replay` | 稳定；含 `hybrid`（BM25 + 可插拔 embedding） |
| Memory Palace v1 | `memory/` domain + application | 稳定；`palace_to_store` 桥接 |
| 产品持久化 | `persistence/` + `palace_workspace` | M1 已合入 |
| Agent 闭环 | `mcp_server` + `hooks_install` | M2 已合入 |
| Stage 6 机制 | multi-rep / forgetting / `PathReasoner` | 已合入 |

## 3. 愿景阶段对照（Stage 1–3）

| Vision | 计划含义 | 仓库现状 |
| --- | --- | --- |
| Stage 1 Memory-Augmented RAG | 结构 + 权重/异常 + 路径检索 | **完成**：多 retriever、`MemoryPath`、anomaly、加权/扩散 |
| Stage 2 Graph Memory System | 关系一等公民 + 受控激活 | **基本完成**：矛盾边、激活服务、domain packs；双轨 API 见 `api-tracks.md` |
| Stage 3 Brain-like | 强化/衰减、多表示、query→path→answer | **MVP 完成**：状态机 + mild/aggressive + dual views + PathReasoner |

## 4. 评测三层对照（Layer A / B / C）

| Layer | 目标 | 现状 | 验收命令 |
| --- | --- | --- | --- |
| **A** 外部站位 | LongMemEval / HotpotQA | tiny+turn 基线；medium 30/50q KPI；M5 ngram vs fastembed 对比 | `mpe bench longmemeval --label tiny [--embedding fastembed]` |
| **B** 机制验证 | path / semantic / contradiction / dynamic | fixtures + Stage1 标注规格 + Layer B / ablation 报告 | `python scripts/generate_layer_b_report.py` |
| **C** 真实迁移 | 噪声文档 + 私有金标流程 | stand-in 可跑 + inventory/annotation 模板 + 试点清单 | `python scripts/run_layer_c_benchmark.py` |

## 5. 计划阶段序列验收（阶段 0–6）

| 阶段 | 计划内容 | 结论 | 本轮复核 |
| --- | --- | --- | --- |
| 0 基线守住 | CI + Layer B 不回退 | **通过** | `unittest` **191** OK；Layer B/ablation/Layer C/Layer A tiny 脚本可跑 |
| 1 Layer B 机制 | 矛盾/异常可证伪 | **通过** | 既有 fixtures + **新** [`stage1_plan_appendix_benchmark.json`](../benchmarks/structured_memory/stage1_plan_appendix_benchmark.json)（10 case）+ [`STAGE1_CASE_SPEC.md`](../benchmarks/structured_memory/STAGE1_CASE_SPEC.md) |
| 2 消融工业化 | ablation + 延迟汇总 | **通过** | `generate_ablation_report.py` |
| 3 Layer A 规模化 | 公开指标与架构指标分离 | **通过** | Layer A 报告 + nightly（含 hybrid）+ medium KPI 快照 |
| 4 Layer C | 私有金标流程 | **脚手架通过** | [`layer-c-pilot-checklist.md`](layer-c-pilot-checklist.md)；私有 30 文档 / 120–180 case 仍属组织侧 |
| 5 Stage 2 收口 | domain pack + API 心智模型 | **通过** | contract/runbook/research packs + `api-tracks.md` |
| 6 Stage 3 | 类脑机制独立里程碑 | **MVP 通过** | multi-rep / forgetting / PathReasoner + `test_stage6_mechanisms.py` |

## 6. 产品里程碑（并行线）

见 [ROADMAP.md](ROADMAP.md)：

- [x] M1 Persist + CLI + baseline skeleton  
- [x] M2 MCP + hooks + hybrid  
- [x] M3 LongMemEval turn + full 复现说明  
- [x] 公开召回 KPI 提升（hybrid medium R@5 ≈ 0.96–1.0）  
- [x] M4 分发（pipx / uv / Docker MCP / backup·repair·doctor）  
- [x] M5 可插拔 dense embedding（`ngram`/`hash`/`fastembed`/`sentence`）+ KPI 对比

## 7. 计划附录：你还需要准备的数据

### A. Layer B（仓库内 — 已交付）

- 合成 markdown + contradiction / exception fixture（`benchmarks/structured_memory/`）
- **计划附录新包**：`examples/graph_contract_pack/stage1_plan_pack/` + `stage1_plan_appendix_benchmark.json`（10 case）
- 标注规格表：[`STAGE1_CASE_SPEC.md`](../benchmarks/structured_memory/STAGE1_CASE_SPEC.md)
- 稳定 node id 与 domain pack 边类型对齐

### B. Layer C 私有金标（不进公开仓 — 流程已就绪）

按 [private-contract-dataset-guide.md](private-contract-dataset-guide.md) 与  
[`layer-c-pilot-checklist.md`](layer-c-pilot-checklist.md)：

1. **合同 inventory**（约 30 份）  
2. **金标注释表**（约 120–180 case）  
3. 覆盖 multi-hop / exception / contradiction 题型  

模板：`benchmarks/layer_c_minimal/templates/`。

### C. Layer A 公开集

不必自建；使用 `scripts/download_hotpotqa.py` / `download_longmemeval.py`。  
已提交 medium KPI：`benchmarks/external/longmemeval/baselines/longmemeval_kpi_medium{30,50}.*`。  
M5 对比：`longmemeval_kpi_medium50_fastembed.*`（相对默认 ngram）。

## 8. 建议的下一步优先级

1. **组织侧 Layer C**：填 inventory + 私有金标（不提交私密文本）  
2. **可选**：full LongMemEval-S 写入对外表（可配 `--embedding fastembed`）  
3. **可选**：发布 PyPI 正式包名 / Homebrew 等二次分发

## 9. 关闭结论

原计划中的研究阶段 **0–6** 与附录「先动手清单」的**仓库侧交付**已合入主干；并行产品线完成 **M1–M5**。  
未改 plan 文件本身。剩余工作主要是组织侧私有数据与可选 full KPI / PyPI。
