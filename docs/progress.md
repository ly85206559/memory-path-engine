# Memory Path Engine — 当前进展分析

> 对照 [vision.md](vision.md) 三阶段路线图与 [benchmark-strategy.md](benchmark-strategy.md) 三层评测模型。  
> 本文件是可交付的现状快照；不取代历史 plan 文件。

**快照日期：** 与 `master` 同步的 Product M3 之后（含 MCP / hybrid / LongMemEval turn KPI）。

## 1. 定位与成熟度

| 维度 | 现状 |
| --- | --- |
| 定位 | 从研究原型转向 **本地记忆产品**（可回放路径） |
| 版本 | `pyproject.toml` → `0.4.0` |
| 产品入口 | `mpe` CLI、`.mpe/` SQLite palace、`mpe mcp`、hooks |
| 研究内核 | typed graph、`MemoryPath`、Palace v1、Stage 6 类脑机制 |

一句话：

> **公开召回做门面（Layer A）+ CLI/MCP 做闭环 + 路径/图做差异（Layer B）**。

## 2. 技术架构（两轨 + 产品面）

| 轨道 | 位置 | 状态 |
| --- | --- | --- |
| v0 图检索栈 | `schema` / `store` / `retrieve` / `scoring` / `replay` | 稳定；含 `hybrid` |
| Memory Palace v1 | `memory/` domain + application | 稳定；`palace_to_store` 桥接 |
| 产品持久化 | `persistence/` + `palace_workspace` | M1 已合入 |
| Agent 闭环 | `mcp_server` + `hooks_install` | M2 已合入 |
| Stage 6 机制 | multi-rep / forgetting / `PathReasoner` | 已合入 |

## 3. 愿景阶段对照（Stage 1–3）

| Vision | 计划含义 | 仓库现状 |
| --- | --- | --- |
| Stage 1 Memory-Augmented RAG | 结构 + 权重/异常 + 路径检索 | **完成**：多 retriever、`MemoryPath`、anomaly、加权/扩散 |
| Stage 2 Graph Memory System | 关系一等公民 + 受控激活 | **基本完成**：矛盾边、激活服务、domain packs；双轨 API 文档化于 `api-tracks.md` |
| Stage 3 Brain-like | 强化/衰减、多表示、query→path→answer | **MVP 完成**：状态机 + mild/aggressive + dual views + PathReasoner |

## 4. 评测三层对照（Layer A / B / C）

| Layer | 目标 | 现状 | 验收命令 |
| --- | --- | --- | --- |
| **A** 外部站位 | LongMemEval / HotpotQA | tiny+turn 基线已提交；medium 30/50q hybrid R@5≈0.96–1.0；full 复现配方在 README | `mpe bench longmemeval --label tiny [--granularity turn]` |
| **B** 机制验证 | path / semantic / contradiction / dynamic | fixtures + Layer B / ablation 报告脚本 | `python scripts/generate_layer_b_report.py` / `generate_ablation_report.py` |
| **C** 真实迁移 | 噪声文档 + 私有金标流程 | `layer_c_minimal` 可跑 + inventory/annotation 模板 | `python scripts/run_layer_c_benchmark.py` |

## 5. 计划阶段序列验收（阶段 0–6）

| 阶段 | 计划内容 | 结论 |
| --- | --- | --- |
| 0 基线守住 | CI + Layer B 不回退 | **通过**（`unittest` 187；Layer B/ablation/Layer C 脚本可跑） |
| 1 Layer B 机制 | 矛盾/异常可证伪 | **通过**（`contradiction_tension_*`、`exception_override_*`、anomaly 策略） |
| 2 消融工业化 | ablation + 延迟汇总 | **通过**（`generate_ablation_report.py`） |
| 3 Layer A 规模化 | 公开指标与架构指标分离 | **通过**（Layer A 报告 + nightly + turn KPI 配方） |
| 4 Layer C | 私有金标流程 | **脚手架通过**；私有 30 文档 / 120–180 case 仍属组织侧数据工作 |
| 5 Stage 2 收口 | domain pack + API 心智模型 | **通过**（contract/runbook/research packs + `docs/api-tracks.md`） |
| 6 Stage 3 | 类脑机制独立里程碑 | **MVP 通过**（见 Stage 6 模块与 `test_stage6_mechanisms.py`） |

## 6. 产品里程碑（并行线）

见 [ROADMAP.md](ROADMAP.md)：

- [x] M1 Persist + CLI + baseline skeleton  
- [x] M2 MCP + hooks + hybrid  
- [x] M3 LongMemEval turn + full 复现说明  
- [ ] M4 分发（pipx / Docker / backup）

## 7. 计划附录：你还需要准备的数据

对照原进展分析计划的「下一阶段数据」清单——**仓库侧已提供模板与公共 stand-in；私有金标仍需业务侧产出**：

### A. Layer B（仓库内，已具备）

- 合成 markdown + contradiction / exception fixture（`benchmarks/structured_memory/`）
- 稳定 node id 与 domain pack 边类型对齐

### B. Layer C 私有金标（不进公开仓）

按 [private-contract-dataset-guide.md](private-contract-dataset-guide.md) 与  
`benchmarks/layer_c_minimal/templates/`：

1. **合同 inventory**（约 30 份）  
2. **金标注释表**（约 120–180 case）  
3. 覆盖 multi-hop / exception / contradiction 题型  

清单与样例列：见 [layer-c-pilot-checklist.md](layer-c-pilot-checklist.md)。

### C. Layer A 公开集

不必自建；使用 `scripts/download_hotpotqa.py` / `download_longmemeval.py`。

## 8. 建议的下一步优先级

1. **产品 M4**：分发与运维（安装体验、Docker MCP、备份）  
2. **组织侧 Layer C**：填 inventory + 私有金标（不提交私密文本）  
3. **可选**：更强 embedding backend；full LongMemEval-S 数字写入 README 表（需下载后跑）

## 9. 关键结论

原计划中的研究阶段 0–6 **机制与评测闭环已合入主干**；并行产品线已完成 **M1–M3**。  
当前最大杠杆不在「再补一层研究机制」，而在：

- **M4 工程分发**（让陌生人 10 分钟装上用）  
- **私有 Layer C 金标**（证明机制在真实噪声下仍重要）
