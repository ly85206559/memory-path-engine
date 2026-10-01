# Stage 1 Case Spec — Contradiction / Exception（计划附录）

对照进展分析计划「下一阶段数据 / A. 阶段 1」：仓库内小型结构化 benchmark。  
本表是可机读 fixture 之上的**标注规格草稿**；权威期望仍以 JSON fixture 为准。

## 素材位置

| 类型 | 路径 |
| --- | --- |
| 例外覆盖合同 | `examples/graph_contract_pack/contracts/01_exception_override_contract.md` |
| 主文 vs 订单表冲突 | `examples/graph_contract_pack/conflict_contracts/` |
| 动态/例外 priming | `examples/contract_priming_pack/`、`examples/runbook` 相关 priming 文档 |
| Fixture | `benchmarks/structured_memory/*contradiction*`、`*exception*`、`*priming*` |

节点 id 约定：`{document_stem}:{section_number}`（与 ingest / domain pack 一致）。

## 高密度 case 规格（≥10；优先 Exception + Contradiction）

| case_id | fixture | query（摘要） | evidence_node_ids | required_contradiction_pairs | required_edge_types | required_semantic_roles | tags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exception-001 | exception_override_benchmark.json | 缺陷货物时谁覆盖 30 日付款规则？ | `:2`, `:3` | `[:1,:2]` | （trace 含 `exception_to` / `depends_on`） | exception, remedy | exception, override |
| exception-path-001 | exception_override_path_benchmark.json | 同上，锁路径形状 | path steps | 可选 | exception_to / depends_on | exception | exception, override, path |
| contradiction-001 | contradiction_tension_benchmark.json | Agreement 30 日 vs Order Form 15 日谁生效？ | `:2`, `:4` | `[:1,:4]`, `[:2,:4]` | contradicts | — | contradiction, tension |
| contradiction-002 | contradiction_tension_benchmark.json | 哪些条款在付款时限上冲突？ | `:1`, `:4` | `[:1,:4]` | contradicts | — | contradiction, multi_hop |
| contradiction-003 | contradiction_tension_benchmark.json | Order Form 与 Agreement 付款是否冲突？ | `:2`, `:4` | `[:2,:4]` | contradicts | — | contradiction, lexical_conflict |
| consolidation-001 | consolidation_gain_benchmark.json | 例外路由巩固增益 | route + exception | — | exception / route | — | consolidation, exception |
| route-001 | route_replay_benchmark.json | 例外覆盖路径回放 | exception path | — | exception_to | — | route, exception |
| research-001 | research_claim_chain_benchmark.json | 研究声明链上的矛盾 | claim nodes | 视 fixture | contradicts | — | research, contradiction |
| research-002 | research_claim_chain_benchmark.json | 例外/可解释性 | claim nodes | — | exception_to 等 | — | research, exception |
| prime-001…008 | contract_exception_priming_benchmark.json | 静态/动态 priming 序列 | priming nodes | 视 case | exception_to 等 | — | priming, exception |
| dynamic prime-001…006 | dynamic_override_sequence_benchmark.json | 动态覆盖序列 | sequence nodes | 视 case | exception_to 等 | — | priming, exception |

合计相关 case **≥ 23**（超过计划建议的 5–15 条高密度起步规模）。

## 验收命令

```bash
python scripts/generate_layer_b_report.py \
  --output /tmp/layer_b_report.json \
  --markdown-output /tmp/layer_b_report.md
python -m unittest discover -s tests
```

miss 分析：使用报告中的 path / palace snapshot 字段对照本表的 `required_*` 列。

## 扩展指引（可选）

若继续扩集：再加 1 篇合成 markdown（主文 vs 附件叙述差异 + 「不适用/除非」链），每篇 2–4 条 case，保持上表列齐全后再写入新 JSON fixture。
