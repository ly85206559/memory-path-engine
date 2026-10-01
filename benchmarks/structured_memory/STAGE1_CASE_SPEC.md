# Stage 1 Case Spec — Contradiction / Exception（计划附录）

对照进展分析计划「下一阶段数据 / A. 阶段 1」：仓库内小型结构化 benchmark。  
本表是可机读 fixture 之上的**标注规格**；权威期望以 JSON fixture 为准。

## 素材位置

| 类型 | 路径 |
| --- | --- |
| **新合成合同（计划附录交付）** | `examples/graph_contract_pack/stage1_plan_pack/03_warranty_attachment_tension_contract.md` |
| **新 JSON fixture（10 case）** | `benchmarks/structured_memory/stage1_plan_appendix_benchmark.json` |
| 既有例外覆盖合同 | `examples/graph_contract_pack/contracts/01_exception_override_contract.md` |
| 既有主文 vs 订单表冲突 | `examples/graph_contract_pack/conflict_contracts/` |
| 既有 priming | `examples/contract_priming_pack/` 等 |

节点 id 约定：`{document_stem}:{section_number}`（与 ingest / domain pack 一致）。

## 新 fixture 高密度 case（10 条，Exception + Contradiction）

| case_id | query（摘要） | evidence | required_contradiction_pairs | required_edge_types | required_semantic_roles | tags |
| --- | --- | --- | --- | --- | --- | --- |
| stage1-exception-001 | refurbished 覆盖 12 月质保？ | `:5`, `:1` | — | exception_to | exception | exception, override |
| stage1-exception-002 | Unless Attachment 否则协议质保？ | `:3` | — | — | exception | exception, unless |
| stage1-contradiction-001 | Agreement vs Attachment 付款谁生效？ | `:4`, `:8` | `[:4,:8]` | contradicts | — | contradiction, payment |
| stage1-contradiction-002 | 哪些条款在付款上冲突？ | `:4`, `:8` | `[:4,:8]` | contradicts | — | contradiction, multi_hop |
| stage1-condition-001 | refurbished 失效 Seller 须做什么？ | `:6`, `:5` | — | depends_on | condition | condition, remedy |
| stage1-exception-003 | Except 未通知时是否须兑现 Attachment 救济？ | `:9` | — | — | exception | exception, notify |
| stage1-contradiction-003 | Attachment 付款是否与协议冲突？ | `:4`, `:8` | `[:4,:8]` | contradicts | — | contradiction, lexical |
| stage1-exception-004 | refurbished 质保时长（notwithstanding）？ | `:5` | — | — | exception | exception, refurbished |
| stage1-exception-005 | 哪条对 12 月质保构成例外？ | `:5`/`:3` | — | — | exception | exception |
| stage1-contradiction-004 | Attachment 付款 shall control？ | `:8` | `[:4,:8]` | contradicts | — | contradiction, control |

既有 contradiction / exception / priming fixtures 另有 **≥23** 条相关 case，合计远超计划建议的 5–15 条起步规模。

## 验收命令

```bash
python scripts/generate_layer_b_report.py \
  --output /tmp/layer_b_report.json \
  --markdown-output /tmp/layer_b_report.md
python -m unittest tests.test_benchmark_fixtures.BenchmarkFixtureTests.test_stage1_plan_appendix_fixture_covers_exception_and_contradiction
```

`activation_spreading_v1` / `weighted_graph` / `hybrid` 在本新 fixture 上 evidence+expectation hit_rate = 1.0。
