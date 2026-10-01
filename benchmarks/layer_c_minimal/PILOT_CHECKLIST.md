# Layer C 私有试点清单（对照进展分析计划）

本清单落实「阶段 4 — 不进仓库的 Layer C 私有金标」准备工作。  
公开 stand-in 与 CSV 模板已在本目录的 `templates/`；**真实合同文本不要提交到公开仓库**。

## 1. 文档 inventory（目标约 30 份）

复制并本地填写：

```bash
cp benchmarks/layer_c_minimal/templates/inventory_example.csv \
   /path/to/private/layer_c_inventory.csv
```

必填列（与 [private-contract-dataset-guide.md](../docs/private-contract-dataset-guide.md) 一致）：

| 列 | 说明 |
| --- | --- |
| `doc_id` | 稳定文档 id |
| `file_path` | 私有存储路径（勿指向本公开仓） |
| `contract_type` / `language` | 类型与语言 |
| `exception_density` | low/medium/high |
| `selected_for_pilot` | yes/no |
| `selection_reason` | 为何进入试点 |

排除：无法 OCR 的扫描件、过度涂黑、无稳定版本。

## 2. 金标注释表（目标 120–180 case，约 4–6 题/文档）

```bash
cp benchmarks/layer_c_minimal/templates/annotation_sheet_example.csv \
   /path/to/private/layer_c_annotations.csv
```

每行至少包含：

- `case_id`, `doc_id`, `query`
- `evidence_node_ids`（与 ingest 后 node id 一致）
- `case_family`：`multi_hop` / `exception_override` / `contradiction_tension` 等
- 可选：`required_edge_types`, `required_semantic_roles`, `required_contradiction_pairs`
- `review_status`

## 3. 题型覆盖（试点最低要求）

| 族 | 最少 case 数（建议） | 目的 |
| --- | ---: | --- |
| Multi-hop | 30 | H1：图路径优于 flat top-k |
| Exception/override | 30 | 异常边与语义角色 |
| Contradiction/tension | 20 | 显式矛盾对 |
| 其他（payment / termination 等） | 余量 | 多样性 |

## 4. 跑通契约（本地 / 私有 runner）

公开 stand-in 先验证 runner：

```bash
python scripts/run_layer_c_benchmark.py \
  --output /tmp/layer_c_report.json \
  --markdown-output /tmp/layer_c_report.md
```

私有数据：在私有镜像中替换 `documents/` 与 JSON fixture，**保持与 Layer B 相同的 `expectation` 字段**，以便同一 runner 消费。

## 5. 完成定义（组织侧）

- [ ] inventory ≥ 30 且 `selected_for_pilot=yes` 分层合理  
- [ ] annotations ≥ 120，三类题型达标  
- [ ] 双人抽检 `review_status=approved` 比例可接受  
- [ ] 本地 Layer C 报告可复现；结论写入内部实验记录（不强制公开）

公开仓库侧阶段 4 **脚手架已完成**；上表勾选属于业务数据工作，不阻塞产品 M4。
