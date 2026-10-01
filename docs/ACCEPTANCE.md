# Acceptance checklist (Product M1 + M2)

Use this page to验收当前可交付版本。全部勾选即可认为 **工程闭环 MVP** 可验收；
公开榜单冲高分属于 M3，不阻塞本次验收。

## A. Local palace + CLI

- [ ] `pip install -e .` 后 `mpe --help` 可用
- [ ] `mpe init && mpe ingest examples/runbook_pack/runbooks --pack example_runbook_pack`
- [ ] `mpe status` 显示 nodes/edges > 0
- [ ] `mpe search "..."` 输出 ANSWER + PATH hops
- [ ] `mpe path "..."` 强调 hop 列表
- [ ] `echo hi | mpe memo --title t` 后 `mpe status` nodes 增加

## B. Hybrid + reinforce

- [ ] `mpe search "..." --mode hybrid` 可运行
- [ ] `mpe reinforce "..." --policy mild` 返回结果（可含 lifecycle_snapshots）

## C. MCP closed loop

- [ ] `mpe hooks install` 生成 `.cursor/mpe-hooks/`
- [ ] `mpe mcp` 可启动（stdio；Ctrl+C 退出）
- [ ] MCP tools 列表包含 `mpe_search` / `mpe_ingest_memo` / `mpe_reinforce`
- [ ] 通过 MCP 或 CLI 完成一次 ingest → search 闭环

## D. Public KPI skeleton

- [ ] `mpe bench longmemeval --label tiny` 写出 JSON + Markdown
- [ ] 报告中含 `product_kpi: true` 与 R@5 列
- [ ] README / `docs/ROADMAP.md` 标明 M3 才冲 full LongMemEval

## E. Regression

- [ ] `python -m unittest discover -s tests -v` 全绿

## Out of scope for this acceptance

- Docker 镜像、多向量后端
- LongMemEval-S full 96%+ 对标数字
- LLM 生成答案（PathReasoner 保持确定性）
