# Final Launch Posts

This file is the **final long-form layer**: polished, channel-specific posts that are ready to publish with minimal or no editing.

For reusable message components, see [`launch-post.md`](launch-post.md).  
For shorter copy-paste variants, see [`launch-ready.md`](launch-ready.md).

Repo: [https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

Recommended preview image:

- `docs/assets/open-graph-cover.png` for GitHub Social Preview and link cards
- Generate it with: `python scripts/export_open_graph_cover.py`

## GitHub Discussion Launch Post

### Title

Memory Path Engine: replayable evidence paths for agent memory retrieval

### Body

I just open-sourced **Memory Path Engine**, a research-first prototype for **structured memory retrieval** in AI agents.

Most retrieval pipelines still stop at `top-k` chunks: split content, embed it, fetch the nearest matches, and leave the actual reasoning implicit. This project explores a different shape. Instead of treating memory as a flat index, it models memory as **typed nodes, edges, and weights**, then returns a `MemoryPath`: a stitched answer plus an ordered, replayable evidence path with per-step scores and edge types.

The goal is not to hide more logic inside the model. The goal is to make retrieval easier to **inspect, compare, and debug**, especially for multi-hop questions.

What is in the repo today:

- multiple retrieval modes in one codebase: lexical baseline, embedding baseline, structure-aware traversal, weighted graph retrieval, and activation spreading experiments
- replayable path output instead of only final answers
- bundled runbook and contract demos
- repository-owned benchmark fixtures for structured retrieval experiments

Quick start:

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

If you are interested in agent memory, graph-aware retrieval, or evidence-backed multi-hop reasoning, I would love feedback.

Repo: [https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

## 中文社区首发帖

### 标题

Memory Path Engine：给 AI Agent 的检索结果加上一条可回放证据路径

### 正文

刚把 **Memory Path Engine** 开源了。

这个项目想解决一个我自己一直觉得别扭的问题：很多检索流程做到 `top-k` 就结束了，最后答案虽然出来了，但证据链条还是隐含在模型里，不太容易看清楚它到底是怎么一步步连到结果的。

`Memory Path Engine` 想试一种更结构化的方式。它把记忆组织成**带类型的节点、边和权重**，检索时除了给出答案，还会返回一条**可回放的证据路径**，里面有每一步的分数、`via` 边类型和简短理由。这样做的重点不是“更花哨”，而是更方便做：

- 多跳检索实验
- baseline 对比
- miss case 分析
- 可解释的 retrieval 调试

目前仓库里已经有：

- lexical / embedding / structure-aware / weighted graph 多种检索模式
- runbook 和 contract 两个内置 demo
- 结构化 benchmark fixtures，方便自己做实验
- 一个统一的 demo 入口，直接能看到 path-aware 输出

最快可以这样体验：

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

如果你也在关注 agent memory、graph retrieval、multi-hop reasoning 或 explainable retrieval，欢迎看看，也欢迎直接提 issue / PR / 吐槽。

仓库地址：

[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)