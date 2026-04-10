# Launch-ready Posts

This file is the **short-form copy-paste layer**. Use it when you want to post quickly without rewriting.

For the full message toolkit, see [`launch-post.md`](launch-post.md).  
For polished long-form channel posts, see [`final-launch-posts.md`](final-launch-posts.md).

**Repo:** [https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)  
**Suggested image for link cards:** `docs/assets/open-graph-cover.png` (generate it with `python scripts/export_open_graph_cover.py`, then upload it in GitHub **Settings → General → Social preview**).

---

## English — medium post

I just open-sourced **Memory Path Engine**, a small research prototype for **agent memory retrieval**.

The usual pattern is chunk -> embed -> `top-k`, then the model fills in the reasoning. Here the memory is a **typed graph** (nodes, edges, weights), and retrieval returns a `MemoryPath`: a stitched answer plus a **replayable hop list** (scores, `via=` edge types, short reasons). That makes multi-hop evidence easier to **inspect** than hiding everything inside the model.

**In the repo**

- Multiple retrieval modes in one codebase (lexical, embedding, structure-aware, weighted graph, activation spreading v1)
- Bundled **runbook** and **contract** demos
- Structured benchmark fixtures for your own experiments

**Try it**

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)  
MIT · feedback welcome

---

## English — short post

Open-sourced **Memory Path Engine**: agent memory retrieval that returns **replayable evidence paths**, not just `top-k` chunks.

Typed graph + weighted retrieval. Runbook + contract demos.

```bash
python -m memory_engine.demo --scenario runbook
```

[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

---

## 中文 — 中等长度帖子

刚把 **Memory Path Engine（记忆路径引擎）** 开源了。

很多做法还是：切 chunk、做向量、`top-k` 检索，然后让模型自己把推理补上。这个项目想试的是：把记忆建成**带类型的图**（节点、边、权重），检索时除了答案，再给一条**可回放的证据路径**——每一步有分数、`via` 边类型和简短理由，方便看多跳证据是怎么连起来的。

**仓库里有什么**

- 同一套代码里对齐多种检索：词法、向量、结构遍历、加权图、激活传播实验
- 自带 **runbook** 和 **合同** 两类 demo
- 结构化 benchmark 夹具，方便自己做对比实验

**一条命令体验**

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

仓库：[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)  
MIT，欢迎 issue / PR / 吐槽

---

## 中文 — 短帖

开源 **Memory Path Engine**：面向 AI agent 的结构化记忆检索，返回**可回放证据路径**，而不只是 `top-k` 片段。

```bash
python -m memory_engine.demo --scenario runbook
```

[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

---

## English — ultra-short post

Open-sourced **Memory Path Engine**.

Structured memory retrieval for agents with **replayable evidence paths** instead of flat `top-k` chunks.

[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

## 中文 — 超短帖

开源 **Memory Path Engine**：给 AI agent 的检索结果加上一条**可回放证据路径**，而不只是 `top-k` 片段。

仓库：[https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)