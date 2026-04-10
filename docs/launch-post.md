# Launch Post Pack

This file is the **message toolkit** for outbound sharing. It is the source-of-truth for positioning, claims, titles, assets, and reusable copy blocks.

Use the other two files like this:

- `launch-ready.md`: short copy-paste versions for fast posting
- `final-launch-posts.md`: polished long-form posts for the two main channels

Keep everything here aligned with the current README and demos: this repo is a **research prototype**, not a production product or leaderboard claim.

## 1) Core positioning

**English**

Structured memory retrieval for AI agents that returns replayable evidence paths, not only flat `top-k` chunks.

**中文**

面向 AI 智能体的结构化记忆检索：不只返回 `top-k` 文本块，而是返回可逐步回放的证据路径。

## 2) Core message blocks

### Problem framing

**English**

Most retrieval stacks stop at `top-k` chunks and leave the reasoning implicit.

**中文**

很多检索系统在返回 `top-k` 片段之后，就把真正的证据链条留给模型自己补了。

### What this repo does

**English**

`Memory Path Engine` models memory as typed nodes, edges, and weights, then returns a `MemoryPath`: a stitched answer plus an ordered, replayable evidence path with per-step scores and edge types.

**中文**

`Memory Path Engine` 把记忆组织成带类型的节点、边和权重，检索时返回 `MemoryPath`：不仅有拼接后的答案，还有一条可回放的证据路径，包含每一步的分数和边类型。

### Honest scope

**English**

This is a research-first prototype for agent memory, not production infrastructure, not a full UI, and not a large-scale benchmark platform.

**中文**

这是一个面向 agent memory 的研究原型，不是生产级基础设施，不是完整 UI，也不是大规模 benchmark 平台。

## 3) Reusable feature bullets

**English**

- lexical, embedding, structure-aware, and weighted graph retrieval modes
- replayable path output instead of only final answers
- bundled runbook and contract demos
- repository-owned benchmark fixtures for structured retrieval experiments

**中文**

- lexical / embedding / structure-aware / weighted graph 多种检索模式
- 可回放的路径输出，而不只是最终答案
- runbook 和 contract 两个内置 demo
- 自带结构化 benchmark fixtures，方便做实验比较

## 4) Reusable CTA block

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

Repo: [https://github.com/ly85206559/memory-path-engine](https://github.com/ly85206559/memory-path-engine)

## 5) Candidate titles / headlines

**English**

1. Memory Path Engine: retrieval that returns evidence paths, not just top-k chunks
2. From flat top-k retrieval to replayable evidence paths
3. A graph-aware retrieval prototype for inspectable agent memory
4. Structured memory for agents with weighted graph paths
5. Runbook + contract demos for path-aware retrieval experiments

**中文**

1. Memory Path Engine：不只 top-k，让智能体记忆检索可回放
2. 从扁平检索到证据路径：一个结构化记忆原型
3. 多跳证据怎么连起来？用图和权重做可解释检索
4. 面向 Agent 的结构化记忆实验：runbook 与合同双场景 demo
5. 一个仓库比较多种检索基线：词法、向量、结构遍历、加权图

## 6) Asset usage

- `docs/assets/runbook-demo-terminal.svg`: README-friendly terminal capture of real stdout
- `docs/assets/social-banner.svg`: wide social banner (`1200×630`) for general social posts
- `docs/assets/open-graph-cover.png`: GitHub Social Preview / link-card image

## 7) Screenshot checklist

1. Use the runbook scenario for first impressions: `python -m memory_engine.demo --scenario runbook`
2. Keep the banner, query, `BEST ANSWER`, and first two replay steps visible
3. Keep `score=` and `via=` columns in frame
4. Use the contract scenario when you want baseline vs path-aware comparison

