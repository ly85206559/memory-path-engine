# Launch Post Pack

Ready-to-post copy for GitHub, X/Twitter, and Chinese developer communities. Keep all claims aligned with the current README and demos: this repo is a **research prototype**, not a production product or benchmark leaderboard project.

---

## 1) One-line positioning

**English:** Structured memory retrieval for AI agents that returns replayable evidence paths, not only flat `top-k` chunks.

**中文：**面向 AI 智能体的结构化记忆检索：不只返回 `top-k` 文本块，而是返回可逐步回放的证据路径。

---

## 2) Short English community post

Most retrieval stacks stop at `top-k` chunks and leave the reasoning implicit. I wanted a shape that makes the evidence chain visible, so I open-sourced **Memory Path Engine**.

It is a research-first prototype for agent memory that models memory as **typed nodes, edges, and weights**. Retrieval returns a **`MemoryPath`**: a stitched answer plus an ordered, replayable hop list with scores, `via=` edge types, and short reasons.

What is in the repo:

- lexical, embedding, structure-aware, and weighted graph retrieval modes
- replayable evidence paths instead of only final answers
- bundled runbook and contract demos
- repository-owned structured benchmark fixtures for experiments

Try it:

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
python -m memory_engine.demo --scenario contract
```

Repo: https://github.com/ly85206559/memory-path-engine

---

## 3) Concise English X post

Open-sourced **Memory Path Engine**.

It explores a simple idea: agent retrieval should return a **replayable evidence path**, not just `top-k` chunks.

Typed graph memory, weighted retrieval, runbook + contract demos.

```bash
python -m memory_engine.demo --scenario runbook
```

https://github.com/ly85206559/memory-path-engine

## 4) Technical English post

Most RAG stacks still flatten knowledge to chunks and stop at similarity search. **Memory Path Engine** asks a different question: can retrieval return a **walk** you can inspect?

In v0, bundled markdown packs are ingested into an in-memory graph (`MemoryNode` / `MemoryEdge` with weights). Retrievers return a composed answer and a **REPLAY PATH** with per-hop scores, edge types, and short reasons. The contract demo prints a flat baseline block and the path-aware result side by side so you can compare **evidence shape**, not only a single scalar.

The project is set up as a research harness: candidate generation, embedding backend, scoring, and path replay are separated so strategies can be compared without rewriting the whole loop. Current hypotheses include multi-hop gains over vanilla top-k, weighting effects, and explainability vs latency.

Honest scope: no production infra, no MCP, no multimodal, no large-scale benchmark claims, no full UI.

```bash
python -m pip install --no-build-isolation -e .
python -m unittest discover -s tests -v
python -m memory_engine.demo --scenario contract
```

Repo: https://github.com/ly85206559/memory-path-engine

---

## 5) Short Chinese community post

很多检索系统在返回 `top-k` 片段之后，就把真正的推理链条留给模型自己补了。**Memory Path Engine** 想试另一种做法。

这是一个面向 AI agent 记忆的研究原型：把记忆组织成**带类型的节点、边和权重**，检索结果不只给答案，还给一条**可回放的证据路径**，你可以看到每一步是怎么走到最终结果的。

仓库里现在有：

- lexical / embedding / structure-aware / weighted graph 多种检索模式
- 可回放的 evidence path 与逐步分数
- runbook 和 contract 两个内置 demo
- 自带结构化 benchmark fixtures，方便做实验比较

快速体验：

```bash
python -m pip install --no-build-isolation -e .
python -m memory_engine.demo --scenario runbook
```

仓库：https://github.com/ly85206559/memory-path-engine

## 6) Concise Chinese X post

开源了 **Memory Path Engine**。

它不是只返回 `top-k` 片段，而是给 AI agent 的检索结果加上一条**可回放证据路径**。

```bash
python -m memory_engine.demo --scenario runbook
```

https://github.com/ly85206559/memory-path-engine

## 7) Technical Chinese post

**Memory Path Engine** 是一个面向智能体记忆的研究原型：用图结构组织记忆单元，用多种检索策略做对照，并把结果组织成 **`MemoryPath`**，而不是只返回一堆相似片段。

v0 的核心形态是：

- 将示例 markdown 包摄入为内存图：`MemoryNode` / `MemoryEdge` 与权重
- 在同一套代码路径里比较词法 baseline、向量 baseline、结构遍历、加权图检索和激活传播实验
- 返回拼接答案 + 有序路径步骤，路径里有每步分数、`via` 边类型和简短理由

如果你在看 agent memory、multi-hop retrieval 或 explainable retrieval，这个方向会比较有意思。它不是生产级记忆中台，而是一个方便做实验、做对比、看路径形态的开源原型。

快速验证：

```bash
python -m pip install --no-build-isolation -e .
python -m unittest discover -s tests -v
python -m memory_engine.demo --scenario contract
```

仓库：https://github.com/ly85206559/memory-path-engine

---

## 8) Candidate titles / headlines

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

---

## 9) Asset usage

- `docs/assets/runbook-demo-terminal.svg`: README-friendly terminal capture of real stdout
- `docs/assets/social-banner.svg`: wide social banner (`1200×630`) for general social posts
- `docs/assets/open-graph-cover.svg`: **GitHub Social preview / Open Graph** layout (`1280×640`); export to PNG and upload under repo **Settings → General → Social preview** for a stable link-card image

## 10) Screenshot checklist

1. Use the runbook scenario for first impressions: `python -m memory_engine.demo --scenario runbook`
2. Keep the banner, query, `BEST ANSWER`, and first two replay steps visible
3. Keep `score=` and `via=` columns in frame
4. Use the contract scenario when you want baseline vs path-aware comparison
