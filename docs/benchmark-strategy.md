# Benchmark Strategy

This document defines how `memory-path-engine` should prove that its current retrieval and memory features are effective.

The short version:

- public benchmarks prove external comparability
- repository-owned structured benchmarks prove the architecture claims of this project
- private real-world datasets prove that the same claims still matter on realistic documents

Use this file as the strategy layer. For concrete metrics, retriever modes, and benchmark runner behavior, see [`evaluation.md`](evaluation.md). For current repository-owned fixtures, see [`../benchmarks/structured_memory/README.md`](../benchmarks/structured_memory/README.md).

## Why benchmark now

The project already has enough moving parts that qualitative demos are no longer sufficient:

- multiple retriever modes
- typed benchmark models
- path, semantic, contradiction, and activation diagnostics
- static vs dynamic memory experiment modes

That means the next useful question is no longer "does the demo look interesting?" but "which design claims are actually supported by repeatable evidence?"

## What must be proven

This strategy is anchored to the research hypotheses in [`hypotheses.md`](hypotheses.md):

- `H1`: graph-aware retrieval should beat flat `top-k` retrieval on multi-hop questions
- `H2`: weighting and anomaly-aware scoring should improve critical evidence recall without unacceptable latency
- `H3`: replayable `MemoryPath` output should be causally useful, not cosmetic

The current benchmark stack should also prove an additional v0.5 claim:

- dynamic memory state should produce measurable retrieval differences across ordered query sequences

## Three-Layer Benchmark Model

### Layer A: Public benchmarks

Purpose:

- give the project an external comparison point
- show that graph-aware memory ideas do not collapse on standard retrieval or memory tasks
- make it possible to compare against benchmark-first projects

What this layer is good for:

- general long-context or multi-hop QA
- evidence retrieval sanity checks
- broad memory-system positioning

What this layer is not good for:

- typed edge correctness
- exception or contradiction semantics
- replayable path-shape validation
- dynamic priming effects specific to this project

Recommended use:

- treat public benchmarks as external sanity checks, not as the only source of truth
- report answer metrics, evidence metrics when available, and latency
- be explicit about what each dataset does not validate

### Layer B: Repository-owned structured benchmarks

Purpose:

- directly test the architecture claims unique to `memory-path-engine`
- keep evaluation aligned with the domain model and `MemoryPath`
- support TDD, regressions, and explainable miss analysis

This is the most important layer for the current stage because it can measure things public datasets usually cannot:

- path validity
- semantic-role coverage
- contradiction and exception surfacing
- activation-trace behavior
- static vs dynamic divergence on ordered cases

Current examples live in [`../benchmarks/structured_memory`](../benchmarks/structured_memory):

- `example_contract_benchmark.json`
- `example_runbook_benchmark.json`
- `exception_override_benchmark.json`
- `multi_hop_chain_benchmark.json`
- `dynamic_memory_priming_benchmark.json`
- `contract_exception_priming_benchmark.json`

### Layer C: Private real-world datasets

Purpose:

- validate external realism
- test whether synthetic conclusions survive real document noise
- support domain-specific decisions without forcing sensitive data into the public repo

This layer is where you prove the system matters on actual contracts, SOPs, runbooks, policy docs, or internal knowledge bases.

Private datasets should not replace Layer B. They complement it:

- Layer B proves the mechanism
- Layer C proves the mechanism still matters in practice

## Recommended Public Benchmark Shortlist

### Strong fit

#### `HotpotQA`

Best for:

- multi-document retrieval
- bridge-style multi-hop questions
- evidence-support evaluation

What it validates:

- whether graph-aware retrieval helps find linked evidence

What it does not validate:

- typed graph semantics
- dynamic memory
- exact `MemoryPath` correctness

Integration difficulty: medium to high

Recommended reported metrics:

- answer EM/F1
- evidence recall or support-sentence recall
- latency

#### `MuSiQue`

Best for:

- harder compositional multi-hop retrieval
- robustness under distractors

What it validates:

- whether retrieval still works when reasoning requires several linked facts

What it does not validate:

- exception or contradiction logic
- path replay fidelity
- sequential memory effects

Integration difficulty: high

Recommended reported metrics:

- answer EM/F1
- evidence recall
- breakdown by hop count or question type

#### `2WikiMultiHopQA`

Best for:

- explicit multi-hop retrieval patterns
- relation-following style questions

What it validates:

- whether graph-style traversal gives a meaningful advantage over flat retrieval

What it does not validate:

- internal semantic roles
- dynamic priming behavior

Integration difficulty: medium to high

Recommended reported metrics:

- answer EM/F1
- evidence recall
- latency

### Partial fit

#### `HoVer`

Useful for:

- hierarchical evidence retrieval
- multi-step claim verification

Useful but incomplete because:

- it is closer to evidence verification than to replayable memory paths

#### `LoCoMo`

Useful for:

- long conversational memory
- timeline and history-sensitive recall

Useful but incomplete because:

- it does not naturally validate edge types, path shapes, or exception semantics

#### `LongMemEval`

Useful for:

- broad memory-system sanity checks
- external positioning against long-memory systems

Useful but incomplete because:

- it is much broader than your current architectural claims
- it reports retrieval sanity and external positioning, not path/semantic/dynamic correctness

### Weak fit

These are worth using only as secondary background checks:

- `BEIR` or `MS MARCO`: good for flat retrieval baselines, weak for graph/path claims
- `FEVER`: useful for evidence retrieval, but not for your main graph-memory story
- `SQuAD`, `Natural Questions`, similar QA sets: weak fit for path and structured-memory validation

## Recommended Evaluation Contract

Use [`evaluation.md`](evaluation.md) as the source of truth for the metric definitions. At strategy level, the project should consistently report:

- `answer_recall` or task-native answer score on public datasets
- `evidence_recall` or `evidence_hit_rate`
- `path_hit_rate` on repository-owned and private structured datasets
- `semantic_hit_rate` where semantic roles matter
- `contradiction_hit_rate` where exception or conflict cases exist
- `activation_trace_hit_rate` for spreading-based experiments
- `avg_latency_ms`

Current implemented public benchmark adapters:

- `HotpotQA` retrieval-only evidence benchmark with local/nightly full-dev support
- `LongMemEval` retrieval-only session benchmark (`R@5`, `R@10`, `NDCG@10`) for external positioning

At minimum, every benchmark report should make it easy to compare:

- `lexical_baseline`
- `embedding_baseline`
- `structure_only`
- `weighted_graph`
- `activation_spreading_v1`
- `weighted_graph_static`
- `weighted_graph_dynamic`
- `activation_spreading_static`
- `activation_spreading_dynamic`

## Private Dataset Strategy

Private datasets should be built in two buckets:

### 1. Golden set

This is the high-quality, manually labeled evaluation set.

Properties:

- small to medium size
- carefully reviewed
- stable over time
- used for milestone decisions and release comparisons

Suggested first size:

- `20-50` cases per document family for the first real version

### 2. Shadow set

This is a larger, lower-touch set used for replay and drift detection.

Properties:

- less detailed labeling
- easier to refresh
- useful for regression replay
- not necessarily suitable for publishable claims

## How To Build A Private Dataset

### Step 1: Choose document families

Start with document types that match your intended memory use cases:

- master service agreements
- policy or compliance documents
- operational runbooks
- incident postmortems
- support procedures

Keep each dataset focused. Do not mix unrelated document families into one benchmark unless cross-document retrieval is the explicit point.

### Step 2: Freeze document snapshots

Before writing cases, freeze a stable version of the source documents:

- assign a dataset version
- assign stable file names
- keep document hashes or archival copies internally

This avoids silent benchmark drift.

### Step 3: Define stable node IDs

Follow the same pattern already used in the repo:

- `{document_stem}:{unit_number}`

Examples:

- `msa_v3:12`
- `incident_runbook_api:7`

Every gold label should point to these stable IDs, not to raw prose fragments.

### Step 4: Write cases around retrieval behavior, not only final answers

Each case should include:

- `query`
- `evidence_node_ids`
- optional `path`
- optional `required_edge_types`
- optional `required_semantic_roles`
- optional `required_contradiction_pairs`
- optional `activation_trace`

That keeps the benchmark aligned with `memory-path-engine` rather than reducing everything to answer strings.

## Gold Labeling Rules

### Evidence labels

Use `evidence_node_ids` for the minimum nodes that must support the answer.

Guidelines:

- label direct evidence, not every related node
- for multi-hop questions, include every essential support node
- use `minimum_evidence_matches` to control whether all or only some evidence must be found

### Path labels

Use `path` when "finding the right evidence" is not enough and "walking the right route" matters.

Good uses:

- runbook next-step chains
- contract exception override chains
- dependency chains

Avoid path labels when multiple path shapes are equally valid and the benchmark would become brittle.

### Semantic-role labels

Use `required_semantic_roles` when you want to confirm that the system surfaced the right semantic class:

- `exception`
- `remedy`
- `condition`
- `escalation`

### Contradiction labels

Use `required_contradiction_pairs` when the value of the case depends on the system surfacing a tension between nodes:

- general rule vs exception
- baseline obligation vs override
- normal flow vs emergency override

### Dynamic sequence labels

For dynamic-memory cases, place several `prime-*` cases before a final `probe-*` case.

Guidelines:

- the prime cases should reinforce one region of the graph
- the probe should test later behavior, not restate the same question
- compare static and dynamic modes on the same ordered dataset

## Annotation Quality Rules

To keep private gold labels trustworthy:

- require at least two annotators for high-value cases
- resolve disagreements with a short written rationale
- write a one-page internal annotation guide before scaling
- use tags like `multi_hop`, `exception`, `contradiction`, `sequential`, `probe`
- include hard negative cases with similar wording but different evidence

## Compact Example Cases

### Contract case

```json
{
  "case_id": "private-msa-termination-001",
  "query": "If the customer does not cure a material breach after notice, what happens next?",
  "tags": ["contract", "termination", "multi_hop"],
  "expectation": {
    "evidence_node_ids": ["acme_msa:14", "acme_msa:15"],
    "minimum_evidence_matches": 1,
    "path_scope": "best_path",
    "path": {
      "match_mode": "prefix",
      "steps": [
        { "node_id": "acme_msa:15", "via_edge_type": null },
        { "node_id": "acme_msa:14", "via_edge_type": "depends_on" }
      ]
    }
  }
}
```

### Exception case

```json
{
  "case_id": "private-payment-exception-001",
  "query": "Does the standard 30-day payment rule still apply when delivered goods are defective?",
  "tags": ["contract", "exception", "contradiction"],
  "expectation": {
    "evidence_node_ids": ["supply_terms:3", "supply_terms:4"],
    "minimum_evidence_matches": 2,
    "required_semantic_roles": ["exception", "remedy"],
    "required_contradiction_pairs": [
      ["supply_terms:2", "supply_terms:3"]
    ]
  }
}
```

### Runbook sequence case

```json
{
  "case_id": "private-runbook-probe-001",
  "query": "What comes after beta cache diagnostics?",
  "tags": ["runbook", "sequential", "probe"],
  "expectation": {
    "evidence_node_ids": ["incident_beta_runbook:7"],
    "minimum_evidence_matches": 1,
    "path_scope": "best_path",
    "path": {
      "match_mode": "prefix",
      "steps": [
        { "node_id": "incident_beta_runbook:7", "via_edge_type": null },
        { "node_id": "incident_beta_runbook:6", "via_edge_type": "depends_on" }
      ]
    }
  }
}
```

## Suggested First Milestone

A practical first benchmark package would be:

- `1-2` public benchmarks for external sanity checking
- all current repository-owned structured benchmarks as the architectural regression suite
- `20-50` private gold cases across one contract family and one runbook family

That is enough to support a credible claim that the system is both:

- benchmarked in public
- tested on architecture-specific fixtures
- validated on realistic private material

## Phased evaluation plan (weekly)

The table below is a **default rolling schedule** for a small team. Adjust dates to your calendar; keep the **ordering**: Layer B gates before Layer A scale, and Layer C in parallel once Layer B is stable.

| Week | Primary layer | What to run | Gate (pass / fail) | Notes |
|------|---------------|-------------|--------------------|-------|
| 0 | B | Full unit suite + load all `benchmarks/structured_memory/*.json` | CI green; every fixture loads | Baseline regression lock |
| 1 | B | Same + `run_suite` on key modes for each fixture (see below) | `evidence_hit_rate` and `path_hit_rate` each no worse than `-2%` absolute vs last tagged baseline; `avg_latency_ms` no worse than `+20%` | Use `docs/evaluation.md` mode list |
| 2 | A (HotpotQA v0) | Adapter + **2–5** hand-checked examples in tests (no full download) | Unit tests pass; gold `supporting_facts` → `node_id` mapping verified | See appendix HotpotQA v0 |
| 3 | A | **Dev distractor** tiny split (`32–128` items) in CI or skip-if-missing | Pipeline runs end-to-end; `embedding_baseline.evidence_hit_rate >= lexical_baseline.evidence_hit_rate`; both latencies captured | Retrieval-only v0 |
| 4 | A | Full **dev distractor** locally (not necessarily CI) | Table of `evidence_hit_rate` by `type` (`bridge` / `comparison`) + latency | Still no EM/F1 unless you add a reader |
| 5 | B + A | Ablations from `evaluation.md` on B fixtures + HotpotQA tiny | At least one targeted Layer B fixture must show the expected direction for each ablation family: no-structure, no-weight, no-path-expansion | Document failures explicitly |
| 6 | C | Private **golden** pilot `20` cases | Dual annotation complete; evidence-label agreement `>= 0.85`; review-complete set exported to benchmark JSON | Aggregate only if policy allows |

**Recommended minimum mode matrix per Layer B run** (when comparing architecture, not only smoke):

- `lexical_baseline`, `embedding_baseline`, `structure_only`, `weighted_graph`, `activation_spreading_v1`
- For dynamic claims only: `activation_spreading_static` vs `activation_spreading_dynamic` on priming fixtures

**Gate philosophy**

- **Hard gate**: anything that already has CI today (tests + JSON load) must stay green.
- **Soft gate**: public benchmark deltas are reported honestly; do not block merges on HotpotQA leaderboard scores until the adapter is stable.
- **Release gate** (optional): before tagging `v0.x`, require Week 1 B-suite comparison + Week 3 HotpotQA tiny evidence metrics logged in release notes.

## Suggested run matrix

- `benchmarks/structured_memory/*.json`: main CI
- `benchmarks/external/hotpotqa/hotpot_tiny_fixture.json`: main CI smoke
- `benchmarks/external/hotpotqa/data/*.json`: nightly or local only
- `benchmarks/external/longmemeval/longmemeval_tiny_fixture.json`: local smoke
- `benchmarks/external/longmemeval/data/*.json`: manual or nightly after the adapter stabilizes

For a contract-focused private rollout plan, including inventory template, sampling rules, and golden annotation workflow, see [`private-contract-dataset-guide.md`](private-contract-dataset-guide.md).

## Appendix: HotpotQA v0 minimal integration

First public benchmark should be **HotpotQA dev distractor** (`hotpot_dev_distractor_v1.json`), not fullwiki: each question ships with a fixed context block (two gold paragraphs plus distractors), which matches per-sample `MemoryStore` construction without a corpus-wide index.

**Data**

- Official splits and format: [HotpotQA](https://hotpotqa.github.io/). Typical fields: `question`, `answer`, `context` as `[[title, [sentences...]], ...]`, `supporting_facts` as `[[title, sent_id], ...]` with `sent_id` **0-based within that title’s sentence list**.
- License and citation: copy the **exact** terms from the official release into `benchmarks/external/hotpotqa/README.md` (do not paraphrase legal text).
- HuggingFace option: dataset id such as `hotpotqa/hotpot_qa` can simplify download; pin revision for reproducibility.

**Ingestion (v0)**

- **One sentence → one `MemoryNode`**: `node_id = "{normalized_title}:{sent_idx}"` for sentences under each title in **that sample’s** `context`.
- **Edges (v0)**: within the same title only, chain adjacent sentences (`title:s` → `title:s+1`). Do not add cross-title cliques in v0.
- **Normalization**: implement one shared `normalize_title` for both ingestion and gold mapping; watch Unicode, spacing, and punctuation.

**Gold → benchmark**

- For each `[title, sent_id]` in `supporting_facts`, resolve to the same `node_id` scheme; dedupe into `evidence_node_ids`.
- `minimum_evidence_matches`: pick one policy and keep it fixed in reports (`all gold sentences` vs `at least one`).

**Metrics**

- **Retrieval-only phase** (no answer generation): report `evidence_hit` / `evidence_hit_rate` aligned with `StructuredBenchmarkRunner`, plus **recall@k** only after you define what “top-k” means for your retriever (best path steps vs union of paths—document the choice).
- **EM / F1**: report only after you add a reader that produces a span or short answer and reuse HotpotQA’s official normalization.

**Repo layout (suggested)**

- `benchmarks/external/hotpotqa/README.md` — download, license, metrics, commands
- `benchmarks/external/hotpotqa/hotpot_tiny_fixture.json` — two-item sanity fixture for CI-style tests
- `scripts/download_hotpotqa.py` — optional fetch + checksum (not required for v0)
- `src/memory_engine/benchmarking/adapters/hotpotqa.py` — implemented: sample → `MemoryStore`, gold mapping, `run_hotpotqa_benchmark`

**CI**

- Use a **tiny** slice (`32–128` dev examples) or `skipif` when data absent; full dev set stays a **local / nightly** job.

**Pitfalls**

- Do not merge all dev questions into one global graph (leaks cross-question structure).
- Distractor setting: most nodes are negatives; compare modes under the same `top_k` policy.
- Title matching between `context` and `supporting_facts` must be identical after normalization.

## Reporting Guidance

When you publish or present results:

- use public benchmarks for external comparison
- use repository-owned benchmarks to prove structure, path, and dynamic-memory claims
- use private datasets to show realistic value, but report them carefully and transparently

Good practice:

- publish aggregate numbers
- describe the private data types and annotation protocol
- include a few anonymized example cases if policy allows
- never let private-only results become the sole support for a general technical claim
