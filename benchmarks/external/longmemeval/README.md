# LongMemEval (external benchmark)

This directory holds documentation for the **LongMemEval adapter** in `memory_engine.benchmarking.adapters.longmemeval`.

## What is implemented

- Session-level `MemoryStore` construction from official LongMemEval JSON fields
- One node per history session, connected by `next_session` edges in timestamp order
- Retrieval-only evaluation against `answer_session_ids`
- Public benchmark metrics: `R@5`, `R@10`, `NDCG@10`, `avg_latency_ms`

## Data

Download the cleaned LongMemEval-S release from the official dataset:

- `longmemeval_s_cleaned.json`

You can download it into the repository's external-benchmark area with:

```bash
python scripts/download_longmemeval.py
```

## Run (local)

Run the checked-in tiny fixture:

```bash
python scripts/run_longmemeval_benchmark.py
```

Run a downloaded official file:

```bash
python scripts/run_longmemeval_benchmark.py --dataset "benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json" --limit 50 --top-k 10 --modes lexical_baseline,embedding_baseline,weighted_graph,hybrid,activation_spreading_v1
```

Pretty-print the full suite JSON:

```bash
python scripts/run_longmemeval_benchmark.py --pretty
```

Write the full suite report JSON to a file:

```bash
python scripts/run_longmemeval_benchmark.py --output "benchmarks/external/longmemeval/data/local-report.json"
```

Write a compact summary JSON for dashboards or nightly artifacts:

```bash
python scripts/run_longmemeval_benchmark.py --summary-output "benchmarks/external/longmemeval/data/local-summary.json"
```

## Nightly note

The repository now includes `longmemeval-nightly.yml` for scheduled or manual runs against the downloaded cleaned file. It uploads both the full suite report and a compact summary artifact so the LongMemEval baseline can be tracked continuously.

Nightly defaults to a **medium** slice (`50` samples). Use `slice_profile=full` for the complete file. Summaries include `metric_scope=external_positioning` and an explicit disclaimer that path/semantic/contradiction claims belong to Layer B / Layer C.

## Important limitation

This adapter supports **session** and **turn** granularity, and is **retrieval-only**:

- `session`: gold = `answer_session_ids` mapped to session nodes
- `turn`: gold = `has_answer` turns inside those sessions (drawer-like units)
- it does **not** run answer generation or official QA grading
- Layer B / Layer C still own path, semantic, contradiction, and dynamic-memory claims

## Public recall KPI snapshots

Committed medium-slice KPI tables (no raw dataset):

- `baselines/longmemeval_kpi_medium30.md` — 30q session (ngram)
- `baselines/longmemeval_kpi_medium50.md` — 50q session (ngram)
- `baselines/longmemeval_kpi_medium50_fastembed.md` — 50q session (`fastembed` / BGE-small)
- `baselines/longmemeval_kpi_medium50_turn.md` — 50q **turn** (ngram; Product M11)
- `baselines/longmemeval_kpi_full_ngram.md` — **500q full** LongMemEval-S (ngram)
- `baselines/longmemeval_kpi_full_fastembed.md` — **500q full** with optional `fastembed`

Turn mid-slice recipe:

```bash
mpe bench longmemeval \
  --dataset benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json \
  --label medium50 --granularity turn --limit 50
# writes longmemeval_baseline_medium50-turn.*; also commit kpi_*_turn for public tables
```

Labels auto-suffix `-turn` and `-<embedding>` (when not ngram) so dense runs do not overwrite ngram artifacts.

Primary product mode for public recall is **`hybrid`** (BM25-aware lexical + pluggable embeddings + score-ordered ranking). Default embedding is dependency-free **`ngram`**; optional dense backends:

```bash
pip install 'memory-path-engine[embed]'
mpe bench longmemeval --label tiny --embedding fastembed
# or: MPE_EMBEDDING=fastembed mpe bench longmemeval --limit 50 ...
```

Full headline (M7): ngram hybrid matches lexical at **R@5=0.950 / R@10=0.974 / NDCG@10=0.863** on 500q. Layer B path/contradiction fixtures remain the architecture proof surface.

## Product KPI baseline

Treat LongMemEval recall as a **product KPI**:

```bash
# Tiny fixture (CI / local smoke)
mpe bench longmemeval --label tiny --granularity session
mpe bench longmemeval --label tiny --granularity turn

# Full cleaned LongMemEval-S (after download)
python scripts/download_longmemeval.py
mpe bench longmemeval \
  --dataset benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json \
  --label full \
  --granularity session \
  --limit 0
mpe bench longmemeval \
  --dataset benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json \
  --label full \
  --granularity turn \
  --limit 0
```

Artifacts land in `benchmarks/external/longmemeval/baselines/` (`*.json` + `*.md`).
