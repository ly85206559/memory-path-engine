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
python scripts/run_longmemeval_benchmark.py --dataset "benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json" --limit 50 --top-k 10 --modes embedding_baseline,weighted_graph
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

This adapter is currently **session-only** and **retrieval-only**:

- it evaluates whether gold `answer_session_ids` appear in the retrieved top-k session list
- it does **not** run answer generation or official QA grading
- Layer B / Layer C still own path, semantic, contradiction, and dynamic-memory claims

## Product KPI baseline (M1)

Treat LongMemEval recall as a **product KPI**, not only a research footnote:

```bash
mpe bench longmemeval --label tiny
# or against a downloaded cleaned file:
mpe bench longmemeval --dataset benchmarks/external/longmemeval/data/longmemeval_s_cleaned.json --label full --limit 0
```

Artifacts land in `benchmarks/external/longmemeval/baselines/` (`*.json` + `*.md`). See [`docs/ROADMAP.md`](../../../docs/ROADMAP.md) for hybrid / turn-level upgrades (M3).
