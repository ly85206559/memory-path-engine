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

## Important limitation

This adapter is currently **session-only** and **retrieval-only**:

- it evaluates whether gold `answer_session_ids` appear in the retrieved top-k session list
- it does **not** run answer generation or official QA grading
- it is intended as an external positioning benchmark, not as proof of path, semantic, contradiction, or dynamic-memory claims
