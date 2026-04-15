# HotpotQA (external benchmark)

This directory holds documentation and optional tiny fixtures for the **HotpotQA adapter** in `memory_engine.benchmarking.adapters.hotpotqa`.

## What is implemented

- Per-question `MemoryStore` construction from official HotpotQA JSON fields (`context`, `supporting_facts`)
- Sentence-level nodes: `{paragraph_stem}:{sentence_index}`
- Within-paragraph edges: `next_unit` (bidirectional) between adjacent sentences
- Evaluation via existing `StructuredBenchmarkRunner` (**retrieval-only**): `evidence_hit` / `evidence_hit_rate` (not official EM/F1)
- Summary breakdowns by `type` (`bridge` / `comparison`) for local or nightly runs

## Data

Download the official release from [HotpotQA](https://hotpotqa.github.io/). For the first integration milestone, use **dev distractor**:

- `hotpot_dev_distractor_v1.json`

Copy the **license and citation text** from the official release into any public report; do not paraphrase legal wording.

You can download the file into the repository's external-benchmark area with:

```bash
python scripts/download_hotpotqa.py
```

## Run (local)

Run the checked-in tiny fixture:

```bash
python scripts/run_hotpotqa_benchmark.py
```

Run a downloaded official file:

```bash
python scripts/run_hotpotqa_benchmark.py --dataset "benchmarks/external/hotpotqa/data/hotpot_dev_distractor_v1.json" --limit 64 --top-k 10 --modes lexical_baseline,embedding_baseline,weighted_graph,activation_spreading_v1
```

Pretty-print the full suite JSON:

```bash
python scripts/run_hotpotqa_benchmark.py --pretty
```

Write the full suite report JSON to a file:

```bash
python scripts/run_hotpotqa_benchmark.py --output "benchmarks/external/hotpotqa/data/local-report.json"
```

Write a compact summary JSON (overall + per-type breakdowns) for dashboards or nightly artifacts:

```bash
python scripts/run_hotpotqa_benchmark.py --summary-output "benchmarks/external/hotpotqa/data/local-summary.json"
```

## CI note

The main CI workflow does **not** download HotpotQA. A dedicated sanity job uses the checked-in `hotpot_tiny_fixture.json` (two synthetic items) so pull requests stay fast and deterministic.

A separate GitHub Actions workflow, `hotpotqa-nightly.yml`, is intended for scheduled or manual runs against the downloaded official `dev distractor` file and uploads both the full suite JSON and a compact summary artifact.

## Important limitation

Do **not** merge all dev questions into one `MemoryStore`. Each HotpotQA distractor question must keep an isolated graph. Use `run_hotpotqa_benchmark`, which builds a fresh store per sample.
