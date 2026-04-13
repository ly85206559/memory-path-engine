# HotpotQA (external benchmark)

This directory holds documentation and optional tiny fixtures for the **HotpotQA adapter** in `memory_engine.benchmarking.adapters.hotpotqa`.

## What is implemented

- Per-question `MemoryStore` construction from official HotpotQA JSON fields (`context`, `supporting_facts`)
- Sentence-level nodes: `{paragraph_stem}:{sentence_index}`
- Within-paragraph edges: `next_unit` (bidirectional) between adjacent sentences
- Evaluation via existing `StructuredBenchmarkRunner` (**retrieval-only**): `evidence_hit` / `evidence_hit_rate` (not official EM/F1)

## Data

Download the official release from [HotpotQA](https://hotpotqa.github.io/). For the first integration milestone, use **dev distractor**:

- `hotpot_dev_distractor_v1.json`

Copy the **license and citation text** from the official release into any public report; do not paraphrase legal wording.

## Run (local)

Run the checked-in tiny fixture:

```bash
python scripts/run_hotpotqa_benchmark.py
```

Run a downloaded official file:

```bash
python scripts/run_hotpotqa_benchmark.py --dataset "F:/data/hotpot_dev_distractor_v1.json" --limit 64 --top-k 10 --modes lexical_baseline,embedding_baseline
```

Pretty-print the full suite JSON:

```bash
python scripts/run_hotpotqa_benchmark.py --pretty
```

## CI note

The main CI job does **not** download HotpotQA. Use the checked-in `hotpot_tiny_fixture.json` (two synthetic items) or embed samples in unit tests.

## Important limitation

Do **not** merge all dev questions into one `MemoryStore`. Each HotpotQA distractor question must keep an isolated graph. Use `run_hotpotqa_benchmark`, which builds a fresh store per sample.
