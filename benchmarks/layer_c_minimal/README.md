# Layer C Minimal Dataset

This directory is the runnable entry point for **Layer C** transfer benchmarks.

Purpose:

- move beyond purely synthetic Layer B fixtures
- keep noise-realistic public stand-in documents that exercise the same structured runner contract
- provide inventory / annotation templates for private golden sets (see `docs/private-contract-dataset-guide.md`)

## What is runnable now

| File | Role |
| --- | --- |
| `layer_c_contract_benchmark.json` | Structured cases over noisy services-agreement stand-in |
| `layer_c_runbook_benchmark.json` | Structured cases over noisy incident-playbook stand-in |
| `documents/contracts/` | Contract-like markdown pack inputs |
| `documents/runbooks/` | Runbook-like markdown pack inputs |
| `templates/inventory_example.csv` | Pilot document inventory columns |
| `templates/annotation_sheet_example.csv` | Golden annotation sheet columns |
| `layer_c_minimal_seed.json` | Planning seed for future private case expansion |

These fixtures use the **same** `StructuredBenchmarkDataset` / runner contract as Layer B (`evidence_node_ids`, `required_edge_types`, `required_contradiction_pairs`, semantic roles, etc.).

## Run

```bash
python scripts/run_layer_c_benchmark.py \
  --output "benchmarks/layer_c_minimal/layer_c_report.json" \
  --markdown-output "benchmarks/layer_c_minimal/layer_c_report.md"
```

## Target shape for private expansion

Grow from the current public stand-ins toward **20-50** high-quality private cases, then the larger private-contract pilot in `docs/private-contract-dataset-guide.md` (~30 docs / 120-180 cases).

Each private case should keep:

- stable `case_id`
- query + gold evidence node ids
- scenario / case-family tags
- optional contradiction pairs, semantic roles, and path shape expectations

## Notes

- Keep source material and benchmark annotation separate for private data.
- Prefer fewer high-confidence cases over a large noisy set.
- Do not commit private contract text to the public repository; replace `documents/` locally or via a private mirror.
