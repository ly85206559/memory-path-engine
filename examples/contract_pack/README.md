# Contract Pack

`contract_pack` is the first validation domain for `memory-engine`.

It is intentionally synthetic. The goal is not legal correctness. The goal is to provide a compact dataset with:

- hierarchical clauses
- cross-clause dependencies
- exceptions
- risk-bearing obligations
- multi-hop questions with explicit evidence targets

## Contents

- `contracts/`: sample markdown contracts
- `eval/questions.json`: annotated evaluation questions
- `demo.py`: minimal ingestion and retrieval example

## Evaluation intent

This pack is designed to test whether:

- structure helps connect related clauses
- weights surface risky or exceptional clauses
- replayed paths make reasoning auditable
