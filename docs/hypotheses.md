# Research Hypotheses

This repository starts with explicit, falsifiable hypotheses. If these hypotheses do not hold, the design should change.

## H1: graph-aware retrieval beats vanilla top-k on multi-hop questions

**Claim**

For questions that require traversing at least one relationship edge, weighted graph retrieval should outperform plain lexical or embedding-only top-k retrieval.

**Success criteria**

- at least `+15%` relative improvement in `evidence_recall` on the multi-hop subset
- at least `+10%` relative improvement in `answer_recall` on the same subset

**Failure signal**

- the graph retriever behaves the same as baseline top-k even when the target evidence is split across linked clauses

## H2: anomaly and importance weighting improve critical evidence recall

**Claim**

Adding risk, importance, or anomaly weight should increase the chance that critical clauses appear in the top result set.

**Success criteria**

- weighted retrieval returns at least one critical evidence node in top-3 results more often than the unweighted structure-only variant
- no more than `20%` median latency regression against the structure-only retriever

**Failure signal**

- weighting only reshuffles already obvious hits and does not improve critical evidence recall

## H3: replayable memory paths improve explainability without unacceptable latency

**Claim**

Returning a replayable path of nodes and edges should make retrieval behavior auditable while keeping query latency within a practical research-prototype range.

**Success criteria**

- every evaluation result includes a machine-readable `MemoryPath`
- path steps can be traced back to source evidence
- median query latency stays under `200 ms` for the current synthetic benchmark dataset on local runs

**Failure signal**

- answer quality only comes from a hidden heuristic and the path object is cosmetic rather than causally useful

## Evaluation notes

- Use [`docs/evaluation.md`](evaluation.md) as the baseline comparison contract.
- Track ablations for:
  - no structure
  - no weights
  - no path expansion
