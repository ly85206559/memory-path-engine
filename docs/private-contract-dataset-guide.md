# Private Contract Dataset Guide

This document turns the private-dataset strategy into a contract-focused execution plan.

Use it when you want to build the first internal `golden set` for `memory-path-engine` from real contract text.

## Goal

The first private contract dataset should answer one question:

> On real contract documents, does graph-aware retrieval surface the right evidence, path shape, and exception logic better than flat retrieval?

This is not a document-archive project. It is an evaluation asset.

The first version should therefore optimize for:

- stable source documents
- high-quality evidence labels
- enough diversity to expose graph and exception behavior
- low enough scope that a small team can finish it in `2-4` weeks

## Pilot scope

Recommended first pilot:

- `30` contracts
- `120-180` golden cases
- `4-6` cases per contract
- one primary language per pilot if possible

Recommended first report:

- `lexical_baseline`
- `embedding_baseline`
- `structure_only`
- `weighted_graph`

Add `activation_spreading_v1` only after the golden set is stable.

## Deliverable 1: Contract inventory template

Before annotation starts, build a document inventory.

Suggested columns:


| Column                   | Required | Purpose                                                      |
| ------------------------ | -------- | ------------------------------------------------------------ |
| `doc_id`                 | yes      | Stable internal document identifier                          |
| `file_name`              | yes      | Source filename                                              |
| `file_path`              | yes      | Original storage path                                        |
| `contract_type`          | yes      | MSA, NDA, procurement, software license, DPA, services, etc. |
| `template_family`        | yes      | Standard template family or `non_standard`                   |
| `template_version`       | no       | Version if known                                             |
| `language`               | yes      | `zh`, `en`, `bilingual`                                      |
| `business_line`          | no       | Business grouping for later stratification                   |
| `counterparty_type`      | no       | Supplier, customer, partner, employee, etc.                  |
| `signed_date`            | no       | Useful for version drift and legacy wording                  |
| `page_count`             | no       | Complexity proxy                                             |
| `has_appendix`           | yes      | `yes` / `no`                                                 |
| `has_sow_or_order_form`  | yes      | `yes` / `no`                                                 |
| `risk_level`             | yes      | `low`, `medium`, `high`                                      |
| `exception_density`      | yes      | `low`, `medium`, `high` by quick reviewer estimate           |
| `termination_complexity` | yes      | `low`, `medium`, `high`                                      |
| `selected_for_pilot`     | yes      | `yes` / `no`                                                 |
| `selection_reason`       | no       | Why this contract entered the pilot                          |


Minimum rule:

- do not annotate anything before this inventory exists

## Deliverable 2: First-30 sampling rules

Do not choose the first `30` contracts randomly.

Use stratified sampling with the following target mix:

### Contract type mix

- `8-10` MSA / services agreements
- `5-6` procurement or supply agreements
- `4-5` software license / SaaS agreements
- `3-4` DPA / privacy / security agreements
- `3-4` NDA / confidentiality or framework agreements
- `3-4` non-standard or heavily negotiated contracts

### Language mix

If your corpus is mostly Chinese:

- `20-24` Chinese contracts
- `4-6` English contracts
- `2-4` bilingual contracts

If bilingual quality is noisy, move bilingual contracts into phase 2 instead of phase 1.

### Structural complexity mix

- at least `10` contracts with clear termination sections
- at least `10` contracts with visible exception wording
- at least `8` contracts with appendices, schedules, or order-form dependencies
- at least `6` contracts with negotiated deviations from a standard template

### Exclusion rules for pilot v1

Exclude:

- scanned image PDFs without reliable OCR
- contracts with severe redaction that breaks clause logic
- duplicates of the same template unless version comparison is the explicit goal
- contracts whose clause numbering is too broken to assign stable node ids quickly

## Deliverable 3: Golden annotation sheet

Do not force annotators to write benchmark JSON directly.

Use a tabular annotation sheet first, then export to JSON.

Suggested columns:


| Column                         | Required | Purpose                                                              |
| ------------------------------ | -------- | -------------------------------------------------------------------- |
| `case_id`                      | yes      | Stable case id                                                       |
| `doc_id`                       | yes      | Join back to the inventory                                           |
| `query`                        | yes      | Business-style question                                              |
| `answer_short`                 | yes      | Short canonical answer for human review                              |
| `case_family`                  | yes      | One of the taxonomy families below                                   |
| `difficulty`                   | yes      | `easy`, `medium`, `hard`                                             |
| `tags`                         | yes      | Comma-separated tags such as `multi_hop`, `exception`, `termination` |
| `evidence_node_ids`            | yes      | Gold evidence nodes                                                  |
| `minimum_evidence_matches`     | yes      | Usually `1` or full evidence count                                   |
| `path_scope`                   | no       | `best_path` or `any_path`                                            |
| `path_steps`                   | no       | Ordered node ids plus optional edge types                            |
| `required_edge_types`          | no       | Needed edge types                                                    |
| `required_semantic_roles`      | no       | Roles such as `exception`, `remedy`, `condition`                     |
| `required_contradiction_pairs` | no       | Gold contradiction or override pairs                                 |
| `annotator_a`                  | yes      | Initial annotator                                                    |
| `annotator_b`                  | yes      | Reviewer annotator                                                   |
| `review_status`                | yes      | `draft`, `reviewed`, `approved`, `rejected`                          |
| `review_notes`                 | no       | Why a label changed                                                  |


## Deliverable 4: Question taxonomy v1

The first pilot should cover the following question families.

### A. Direct evidence lookup

Purpose:

- baseline sanity checks
- confirm the parser and node ids are stable

Examples:

- "What is the invoice payment term?"
- "How many days does the customer have to cure a breach?"

Target share:

- `20-25%`

### B. Multi-hop dependency

Purpose:

- validate graph-aware retrieval
- connect precondition, trigger, and consequence clauses

Examples:

- "If the customer materially breaches and fails to cure, what may the supplier do next?"
- "When termination happens, what post-termination obligations remain?"

Target share:

- `25-30%`

### C. Exception / override

Purpose:

- validate `exception_to`, anomaly-like wording, and semantic roles

Examples:

- "Does the standard 30-day payment rule still apply if the deliverables are defective?"
- "When does a limitation-of-liability cap not apply?"

Target share:

- `20-25%`

### D. Contradiction / tension

Purpose:

- detect rule tension between clauses, schedules, or amendments

Examples:

- "The main agreement says thirty days, but the order form says fifteen days. Which governs?"
- "One section allows subcontracting; another forbids disclosure to third parties. What constraint actually applies?"

Target share:

- `10-15%`

### E. Sequential / condition chain

Purpose:

- useful for procedural contract logic such as notice -> cure -> termination -> return/delete/certify

Examples:

- "After notice and failed cure, what is the next contractual consequence?"
- "What happens after termination notice is delivered and the cure period expires?"

Target share:

- `10-15%`

### F. Hard negatives

Purpose:

- prevent misleading wins from clause wording overlap

Examples:

- two payment clauses in different contexts
- one termination clause for convenience and another for breach

Target share:

- `10%` embedded across the above families

## Stable node-id policy

Your gold labels must target stable unit ids, not free text.

Recommended rule:

- `node_id = "{document_stem}:{unit_number}"`

Where `unit_number` comes from the parser's final clause or sentence segmentation layer.

Required discipline:

- once a pilot dataset is frozen, do not silently renumber node ids
- if parsing changes, create a new dataset version

## Annotation workflow

### Stage 1: Preprocessing

1. Convert contracts into stable text.
2. Split into structural units.
3. Assign candidate `node_id`s.
4. Run light rule tagging for likely:
  - `exception`
  - `termination`
  - `notice`
  - `cure`
  - `liability`
  - `payment`

### Stage 2: Draft annotation

Annotator A fills:

- `query`
- `answer_short`
- `evidence_node_ids`
- optional `path`
- optional semantic and contradiction fields

### Stage 3: Review annotation

Annotator B independently checks:

- whether evidence is minimal and sufficient
- whether the path is over-constrained
- whether the case should really be tagged `exception` or `contradiction`

### Stage 4: Adjudication

A reviewer resolves disagreements and writes a short note when changing:

- evidence set
- path shape
- contradiction pair
- case family

### Stage 5: Export

Only `approved` rows are exported to benchmark JSON.

## Quality gates for the pilot

The pilot should not be accepted unless:

- at least `120` approved cases exist
- evidence-label agreement between A and B is `>= 0.85`
- every case has valid `node_id`s that resolve in the frozen document snapshot
- at least `30%` of cases are non-trivial (`multi_hop`, `exception`, or `contradiction`)
- at least `10` hard-negative pairs exist across the dataset

## Recommended tooling split

Use simple tools first:

- spreadsheet or Airtable for inventory
- spreadsheet or internal labeling sheet for golden annotations
- export script to benchmark JSON

Do not start with a complex custom annotation app unless your team already has one.

## Example pilot case

```json
{
  "case_id": "pilot-msa-termination-001",
  "query": "If the customer materially breaches the agreement and does not cure after notice, what may happen next and what obligation remains after termination?",
  "tags": ["contract", "termination", "multi_hop"],
  "expectation": {
    "evidence_node_ids": ["msa_2024_v3:21", "msa_2024_v3:22"],
    "minimum_evidence_matches": 1,
    "path_scope": "best_path",
    "path": {
      "match_mode": "prefix",
      "steps": [
        { "node_id": "msa_2024_v3:22", "via_edge_type": null },
        { "node_id": "msa_2024_v3:21", "via_edge_type": "depends_on" }
      ]
    }
  }
}
```

## Suggested execution plan

### Week 1

- build inventory
- choose first 30 contracts
- freeze source snapshots

### Week 2

- finish parsing and stable node ids
- annotate first 40-60 cases

### Week 3

- review and adjudicate first half
- annotate second half

### Week 4

- export approved cases
- run the first private benchmark comparison
- write the first error analysis memo

## What to do next after pilot v1

After the first pilot is stable, expand in this order:

1. more contracts from the same families
2. bilingual contracts
3. amendment chains and appendices
4. dynamic-memory sequential cases
5. comparison across template versions

