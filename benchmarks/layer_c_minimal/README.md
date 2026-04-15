# Layer C Minimal Dataset

This directory is the starting point for the **Layer C real-data set** mentioned in the roadmap.

Purpose:

- move beyond synthetic structured fixtures
- capture real long-range recall and consolidation behavior
- keep the first dataset intentionally small and hand-auditable

## Target shape

Start with **20-50 high-quality cases** across a small number of scenarios.

Each case should include:

- a stable `case_id`
- the user query
- the expected semantic memory or episodic memory ids
- the scenario tags involved
- whether the case is testing:
  - cross-session recall
  - generalized-rule recall
  - cross-episode abstraction
  - lifecycle-sensitive ranking

## Files

- `layer_c_minimal_seed.json`: starter scaffold with a few sample cases and field shape

## Notes

- Keep source material and benchmark annotation separate.
- Prefer fewer but higher-confidence cases over a large noisy set.
- Treat this directory as the first staging area before a larger private dataset is introduced.
