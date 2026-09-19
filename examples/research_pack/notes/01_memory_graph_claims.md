# Memory Graph Research Notes

## Core Claims
1 Structured retrieval should beat flat top-k on multi-hop questions that cross typed edges.
2 Evidence paths improve explainability when each hop maps to a typed edge and a replayable score.

## Counterpoints
3 Flat embedding retrieval conflicts with claim 1 when questions require exception traversal across linked notes.
4 Unless path replay is causally used in scoring, explainability gains from claim 2 remain cosmetic.

## Follow-ups
5 After validating claim 1 on Layer B fixtures, extend evaluation to private Layer C documents.
6 Claim 2 depends on path replay remaining inspectable in benchmark miss analysis.
