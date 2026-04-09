# Contributing

Thanks for your interest in `memory-path-engine`.

This repository is currently a research prototype, so the most helpful contributions are:

- small, reviewable bug fixes
- retrieval or evaluation improvements with clear before/after evidence
- new domain-pack experiments that preserve the core abstractions
- test cases that expose retrieval failures or broken path replay behavior

## Development

```bash
python -m pip install --no-build-isolation -e .
python -m unittest discover -s tests -v
```

## Ground rules

- keep the core abstractions small and explicit
- prefer tests and reproducible examples over architectural speculation
- when changing retrieval logic, document the expected metric impact
- do not add opaque magic constants without explaining them in code or docs

## Pull requests

Include:

- what changed
- why it changed
- what hypothesis or failure case it addresses
- how you validated it
