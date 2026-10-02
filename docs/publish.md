# Publish to PyPI (Product M8+)

This repository is published as **`memory-path-engine`** on PyPI:

https://pypi.org/project/memory-path-engine/

Default install stays dependency-light (`pydantic` only). Dense backends remain
optional extras.

## Install

```bash
pip install memory-path-engine
# optional dense embeddings:
pip install 'memory-path-engine[embed]'
mpe doctor
```

Isolated CLI:

```bash
pipx install memory-path-engine
# or: uv tool install memory-path-engine
```

Dev clone / Docker: [`install.md`](install.md).

## Automated release (subsequent versions)

Trusted Publisher is already configured for this repo. From a clean `master`:

```bash
# bump version in pyproject.toml + CHANGELOG.md, then:
bash scripts/release.sh
```

This script builds, tags `vX.Y.Z`, pushes the tag, and watches the `publish`
workflow.

Print publisher fields (for debugging / new forks):

```bash
bash scripts/print_pypi_trusted_publisher.sh
```

## Manual / dry-run workflow

`workflow_dispatch` on **publish**:

- `dry_run=true` (default): build + twine check only
- `dry_run=false`: build + upload via Trusted Publisher / OIDC

## Non-goals

- Publishing private forks with a different package name
- Bundling `fastembed` / `sentence-transformers` into the default wheel
- Storing long-lived PyPI API tokens in the repository
