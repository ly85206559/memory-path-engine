# Publish to PyPI (Product M8)

This repository is packaged as **`memory-path-engine`** on PyPI.
Default install stays dependency-light (`pydantic` only). Dense backends remain
optional extras.

## Install (after first release)

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

Until the first PyPI release lands, continue using the git URL documented in
[`install.md`](install.md).

## Automated release (recommended)

One-time (human, ~2 minutes) — register Trusted Publisher on PyPI:

```bash
bash scripts/print_pypi_trusted_publisher.sh
```

On https://pypi.org/manage/account/publishing/ add a **pending publisher** with
those fields. Leave **Environment name blank** (workflow no longer requires a
GitHub Environment).

Then from a clean `master`:

```bash
bash scripts/release.sh
```

This script:

1. Reads version from `pyproject.toml`
2. Runs `scripts/check_package_build.py` (sdist/wheel + `twine check`)
3. Creates annotated tag `vX.Y.Z` and pushes it
4. Watches the `publish` GitHub Actions workflow when `gh` is available

## Manual checklist (equivalent)

1. Version bump in `pyproject.toml` + `CHANGELOG.md`
2. `python scripts/check_package_build.py`
3. Merge to `master` with CI green
4. `git tag v0.8.0 && git push origin v0.8.0`
5. Verify: `pip index versions memory-path-engine`

## Manual / dry-run workflow

`workflow_dispatch` on **publish**:

- `dry_run=true` (default): build + twine check only
- `dry_run=false`: build + upload via Trusted Publisher / OIDC

## Non-goals

- Publishing private forks with a different package name
- Bundling `fastembed` / `sentence-transformers` into the default wheel
- Storing long-lived PyPI API tokens in the repository
