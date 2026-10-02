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

## One-time PyPI Trusted Publisher setup

Publishing uses **OpenID Connect** (no long-lived API token in GitHub secrets).

1. Create the project on https://pypi.org (name: `memory-path-engine`) if it does
   not exist yet — first publish can also create it via Trusted Publisher.
2. On PyPI → **Publishing** → **Add a new pending publisher**:
   - Owner: `ly85206559`
   - Repository: `memory-path-engine`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub → **Settings → Environments → New environment**: name it `pypi`.
   Optional: require reviewers before deploy.

## Release checklist

1. Version bump in `pyproject.toml` (and `CHANGELOG.md` entry).
2. Local dry-run:

   ```bash
   python -m pip install -e '.[dev]'
   python scripts/check_package_build.py
   ```

3. Merge to `master` with CI green.
4. Tag and push:

   ```bash
   git tag v0.8.0
   git push origin v0.8.0
   ```

5. GitHub Actions workflow **publish** builds the sdist/wheel, runs `twine check`,
   then uploads to PyPI via Trusted Publisher.
6. Verify:

   ```bash
   pip index versions memory-path-engine
   pipx install memory-path-engine==0.8.0
   mpe --help
   ```

## Manual / dry-run workflow

`workflow_dispatch` on **publish** with `dry_run=true` (default) only builds and
checks artifacts; it does **not** upload.

## Non-goals

- Publishing private forks with a different package name
- Bundling `fastembed` / `sentence-transformers` into the default wheel
