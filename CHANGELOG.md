# Changelog

All notable product releases are listed here.

## 0.9.0 — Product M10 (HotpotQA mid-slice KPI)

- HotpotQA medium64 public KPI tables (ngram + fastembed)
- `scripts/download_hotpotqa.py` HuggingFace Hub fallback when CMU URL is unreachable
- `scripts/write_hotpotqa_kpi.py` for committed baseline artifacts
- Optional dense embedding disk cache via `MPE_EMBEDDING_CACHE_DIR`
- Nightly HotpotQA workflow includes `hybrid` and HF fallback

## 0.8.1 — Product M9 (release polish)

- Docs/ROADMAP mark PyPI `0.8.0` as published
- `mpe doctor` install tips prefer PyPI / pipx / uv
- GitHub Release notes for `v0.8.0`
- PyPI badge on README

## 0.8.0 — Product M8 (PyPI-ready)

- Package metadata enriched for PyPI (classifiers, keywords, URLs, `pydantic>=2`)
- Trusted Publisher publish workflow (`.github/workflows/publish.yml`)
- CI package build + `twine check` job
- `scripts/release.sh` automates tag + publish watch
- Publish guide: [`docs/publish.md`](docs/publish.md)
- **Published to PyPI:** https://pypi.org/project/memory-path-engine/0.8.0/

## 0.7.0 — Product M7 (hybrid NDCG)

- Hybrid seed-score rerank for public ranking (NDCG no longer collapses after graph expansion)
- Full LongMemEval-S KPI refresh (ngram hybrid matches lexical; fastembed hybrid beats lexical)

## 0.6.0 — Product M6 (batch embeddings + full KPI)

- `embed_many` + retriever prefetch
- Dense long-text truncation / adaptive batch sizing
- Full LongMemEval-S (500q) public KPI tables

## 0.5.1 — Product M5 (pluggable embeddings)

- Embedding backends: `ngram` / `hash` / `fastembed` / `sentence`
- Optional extras: `[embed]`, `[embed-st]`

## 0.5.0 — Product M4 (distribution)

- pipx / uv docs, Docker MCP, `mpe backup` / `repair` / `doctor`
