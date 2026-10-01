# Getting started (5 minutes)

Product closed loop for Memory Path Engine: local palace → ingest → search with
path hops → optional MCP / hooks for agents.

## 1. Install

Dev clone:

```bash
python -m pip install --no-build-isolation -e .
mpe --help
mpe doctor
```

Or isolated tools (Product M4):

```bash
pipx install git+https://github.com/ly85206559/memory-path-engine.git
# uv tool install git+https://github.com/ly85206559/memory-path-engine.git
```

Full install matrix + Docker MCP: [`install.md`](install.md).

## 2. Create a palace and ingest docs

```bash
mpe init --mode hybrid
mpe ingest examples/runbook_pack/runbooks --pack example_runbook_pack
mpe status
```

Palace files live in `./.mpe/` (override with `$MPE_PALACE`).

## 3. Search with a replayable path

```bash
mpe search "What if rollback does not recover the API?" --mode hybrid
mpe path "What if rollback does not recover the API?"
```

You should see an **ANSWER** block plus **PATH** hops (`node_id`, `via`, `score`).

Optional denser embeddings (after `pip install 'memory-path-engine[embed]'`):

```bash
export MPE_EMBEDDING=fastembed
mpe search "What if rollback does not recover the API?" --mode hybrid
# KPI: mpe bench longmemeval --label tiny --embedding fastembed
```

## 4. Save a session memo

```bash
echo "Decided to restart workers before paging DB owner." | mpe memo --title "ops-decision"
```

## 5. Wire MCP (Cursor / Claude)

```bash
mpe hooks install
```

Then merge `.cursor/mpe-hooks/mcp.local.json` into your MCP client config, or run:

```bash
mpe mcp
```

Docker alternative (after `docker build -t mpe-mcp .`):

```bash
docker run -i --rm -v "$PWD/.mpe:/data/palace" -e MPE_PALACE=/data/palace mpe-mcp
```

Exposed tools: `mpe_status`, `mpe_init`, `mpe_ingest`, `mpe_ingest_memo`,
`mpe_search`, `mpe_get_path`, `mpe_reinforce`.

## 6. Backup / repair

```bash
mpe backup
mpe repair
mpe doctor
```

## 7. Optional session hooks

Scripts are installed under `.cursor/mpe-hooks/`:

- `mpe_session_start.sh` — status + recall hint
- `mpe_stop_save.sh` — save stdin / file as a memo

Point your IDE stop/start hooks at those scripts (see `hooks.example.json`).

## 8. Public KPI baseline

```bash
mpe bench longmemeval --label tiny
```

Artifacts: `benchmarks/external/longmemeval/baselines/`.

For acceptance checks, see [`ACCEPTANCE.md`](ACCEPTANCE.md). Roadmap: [`ROADMAP.md`](ROADMAP.md).
Install matrix: [`install.md`](install.md).
