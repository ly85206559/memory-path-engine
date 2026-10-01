# Getting started (5 minutes)

Product closed loop for Memory Path Engine: local palace → ingest → search with
path hops → optional MCP / hooks for agents.

## 1. Install

```bash
python -m pip install --no-build-isolation -e .
mpe --help
```

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

Exposed tools: `mpe_status`, `mpe_init`, `mpe_ingest`, `mpe_ingest_memo`,
`mpe_search`, `mpe_get_path`, `mpe_reinforce`.

## 6. Optional session hooks

Scripts are installed under `.cursor/mpe-hooks/`:

- `mpe_session_start.sh` — status + recall hint
- `mpe_stop_save.sh` — save stdin / file as a memo

Point your IDE stop/start hooks at those scripts (see `hooks.example.json`).

## 7. Public KPI baseline

```bash
mpe bench longmemeval --label tiny
```

Artifacts: `benchmarks/external/longmemeval/baselines/`.

For acceptance checks, see [`ACCEPTANCE.md`](ACCEPTANCE.md). Roadmap: [`ROADMAP.md`](ROADMAP.md).
