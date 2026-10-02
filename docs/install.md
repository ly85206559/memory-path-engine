# Install (Product M4–M8)

## Quick paths

### From PyPI (Product M8)

```bash
pip install memory-path-engine
# optional dense embeddings:
pip install 'memory-path-engine[embed]'
mpe doctor
```

Isolated tools:

```bash
pipx install memory-path-engine
# uv tool install memory-path-engine
```

First-time PyPI project setup / tagging: [`publish.md`](publish.md).

### From a clone (dev)

```bash
python -m pip install --no-build-isolation -e .
mpe doctor
```

Optional dense embeddings (Product M5 — not required for default `ngram`):

```bash
python -m pip install --no-build-isolation -e '.[embed]'      # fastembed / BGE-small
# python -m pip install --no-build-isolation -e '.[embed-st]'  # sentence-transformers
export MPE_EMBEDDING=fastembed   # or pass --embedding on bench / search paths
```

### From git (before / without PyPI)

```bash
pipx install git+https://github.com/ly85206559/memory-path-engine.git
# uv tool install git+https://github.com/ly85206559/memory-path-engine.git
mpe --help
mpe doctor
```

Entry points after install: `mpe` (CLI) and `mpe-mcp` (stdio MCP).

## Palace ops

```bash
mpe init --mode hybrid
mpe backup                 # writes ./mpe-backups/<palace>-<ts>.tar.gz
mpe repair                 # integrity check; quarantine corrupt SQLite
mpe doctor                 # python / package / palace health
```

## Docker stdio MCP

```bash
docker build -t mpe-mcp .
mkdir -p .mpe
docker run -i --rm \
  -v "$PWD/.mpe:/data/palace" \
  -e MPE_PALACE=/data/palace \
  mpe-mcp
```

Wire into Cursor MCP settings (also emitted by `mpe hooks install` as
`memory-path-engine-docker` in `mcp.example.json`):

```json
{
  "mcpServers": {
    "memory-path-engine": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "${PWD}/.mpe:/data/palace",
        "-e", "MPE_PALACE=/data/palace",
        "mpe-mcp"
      ]
    }
  }
}
```

See [`getting-started.md`](getting-started.md) for the 5-minute product loop.
