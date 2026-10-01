# Memory Path Engine — stdio MCP image (Product M4)
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --no-build-isolation .

ENV MPE_PALACE=/data/palace
VOLUME ["/data/palace"]

# Stdio MCP for agent clients (Cursor / Claude Desktop).
ENTRYPOINT ["mpe-mcp"]
