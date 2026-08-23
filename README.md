# cortex-rag

A personal retrieval system that indexes an Obsidian vault alongside the code repositories it describes, and serves the result to any MCP-compatible editor.

The point is context. Instead of re-explaining the same architecture at the start of every session, an agent queries the index and reads the decision, the spec and the implementation together.

---

## How it works

```
Obsidian vault + source repositories
        |
        |  flow.py: CocoIndex walks, splits, embeds
        v
Postgres + pgvector  (cortex_chunks, cortex_links)
        |
        |  server.py: MCP server, streamable HTTP
        v
Claude Code / Windsurf / any MCP client
```

Retrieval fuses three signals: vector similarity, Postgres full-text rank, and a hub prior derived from the vault's own wikilink graph. Results can be scoped to one project or to notes-only / code-only, and chunks under archived paths are demoted rather than dropped.

Ingestion is incremental. CocoIndex tracks what changed, so a re-run only re-embeds files that actually moved.

---

## Quick start

Requires Docker and an [OpenRouter](https://openrouter.ai) key.

```bash
cp .env.example .env
```

Set `OPENROUTER_API_KEY` and point `CORTEX_ROOT` at the directory holding your vault and repositories. Then:

```bash
docker compose up -d postgres mcp
docker compose run --rm ingest
```

The first ingest embeds everything and costs a few cents on `text-embedding-3-small`. Subsequent runs are near-free.

Check it works:

```bash
docker compose run --rm ingest python src/query.py "how does billing work"
```

---

## Configuring what gets indexed

`sources.yaml` lists one entry per walked directory. Paths resolve `${CORTEX_ROOT}` at load time, so the same file works on a laptop and in a container.

```yaml
- name: my-api
  path: ${CORTEX_ROOT}/my-api
  project: myproject
  source_type: code
  repo: my-api
  include:
    - "src/**/*.ts"
    - "**/*.md"
```

| Key | Purpose |
| --- | --- |
| `project` | Scope tag. Lets one search span a repo and its notes |
| `source_type` | `note` or `code` |
| `project_map` | Per-path project override, for a vault holding several projects |
| `archived_prefixes` | Paths whose content is superseded. Indexed, demoted |
| `archived` | The whole source is superseded |

Two denylists in `src/sources.py` are appended to every source and always win over `include`: `SECRET_DENYLIST` blocks `.env`, keys and certificates; `DERIVED_DENYLIST` blocks build output for code sources.

For a machine whose checkout is laid out differently, point `SOURCES_CONFIG` at a local copy rather than editing the committed file.

---

## Querying

### From an editor

```json
{
  "mcpServers": {
    "cortex": {
      "url": "http://localhost:18432/mcp"
    }
  }
}
```

The `search_vault` tool takes `question`, `top_k`, `project`, `source_type` and `mode`. Use `mode="fulltext"` for an exact identifier or error string, where embeddings blur the match.

### From the terminal

```bash
python src/query.py "payfast tokenisation" --project pms --source-type code
python src/main.py "how does a vendor get approved on Capsul?"
```

`query.py` returns ranked chunks. `main.py` sends them to a model and answers.

---

## Running without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d postgres
```

Set `DATABASE_URL=postgresql://cortex:cortex@localhost:5433/cortex` in `.env`, then:

```bash
python src/flow.py          # one-shot
python src/flow.py --live   # re-index on file change
python src/server.py
```

---

## Deploying to a server

Ports bind to loopback by default. Behind a reverse proxy on a shared Docker network, copy `deploy/compose.override.example.yaml` to `compose.override.yaml` and Compose merges it automatically.

Keep repositories current with a scheduled `git pull` and re-run the ingest afterwards. There is nothing to watch for in between, so the one-shot run is a better fit than `--live`.

---

## Layout

| File | Role |
| --- | --- |
| `src/flow.py` | Ingestion pipeline |
| `src/sources.py` | Source config and denylists |
| `src/embedder.py` | OpenRouter embeddings, batched with retries |
| `src/links.py` | Wikilink graph extraction |
| `src/query.py` | Hybrid retrieval and CLI |
| `src/server.py` | MCP server |
| `src/main.py` | Ask a question, get an answer |
