# cortex-rag

Cortex RAG is a personal ingestion system that chunks my knowledge vault and makes it queryable from any MCP-compatible coding environment.

The main reason I built this is to give myself and any agent in a session fast access to context about specific modules, features, and decisions I've already made. Instead of re-explaining the same architecture every session, the agent can query my vault and get up to speed immediately. That accelerates development and removes the friction of lost context.

---

## How It Works

```
Obsidian vault (.md notes)
    ↓ ingest.py — split by heading, embed via OpenRouter, store in ChromaDB
ChromaDB (local vector store)
    ↓ server.py — MCP server on localhost:8000
Windsurf Cascade / any MCP client
    ↓ search_vault tool — retrieves top-k relevant chunks
Agent has context, development continues
```

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenRouter API key
- Windsurf (or any MCP-compatible editor)

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set VAULT_PATH to your Obsidian vault directory
# Set OPENROUTER_API_KEY to your OpenRouter key
```

### Ingest your vault

```bash
python src/ingest.py
```

### Start the MCP server

```bash
python src/server.py
```

### Connect to Windsurf

Add to ~/.codeium/windsurf/mcp_config.json:

```json
{
  "mcpServers": {
    "cortex": {
      "serverUrl": "http://localhost:8000/mcp"
    }
  }
}
```

Restart Windsurf. The search_vault tool will be available in Cascade automatically.

---

## Usage

### Query from terminal

```bash
python src/query.py
```

### Incremental re-index

```bash
python src/ingest.py --changed-only
```

### Setup vault-sync script

The vault-sync script automatically formats your vault, commits changes, and re-indexes only changed files.

1. Copy the script to your scripts directory:

```bash
cp script/vault-sync.sh ~/scripts/vault-sync.sh
chmod +x ~/scripts/vault-sync.sh
```

2. Edit `~/scripts/vault-sync.sh` and update both paths:

```bash
VAULT=/Users/YOUR_USERNAME/path/to/your/obsidian/vault
RAG=/Users/YOUR_USERNAME/path/to/cortex-rag
```

3. Test the script:

```bash
~/scripts/vault-sync.sh
```

### Auto-sync via cron

Add to your crontab (`crontab -e`):

```
0 */3 * * * /bin/bash ~/scripts/vault-sync.sh >> ~/vault-sync.log 2>&1
```

---

## Project Structure

```
cortex-rag/
├── src/
│   ├── ingest.py    — vault reader, chunker, ChromaDB writer
│   ├── query.py     — semantic search
│   └── server.py    — MCP server
├── script/
│   └── vault-sync.sh  — git sync + incremental re-index script
├── .env.example
├── requirements.txt
└── README.md
```

---

## Stack

- **LangChain** — chunking and retrieval pipeline
- **ChromaDB** — local vector store (never committed, always rebuilt)
- **OpenRouter** — embeddings via openai/text-embedding-3-small
- **MCP (FastMCP)** — exposes search as a tool to any compatible client

---

## Notes

- chroma_db/ is never committed — it is derived from your vault and always rebuilt via ingest.py
- main.py is a scratchpad for LLM chain experiments and is not part of the core pipeline
- The .env file is never committed — copy .env.example and fill in your values
