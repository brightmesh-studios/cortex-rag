"""
server.py — Cortex MCP server.

Exposes the hybrid index (vault notes + repository code) as an MCP tool that any
MCP-compatible client can call — Windsurf, Claude Code, Open WebUI, etc.

Usage:
    python src/server.py
"""

import os
from typing import Literal

from mcp.server.fastmcp import FastMCP
from query import search

mcp = FastMCP(
    "cortex",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "18432")),
)


@mcp.tool()
async def search_vault(
    question: str,
    top_k: int = 5,
    project: str | None = None,
    source_type: str | None = None,
    mode: Literal["hybrid", "vector", "fulltext"] = "hybrid",
) -> list[dict]:
    """
    Search the Cortex knowledge index for relevant notes and source code.

    Covers both the Obsidian vault (architecture decisions, specs, findings) and
    indexed repositories, so a question can be answered from the documentation and
    the implementation together. Each hit reports its file, line and source type,
    so results can be cited as path:line.

    Retrieval fuses semantic similarity with full-text matching. Prefer
    mode="fulltext" when searching for an exact identifier, flag name or error
    string, since embeddings blur those.

    Note that well-written prose tends to outrank code for conceptual questions.
    When you specifically want the implementation, pass source_type="code".

    Args:
        question: Natural language question, or an exact identifier.
        top_k: Number of chunks to retrieve. Defaults to 5.
        project: Restrict to one project, e.g. "pms".
        source_type: Restrict to "note" or "code".
        mode: "hybrid" (default), "vector", or "fulltext".
    """
    return await search(
        question,
        k=top_k,
        project=project,
        source_type=source_type,
        mode=mode,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
