"""
server.py — Cortex MCP server.

Exposes the vault search as an MCP tool that any MCP-compatible
client can call — Windsurf, Claude CLI, Open WebUI, etc.

Runs locally on http://localhost:8000/mcp

Usage:
    python src/server.py
"""

from mcp.server.fastmcp import FastMCP
from query import search

mcp = FastMCP("cortex", host="0.0.0.0", port=8000)

@mcp.tool()
def search_vault(question: str, top_k: int = 5) -> list[dict]:
    """
    Search the Cortex knowledge vault for relevant notes.

    Use this when you need context about a project, feature, decision,
    or any topic the user has written notes on. Returns the most
    semantically similar chunks with their source file and heading.

    Args:
        question: Natural language question to search the vault with.
        top_k: Number of chunks to retrieve. Defaults to 5.
    """
    return search(question, k=top_k)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")