"""
main.py: ask a question against the Cortex index.

Retrieves the most relevant chunks, sends them to a model on OpenRouter, and
prints the answer. For retrieval alone, use query.py.

Usage:
    python src/main.py "How does a vendor get approved on Capsul?"
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from query import search

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "minimax/minimax-m2.7")
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = """You are a personal assistant with access to the user's private knowledge vault.
Answer ONLY from the context provided. Be concise and direct.
Cite sources as [n]. If the answer is not in the context, say clearly: "I don't have notes on that."
Do not speculate or use outside knowledge."""


def format_context(hits: list[dict]) -> str:
    """Render hits as numbered, citable blocks."""
    return "\n\n".join(
        f"[{i}] {hit['source']}:{hit['start_line']}  {hit['heading']}\n{hit['content']}"
        for i, hit in enumerate(hits, 1)
    )


async def ask(question: str, top_k: int, project: str | None, source_type: str | None) -> None:
    """Answer a question from the index, printing the model's response."""
    hits = await search(question, k=top_k, project=project, source_type=source_type)

    if not hits:
        print("No matches. Has flow.py been run?")
        return

    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n\n{format_context(hits)}\n\n---\n\nQuestion: {question}",
            },
        ],
    )

    print(f'\n"{question}"\n')
    print("-" * 60)
    print(response.choices[0].message.content)
    print("-" * 60)
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit['source']}:{hit['start_line']}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the Cortex index")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--project", default=None, help='e.g. "pms"')
    parser.add_argument("--source-type", default=None, choices=["note", "code"])
    args = parser.parse_args()

    asyncio.run(ask(args.question, args.top_k, args.project, args.source_type))


if __name__ == "__main__":
    _main()
