"""
main.py — Ask a question against your Cortex vault.

Retrieves the most relevant chunks from ChromaDB via query.py,
sends them as context to an LLM via OpenRouter, and prints the answer.

Usage:
    python src/main.py "How does a vendor get approved on Capsul?"
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from query import search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configurations and constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL")
SYSTEM_PROMPT = """You are a personal assistant with access to the user's private knowledge vault.
Answer ONLY from the context provided. Be concise and direct.
If the answer is not in the context, say clearly: "I don't have notes on that."
Do not speculate or use outside knowledge."""


def format_context(hits: list[dict]) -> str:
    """
    Format search hits into a context string for the LLM.

    Args:
        hits (list[dict]): A list of search hits, each containing metadata and content.
    """

    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] {hit['source']} — {hit['heading']}\n{hit['content']}")
    return "\n\n".join(parts)


def format_context(hits: list[dict]) -> str:
    """
    Format search hits into a context string for the LLM.

    Args:
        hits (list[dict]): A list of search hits, each containing metadata and content.

    Returns:
        str: A formatted string combining all hits for LLM context.
    """
    
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] {hit['source']} — {hit['heading']}\n{hit['content']}")
    return "\n\n".join(parts)


def ask(question: str, top_k: int):
    """
    Ask a question against the user's private knowledge vault.

    Args:
        question (str): The natural language question to ask.
        top_k (int): The number of top search hits to retrieve.
    """

    hits = search(question, k=top_k)

    if not hits:
        print("No relevant notes found. Have you run ingest.py?")
        return

    context = format_context(hits)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context from my vault:\n\n{context}\n\n---\n\nQuestion: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    print(f'\n💬 "{question}"\n')
    print("─" * 60)
    answer = chain.invoke({"context": context, "question": question})
    print(answer)
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query your Cortex vault")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    args = parser.parse_args()

    ask(args.question, top_k=args.top_k)