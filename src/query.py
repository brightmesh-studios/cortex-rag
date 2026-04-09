"""
query.py — Semantic search against the Cortex ChromaDB vector store.

Takes a natural language question, embeds it using the same model
used during ingestion, and returns the top-k most similar chunks
with their metadata and similarity scores.

This module does not call an LLM — it only handles retrieval.
That is main.py's job.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

load_dotenv()

# Configurations
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_vectorstore() -> Chroma:
    """
    Initialize and return a Chroma vector store with OpenAI embeddings.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    return Chroma(
        collection_name="cortex",
        embedding_function=embeddings,
        client=client,
    )

def search(question: str, k: int = TOP_K) -> list[dict]:
    """
    Perform a semantic search against the ChromaDB vector store.

    Args:
        question (str): The natural language question to search for.
        k (int): The number of top results to return.

    Returns:
        list[dict]: A list of search hits, each containing metadata and content.
    """

    vs = get_vectorstore()
    results = vs.similarity_search_with_relevance_scores(question, k=k)

    hits = []
    for doc, score in results:
        heading_parts = []
        for key in ("h1", "h2", "h3"):
            val = doc.metadata.get(key)
            if val:
                heading_parts.append(val)

        hits.append({
            "source": doc.metadata.get("source", "unknown"),
            "file_name": doc.metadata.get("file_name", "unknown"),
            "heading": " › ".join(heading_parts) if heading_parts else "(no heading)",
            "content": doc.page_content,
            "score": round(score, 4),
        })

    return hits

if __name__ == "__main__":
    hits = search("How did we implement biometrics in the Yuleapp?")
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] score: {hit['score']}")
        print(f"     file: {hit['source']}")
        print(f"  heading: {hit['heading']}")
        print(f"  content: {hit['content'][:200]}")
        print("─" * 60)