"""
ingest.py — Cortex vault ingestion pipeline.

Reads all markdown files from the Obsidian vault, splits them into
semantic chunks by heading hierarchy and character limit, embeds them
via OpenRouter, and persists them to a local ChromaDB vector store.

Usage:
    python src/ingest.py                 # full re-index
    python src/ingest.py --changed-only  # only re-index changed files

Change detection is hash-based (MD5), tracked in chroma_db/.ingest_state.json.
Stale chunks are deleted before re-indexing to avoid duplicates.
"""

import os
import json
import hashlib
import argparse
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)
import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ingest")


load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Paths and settings
VAULT_PATH = Path(os.getenv("VAULT_PATH", "/Users/dimez/brightmesh.studio/cortex"))
SKIP_DIRS = {"00-templates"}
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# Chroma DB Path & Embedding Model
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

# State file to track ingested files and their hashes
STATE_FILE = Path(CHROMA_PATH) / ".ingest_state.json"


def collect_vault_files() -> list[Path]:
    """
    Collect all markdown files in the vault, excluding those in skipped directories.

    Returns:
        list[Path]: A sorted list of markdown file paths.
    """
    
    files = []
    for md_file in VAULT_PATH.rglob("*.md"):
        parts = md_file.relative_to(VAULT_PATH).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in parts):
            continue
        files.append(md_file)

    return sorted(files)


def chunk_file(path: Path) -> list:
    """
    Read the content of a markdown file and split it into chunks based on headers and character limits.

    Args:
        path (Path): The path to the markdown file.

    Returns:
        list: A list of document chunks with metadata.
    """
    text = path.read_text(encoding="utf-8")
    
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    header_chunks = md_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = char_splitter.split_documents(header_chunks)

    for chunk in chunks:
        chunk.metadata["source"] = str(path.relative_to(VAULT_PATH))
        chunk.metadata["file_name"] = path.name


    return chunks


def get_vectorstore()-> Chroma:
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


def _is_upstream_rate_limit(exc: BaseException) -> bool:
    """
    Decide whether an embedding failure is a transient upstream rate limit
    that is worth retrying.

    OpenRouter proxies embeddings to OpenAI's real API. When the shared
    upstream pool is throttled, OpenRouter returns an HTTP 200 whose body is
    an ``{"error": {..., "code": 429}}`` object with no ``data`` array. The
    OpenAI SDK sees a 200, never triggers its own 429 backoff, fails to find
    embeddings, and raises ``ValueError("No embedding data received")``.

    We retry on exactly that signature and let every other error fail fast,
    so a genuinely malformed chunk is not retried six times pointlessly.

    Args:
        exc (BaseException): The exception raised while embedding.

    Returns:
        bool: True if the failure looks like a retryable upstream throttle.
    """

    return isinstance(exc, ValueError) and "no embedding data received" in str(exc).lower()


@retry(
    retry=retry_if_exception(_is_upstream_rate_limit),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _embed_with_retry(vs: Chroma, chunks: list) -> None:
    """
    Add chunks to the vector store, retrying transient upstream rate limits
    with exponential backoff (2s, 4s, 8s … capped at 60s, up to 6 attempts).

    Args:
        vs (Chroma): The Chroma vector store instance.
        chunks (list): The document chunks to embed and persist.
    """

    vs.add_documents(chunks)


def ingest_file(vs: Chroma, path: Path) -> int:
    """
    Ingest a markdown file into the Chroma vector store.

    Args:
        vs (Chroma): The Chroma vector store instance.
        path (Path): The path to the markdown file.

    Returns:
        int: The number of chunks ingested from the file.

    Raises:
        RuntimeError: If embedding still fails after all retry attempts,
            with guidance on the likely cause.
    """

    chunks = chunk_file(path)

    if not chunks:
        print(f"No chunks: {path.name}")
        return 0

    try:
        _embed_with_retry(vs, chunks)
    except ValueError as exc:
        if _is_upstream_rate_limit(exc):
            raise RuntimeError(
                f"Embedding '{path.relative_to(VAULT_PATH)}' failed after 6 retries: "
                f"OpenRouter kept returning an empty/throttled response "
                f"(upstream OpenAI rate limit on '{EMBED_MODEL}'). "
                f"This is transient — re-run 'python src/ingest.py --changed-only' shortly, "
                f"or point embeddings directly at api.openai.com to avoid the shared pool."
            ) from exc
        raise

    print(f"{len(chunks)} chunks — {path.relative_to(VAULT_PATH)}")

    return len(chunks)


def delete_file_from_store(vs: Chroma, rel_path: str):
    """
    Delete all chunks from the vector store that match the given relative file path.

    Args:
        vs (Chroma): The Chroma vector store instance.
        rel_path (str): The relative path of the file whose chunks should be deleted.
    """

    collection = vs._collection
    results = collection.get(where={"source": rel_path})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} stale chunks: {rel_path}")


def file_hash(path: Path) -> str:
    """
    Calculate the MD5 hash of a file's contents.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The MD5 hash of the file's contents.
    """

    return hashlib.md5(path.read_bytes()).hexdigest()


def load_state() -> dict:
    """
    Load the ingestion state from the state file.

    Returns:
        dict: The ingestion state.
    """

    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    """
    Save the ingestion state to the state file.

    Args:
        state (dict): The ingestion state to save.
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Cortex vault into ChromaDB")
    parser.add_argument("--changed-only", action="store_true")
    args = parser.parse_args()

    vs = get_vectorstore()
    files = collect_vault_files()
    state = load_state()

    total = 0
    updated = 0

    for f in files:
        rel_path = str(f.relative_to(VAULT_PATH))
        current_hash = file_hash(f)

        if args.changed_only and state.get(rel_path) == current_hash:
            continue

        delete_file_from_store(vs, rel_path)
        total += ingest_file(vs, f)
        updated += 1
        state[rel_path] = current_hash

    save_state(state)
    print(f"\nDone. {updated} files indexed, {total} chunks added.")
    print(f"ChromaDB now has: {vs._collection.count()} chunks")
