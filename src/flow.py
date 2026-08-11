"""
flow.py — Cortex ingestion pipeline (CocoIndex → Postgres/pgvector).

Walks every source in sources.yaml, splits files into language-aware chunks,
embeds them via OpenRouter, and upserts them into one `cortex_chunks` table
tagged with project/source_type/repo so search can be scoped.

Replaces ingest.py's hand-rolled MD5 change tracking: CocoIndex owns incremental
sync, so re-running only re-embeds what actually changed.

Usage:
    python src/flow.py            # one-shot update
    python src/flow.py --live     # stay running, re-index on file change
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import asyncpg
import cocoindex as coco
import numpy as np
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from dotenv import load_dotenv
from numpy.typing import NDArray

import links
from embedder import build_embedder
from sources import Source, load_sources

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("flow")

# One line per embedding request would be tens of thousands of lines on a full
# ingest, burying CocoIndex's own progress output.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cortex:cortex@localhost:5433/cortex")
TABLE_NAME = "cortex_chunks"
LINKS_TABLE = "cortex_links"

# CocoIndex's own incremental-sync and memoization state (local LMDB). This is
# what replaces ingest.py's .ingest_state.json; it is derived, never committed.
os.environ.setdefault(
    "COCOINDEX_DB",
    str(Path(__file__).parent.parent / "cocoindex_state"),
)

# chunk_size is in BYTES here (CocoIndex), unlike ingest.py's character-based
# splitter — roughly comparable for mostly-ASCII notes and code.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# chunk_size is a target, not a ceiling: tree-sitter will not break an
# indivisible syntactic unit, so a single huge declaration can exceed it and be
# rejected by the embedding model's token limit. Hard-split anything past this.
MAX_CHUNK_BYTES = 6000

# Prepend structural context to chunks before embedding. Off only for A/B
# measurement — toggling it changes every embedding, so it forces a full re-embed.
CONTEXT_PREFIX = os.getenv("CONTEXT_PREFIX", "1") != "0"

PG_DB = coco.ContextKey[asyncpg.Pool]("cortex_db")

_EMBEDDER = build_embedder()
_SPLITTER = RecursiveSplitter()

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*?)\s*#*$")

# Generated full-text column + GIN index, which CocoIndex's schema inference has
# no concept of. Attached to the table so CocoIndex manages its lifecycle.
# This is what makes hybrid search possible without a second system.
FULLTEXT_SETUP_SQL = f"""
ALTER TABLE {TABLE_NAME}
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX IF NOT EXISTS {TABLE_NAME}__content_tsv
    ON {TABLE_NAME} USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS {TABLE_NAME}__project_source
    ON {TABLE_NAME} (project, source_type);
"""

FULLTEXT_TEARDOWN_SQL = f"""
DROP INDEX IF EXISTS {TABLE_NAME}__content_tsv;
DROP INDEX IF EXISTS {TABLE_NAME}__project_source;
ALTER TABLE {TABLE_NAME} DROP COLUMN IF EXISTS content_tsv;
"""

# Both directions are queried: dst for inbound counts (the hub prior), src for
# one-hop expansion from a winning note.
LINKS_SETUP_SQL = f"""
CREATE INDEX IF NOT EXISTS {LINKS_TABLE}__dst ON {LINKS_TABLE} (dst_path);
CREATE INDEX IF NOT EXISTS {LINKS_TABLE}__src ON {LINKS_TABLE} (src_path);
"""

LINKS_TEARDOWN_SQL = f"""
DROP INDEX IF EXISTS {LINKS_TABLE}__dst;
DROP INDEX IF EXISTS {LINKS_TABLE}__src;
"""


@dataclass
class LinkRow:
    """One resolved `[[wikilink]]` edge between two notes."""

    id: str
    src_path: str
    dst_path: str
    project: str


@dataclass
class ChunkRow:
    """One row of `cortex_chunks`. Field names are the column names."""

    id: str
    content: str
    embedding: Annotated[NDArray[np.float32], _EMBEDDER]
    source_type: str
    project: str
    path: str
    repo: str | None
    language: str | None
    heading: str | None
    start_line: int
    archived: bool
    updated_at: datetime


def _heading_index(text: str) -> list[tuple[int, int, str]]:
    """
    Collect markdown headings as (line_no, level, title), 1-indexed lines.

    Fenced code blocks are skipped so that shell comments like `# build` are not
    mistaken for headings.
    """
    headings: list[tuple[int, int, str]] = []
    in_fence = False

    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = _HEADING_RE.match(line)
        if match:
            headings.append((line_no, len(match.group(1)), match.group(2)))

    return headings


def _heading_at(headings: list[tuple[int, int, str]], line: int) -> str | None:
    """
    Resolve the `h1 > h2 > h3` breadcrumb active at the given line.

    CocoIndex's splitter starts chunks at headings but does not carry the heading
    hierarchy as metadata, so it is reconstructed here to preserve the `heading`
    field query.py has always returned.
    """
    active: dict[int, str] = {}

    for heading_line, level, title in headings:
        if heading_line > line:
            break
        active[level] = title
        # A new heading closes any deeper levels beneath it.
        for deeper in [lvl for lvl in active if lvl > level]:
            del active[deeper]

    parts = [active[lvl] for lvl in sorted(active)]
    return " > ".join(parts) if parts else None


_FRONTMATTER_NAME_RE = re.compile(r"^name:\s*(.+)$", re.M)

# Declarations worth naming as the enclosing scope of a code chunk.
_DECL_RE = re.compile(
    r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?"
    r"(?:async\s+)?(?P<kind>class|interface|enum|type|function|const|model|"
    r"datasource|generator)\s+(?P<name>[A-Za-z_$][\w$]*)"
)


def _note_title(text: str, rel_path: str) -> str:
    """Prefer the frontmatter `name:` field, falling back to the filename."""
    match = _FRONTMATTER_NAME_RE.search(text[:2000])
    return match.group(1).strip() if match else Path(rel_path).stem


def _symbol_index(text: str) -> list[tuple[int, int, str]]:
    """
    Collect code declarations as (line_no, indent, name), 1-indexed lines.

    Indentation stands in for nesting depth, which lets the enclosing scope of a
    chunk be reconstructed the same way heading levels are for markdown.
    """
    symbols = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = _DECL_RE.match(line)
        if match:
            symbols.append((line_no, len(match.group("indent")), match.group("name")))
    return symbols


def _symbol_at(symbols: list[tuple[int, int, str]], line: int) -> str | None:
    """Resolve the enclosing declaration breadcrumb active at the given line."""
    active: dict[int, str] = {}

    for decl_line, indent, name in symbols:
        if decl_line > line:
            break
        active[indent] = name
        for deeper in [lvl for lvl in active if lvl > indent]:
            del active[deeper]

    parts = [active[lvl] for lvl in sorted(active)]
    return " > ".join(parts) if parts else None


def _context_prefix(
    source: Source,
    rel_path: str,
    language: str | None,
    title: str | None,
    breadcrumb: str | None,
) -> str:
    """
    Build the structural context prepended to a chunk *before embedding*.

    A chunk lifted out of its file embeds without any of the context that makes it
    findable — "restart it after the migration" says nothing on its own, and a code
    chunk is worse off still, carrying no path, module or symbol at all. The prefix
    is embedded but never stored as content, so callers still receive the raw text.
    """
    if source.source_type == "note":
        head = f'Note "{title}" ({rel_path})'
        return f"{head} — section: {breadcrumb}" if breadcrumb else head

    where = f"{source.repo}/{rel_path}" if source.repo else rel_path
    head = f"{language or 'code'} file {where}"
    return f"{head} — {breadcrumb}" if breadcrumb else head


def _cap_size(chunks: list) -> list[tuple[str, int]]:
    """
    Flatten chunks to (text, start_line), hard-splitting any that exceed
    MAX_CHUNK_BYTES so no single chunk can be rejected for length.

    Sub-slices inherit the parent's start line: the split is mid-syntax by
    definition, so a more precise line number would be invented rather than real.
    """
    capped: list[tuple[str, int]] = []

    for chunk in chunks:
        text = chunk.text
        if len(text.encode()) <= MAX_CHUNK_BYTES:
            capped.append((text, chunk.start.line))
            continue

        # Slice on characters; MAX_CHUNK_BYTES is a conservative enough budget
        # that multi-byte characters cannot push a slice over the token limit.
        for offset in range(0, len(text), MAX_CHUNK_BYTES):
            piece = text[offset : offset + MAX_CHUNK_BYTES]
            if piece.strip():
                capped.append((piece, chunk.start.line))

    return capped


@coco.fn(memo=True)
async def process_file(
    file: localfs.File,
    source: Source,
    table: postgres.TableTarget[ChunkRow],
) -> None:
    """
    Split one file into chunks, embed each, and declare the resulting rows.

    Memoized on the file: localfs.File keys on mtime, falling back to a content
    fingerprint, so an unchanged file is never read or split.
    """
    rel_path = str(file.file_path.relative_to(source.path))
    language = detect_code_language(filename=file.file_path.name)

    text = await file.read_text()
    if not text.strip():
        return

    chunks = _SPLITTER.split(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        language=language,
    )
    if not chunks:
        return

    is_note = source.source_type == "note"
    headings = _heading_index(text) if is_note else []
    symbols = [] if is_note else _symbol_index(text)
    title = _note_title(text, rel_path) if is_note else None

    capped = _cap_size(chunks)

    def breadcrumb_for(line: int) -> str | None:
        return _heading_at(headings, line) if is_note else _symbol_at(symbols, line)

    # What gets embedded carries its structural context; what gets stored stays raw.
    if CONTEXT_PREFIX:
        embed_texts = [
            f"{_context_prefix(source, rel_path, language, title, breadcrumb_for(line))}\n\n{body}"
            for body, line in capped
        ]
    else:
        embed_texts = [body for body, _ in capped]

    # Embed concurrently, not in a sequential await loop. The embedder batches
    # *concurrent* calls (@coco.fn.as_async(batching=True)); awaiting one at a
    # time would make every call its own batch of one — 590 round-trips for a
    # large note instead of ~10.
    embeddings = await asyncio.gather(*(_EMBEDDER.embed(t) for t in embed_texts))

    project = source.project_for(rel_path)
    archived = source.is_archived(rel_path)

    now = datetime.now(timezone.utc)
    for index, ((content, start_line), embedding) in enumerate(zip(capped, embeddings)):
        # Stable id: same file + same position upserts rather than duplicating,
        # which is the failure ingest.py's delete-before-write existed to avoid.
        chunk_id = hashlib.sha256(
            f"{source.name}:{rel_path}:{index}".encode()
        ).hexdigest()

        table.declare_row(
            row=ChunkRow(
                id=chunk_id,
                content=content,
                embedding=embedding,
                source_type=source.source_type,
                project=project,
                path=rel_path,
                repo=source.repo,
                language=language,
                heading=breadcrumb_for(start_line),
                start_line=start_line,
                archived=archived,
                updated_at=now,
            )
        )


@coco.fn
async def ingest_links(source: Source, table: postgres.TableTarget[LinkRow]) -> None:
    """
    Resolve every wikilink in a note source into edge rows.

    Resolution needs the whole note index up front (a bare `[[auth]]` is resolved
    against all candidates by proximity), so this walks the source directly rather
    than per-file. Link extraction involves no embedding, so rebuilding the full
    edge set whenever any note changes is cheap.
    """
    matcher = source.matcher()
    rel_paths = [
        str(p.relative_to(source.path))
        for p in sorted(source.path.rglob("*.md"))
        if p.is_file() and matcher.is_file_included(p.relative_to(source.path))
    ]

    by_path, by_name = links.build_index(rel_paths)
    edge_count = 0

    for rel in rel_paths:
        text = (source.path / rel).read_text(encoding="utf-8")
        for dst in links.extract_edges(text, rel, by_path, by_name):
            table.declare_row(
                row=LinkRow(
                    id=hashlib.sha256(f"{source.name}:{rel}->{dst}".encode()).hexdigest(),
                    src_path=rel,
                    dst_path=dst,
                    project=source.project_for(rel),
                )
            )
            edge_count += 1

    logger.info("  links: %d edges from %d notes", edge_count, len(rel_paths))


@coco.fn
async def ingest_source(
    source: Source,
    live: bool,
    table: postgres.TableTarget[ChunkRow],
) -> None:
    """Walk one source and process every file that passes its matcher."""
    walker = localfs.walk_dir(
        source.path,
        recursive=True,
        live=live,
        path_matcher=source.matcher(),
    )
    # Keyed by source name: the default subpath is the function name, so every
    # repo's root README.md would mount to the same path.
    await coco.mount_each(
        coco.component_subpath("process_file", source.name),
        process_file,
        walker.items(),
        source,
        table,
    )


@coco.fn
async def app_main(live: bool) -> None:
    """Mount the table once, declare its indexes, then ingest every source."""
    table = await postgres.mount_table_target(
        PG_DB,
        TABLE_NAME,
        await postgres.TableSchema.from_class(ChunkRow, primary_key=["id"]),
    )
    table.declare_vector_index(column="embedding", metric="cosine", method="hnsw")
    table.declare_sql_command_attachment(
        name="fulltext",
        setup_sql=FULLTEXT_SETUP_SQL,
        teardown_sql=FULLTEXT_TEARDOWN_SQL,
    )

    links_table = await postgres.mount_table_target(
        PG_DB,
        LINKS_TABLE,
        await postgres.TableSchema.from_class(LinkRow, primary_key=["id"]),
    )
    links_table.declare_sql_command_attachment(
        name="link_indexes",
        setup_sql=LINKS_SETUP_SQL,
        teardown_sql=LINKS_TEARDOWN_SQL,
    )

    for source in load_sources():
        logger.info("source: %s (%s, project=%s)", source.name, source.source_type, source.project)
        await ingest_source(source, live, table)

        if source.source_type == "note":
            await ingest_links(source, links_table)


@coco.lifespan
async def cortex_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    """Provide the asyncpg pool for the duration of the run."""
    async with await asyncpg.create_pool(DATABASE_URL) as pool:
        builder.provide(PG_DB, pool)
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the Cortex sources into Postgres")
    parser.add_argument(
        "--live",
        action="store_true",
        help="stay running and re-index on file change",
    )
    args = parser.parse_args()

    app = coco.App("cortex", app_main, args.live)

    coco.start_blocking()
    try:
        app.update_blocking(report_to_stdout=True, live=args.live)
    finally:
        coco.stop_blocking()


if __name__ == "__main__":
    main()
