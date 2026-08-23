"""
query.py — Hybrid search against the Cortex Postgres index.

Runs two retrievers over the same table and fuses them with reciprocal rank
fusion: vector similarity (paraphrase) and Postgres full-text (exact tokens such
as identifiers, flag names and error strings, which embeddings blur).

Results can be scoped by project and source type, which is how an agent narrows
to one codebase without a separate routing step.

This module does not call an LLM — it only handles retrieval. That is main.py's job.

Usage:
    python query.py "How does the dunning ladder work?"
    python query.py "dunningStage" --mode fulltext
    python query.py "how are quotes priced?" --project pms --source-type code
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Literal

import asyncpg
from dotenv import load_dotenv

from embedder import build_embedder

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cortex:cortex@localhost:5433/cortex")
TOP_K = int(os.getenv("TOP_K", "5"))

# Wider than the final k so fusion has room to disagree and reorder.
CANDIDATES = 50

# RRF smoothing. Higher flattens any one retriever's top rank.
RRF_K = 60

# Cap per file so one heavily-chunked document cannot monopolise the results.
MAX_PER_FILE = 3

# Note-restricted searches only. Inbound link counts exist for notes and never
# for code, so applying this to a mixed search silently pushes code down.
HUB_WEIGHT = 0.5

# Top note hits expanded, and neighbours taken from each.
EXPAND_FROM = 3
EXPAND_PER_HIT = 2

# Above this a note is an index or changelog, so a single edge carries no signal.
MAX_SOURCE_OUT_DEGREE = 20

# A linked note still has to be about the question.
EXPAND_MIN_SCORE = 0.45

Mode = Literal["hybrid", "vector", "fulltext"]

# Demotes archived chunks after fusion rather than removing them.
ARCHIVE_PENALTY = 0.35

_SELECT = """
    id, source_type, project, repo, path, language, heading, start_line, archived, content
"""


def _filter_clause(project: str | None, source_type: str | None, start: int) -> tuple[str, list]:
    """Build an optional WHERE fragment and its parameters, numbered from `start`."""
    clauses, params = [], []

    if project:
        clauses.append(f"project = ${start + len(params)}")
        params.append(project)
    if source_type:
        clauses.append(f"source_type = ${start + len(params)}")
        params.append(source_type)

    return (" AND " + " AND ".join(clauses) if clauses else ""), params


async def _vector_hits(conn, vector: list[float], project, source_type, limit) -> list[dict]:
    """Rank by cosine distance against the query embedding."""
    literal = "[" + ",".join(str(x) for x in vector) + "]"
    where, params = _filter_clause(project, source_type, start=3)

    rows = await conn.fetch(
        f"""
        SELECT {_SELECT}, 1 - (embedding <=> $1::vector) AS score
        FROM cortex_chunks
        WHERE TRUE {where}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        literal,
        limit,
        *params,
    )
    return [dict(r) for r in rows]


async def _fulltext_hits(conn, question: str, project, source_type, limit) -> list[dict]:
    """
    Rank by Postgres full-text relevance.

    Uses websearch_to_tsquery, not plainto_tsquery: plainto ANDs every term, so a
    natural-language question of more than a few words matches nothing at all.
    """
    where, params = _filter_clause(project, source_type, start=3)

    rows = await conn.fetch(
        f"""
        SELECT {_SELECT}, ts_rank(content_tsv, websearch_to_tsquery('english', $1)) AS score
        FROM cortex_chunks
        WHERE content_tsv @@ websearch_to_tsquery('english', $1) {where}
        ORDER BY score DESC
        LIMIT $2
        """,
        question,
        limit,
        *params,
    )
    return [dict(r) for r in rows]


async def _inbound_counts(conn, paths: list[str]) -> dict[str, int]:
    """Count inbound wikilinks for the given note paths."""
    if not paths:
        return {}

    rows = await conn.fetch(
        """
        SELECT dst_path, count(*) AS inbound
        FROM cortex_links
        WHERE dst_path = ANY($1)
        GROUP BY dst_path
        """,
        paths,
    )
    return {r["dst_path"]: r["inbound"] for r in rows}


def _hub_ranked(candidates: list[dict], inbound: dict[str, int]) -> list[dict]:
    """
    Rank the already-retrieved candidates by how heavily linked they are.

    This is deliberately a re-ranking of the query's own matches, not a global
    list: a note with 97 inbound links is authoritative *when relevant*, and
    injecting it into every search regardless of the question would be noise.
    """
    linked = [row for row in candidates if inbound.get(row["path"], 0) > 0]
    return sorted(linked, key=lambda row: inbound[row["path"]], reverse=True)


async def _expand_via_links(
    conn, vector: list[float], hits: list[dict], project, exclude: set[str]
) -> list[dict]:
    """
    Pull the best chunk from notes that the top hits link to.

    Notes here routinely delegate — billing.md says the canonical source for the
    numbers is pricing-model — so following those pointers surfaces the note the
    author explicitly nominated, which plain ranking ignores.
    """
    sources = [h["path"] for h in hits[:EXPAND_FROM] if h["source_type"] == "note"]
    if not sources:
        return []

    edges = await conn.fetch(
        """
        SELECT src_path, dst_path
        FROM cortex_links
        WHERE src_path = ANY($1)
          AND src_path IN (
              SELECT src_path FROM cortex_links
              WHERE src_path = ANY($1)
              GROUP BY src_path
              HAVING count(*) <= $2
          )
        """,
        sources,
        MAX_SOURCE_OUT_DEGREE,
    )

    # Keep the linking note so results can say which hit pulled them in.
    via: dict[str, str] = {}
    for edge in edges:
        if edge["dst_path"] not in exclude:
            via.setdefault(edge["dst_path"], edge["src_path"])
    if not via:
        return []

    literal = "[" + ",".join(str(x) for x in vector) + "]"
    where, params = _filter_clause(project, None, start=5)

    rows = await conn.fetch(
        f"""
        SELECT * FROM (
            SELECT DISTINCT ON (path) {_SELECT}, 1 - (embedding <=> $1::vector) AS score
            FROM cortex_chunks
            WHERE path = ANY($2) {where}
            ORDER BY path, embedding <=> $1::vector
        ) best
        WHERE score >= $3
        ORDER BY score DESC
        LIMIT $4
        """,
        literal,
        list(via),
        EXPAND_MIN_SCORE,
        EXPAND_FROM * EXPAND_PER_HIT,
        *params,
    )

    expanded = []
    for row in rows:
        item = dict(row)
        item["via"] = via[row["path"]]
        expanded.append(item)
    return expanded


def _fuse(*ranked_lists: tuple[list[dict], float]) -> list[dict]:
    """Fuse weighted ranked lists: score = sum weight / (RRF_K + rank)."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}

    for ranked, weight in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            scores[row["id"]] = scores.get(row["id"], 0.0) + weight / (RRF_K + rank)
            rows.setdefault(row["id"], row)

    fused = []
    for chunk_id, score in scores.items():
        row = dict(rows[chunk_id])
        row["score"] = score
        fused.append(row)

    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused


def _demote_archived(rows: list[dict]) -> list[dict]:
    """Scale down archived chunks and re-sort. Applied once, whatever the mode."""
    out = []
    for row in rows:
        item = dict(row)
        if item.get("archived"):
            item["score"] = float(item["score"]) * ARCHIVE_PENALTY
        out.append(item)

    out.sort(key=lambda r: float(r["score"]), reverse=True)
    return out


def _cap_per_file(rows: list[dict], k: int) -> list[dict]:
    """Keep the best MAX_PER_FILE chunks per path, preserving order, until k."""
    seen: dict[str, int] = {}
    kept = []

    for row in rows:
        count = seen.get(row["path"], 0)
        if count >= MAX_PER_FILE:
            continue
        seen[row["path"]] = count + 1
        kept.append(row)
        if len(kept) == k:
            break

    return kept


def _shape(row: dict) -> dict:
    """Normalise a row into the result shape callers (and the MCP tool) expect."""
    shaped = {
        "source": row["path"],
        "file_name": row["path"].rsplit("/", 1)[-1],
        "heading": row["heading"] or "(no heading)",
        "content": row["content"],
        "score": round(float(row["score"]), 4),
        "source_type": row["source_type"],
        "project": row["project"],
        "repo": row["repo"],
        "language": row["language"],
        "start_line": row["start_line"],
    }

    # Linked rows carry cosine similarity, fused rows an RRF score. Not comparable.
    if row.get("via"):
        shaped["via"] = row["via"]
        shaped["retrieval"] = "linked"
    else:
        shaped["retrieval"] = "fused"

    return shaped


_POOL: asyncpg.Pool | None = None
_POOL_LOCK = asyncio.Lock()


async def _get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first use."""
    global _POOL
    if _POOL is None:
        async with _POOL_LOCK:
            if _POOL is None:
                _POOL = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _POOL


async def close_pool() -> None:
    """Close the shared pool. For short-lived callers; the server keeps it open."""
    global _POOL
    if _POOL is not None:
        await _POOL.close()
        _POOL = None


async def search(
    question: str,
    k: int = TOP_K,
    project: str | None = None,
    source_type: str | None = None,
    mode: Mode = "hybrid",
    expand_links: bool = True,
) -> list[dict]:
    """
    Search the Cortex index and return the most relevant chunks.

    In hybrid mode, three signals are fused: vector similarity, full-text rank, and
    a hub prior from the vault's own wikilink graph. Results may then be extended
    with chunks from notes the winners link to, marked with a "via" field.

    Args:
        question: Natural language question, or an exact identifier.
        k: Number of chunks to return before link expansion.
        project: Restrict to one project (e.g. "pms").
        source_type: Restrict to "note" or "code".
        mode: "hybrid" (fused), "vector" (semantic only), or "fulltext" (exact only).
        expand_links: Append chunks from notes the top hits wikilink to.

    Returns:
        list[dict]: Hits with source, heading, content, score and location metadata.
    """
    pool = await _get_pool()

    async with pool.acquire() as conn:
        vector: list[float] | None = None

        if mode == "fulltext":
            ranked = await _fulltext_hits(conn, question, project, source_type, CANDIDATES)
        else:
            vector = await build_embedder().embed_query(question)
            vector_hits = await _vector_hits(conn, vector, project, source_type, CANDIDATES)

            if mode == "vector":
                ranked = vector_hits
            else:
                fulltext_hits = await _fulltext_hits(
                    conn, question, project, source_type, CANDIDATES
                )
                lists = [(vector_hits, 1.0), (fulltext_hits, 1.0)]

                # Note-only: the hub signal does not exist for code. See HUB_WEIGHT.
                if source_type == "note":
                    candidates = list(
                        {r["id"]: r for r in vector_hits + fulltext_hits}.values()
                    )
                    inbound = await _inbound_counts(conn, [r["path"] for r in candidates])
                    lists.append((_hub_ranked(candidates, inbound), HUB_WEIGHT))

                ranked = _fuse(*lists)

        hits = _cap_per_file(_demote_archived(ranked), k)

        if expand_links and vector is not None:
            hits += await _expand_via_links(
                conn, vector, hits, project, exclude={h["path"] for h in hits}
            )

        return [_shape(row) for row in hits]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Search your Cortex index")
    parser.add_argument("question", help="Question, or an exact identifier")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--project", default=None, help='e.g. "pms"')
    parser.add_argument("--source-type", default=None, choices=["note", "code"])
    parser.add_argument(
        "--mode", default="hybrid", choices=["hybrid", "vector", "fulltext"]
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="skip pulling in notes that the top hits wikilink to",
    )
    args = parser.parse_args()

    async def run() -> list[dict]:
        try:
            return await search(
                args.question,
                k=args.top_k,
                project=args.project,
                source_type=args.source_type,
                mode=args.mode,
                expand_links=not args.no_expand,
            )
        finally:
            await close_pool()

    hits = asyncio.run(run())

    if not hits:
        print("No matches. Has flow.py been run?")
        return

    print(f'\n"{args.question}"  ({args.mode}, {len(hits)} hits)\n')
    for i, hit in enumerate(hits, 1):
        location = f"{hit['source']}:{hit['start_line']}"
        tag = hit["repo"] or hit["project"]
        marker = f"  ↳ linked from {hit['via']}" if hit.get("via") else ""
        print(
            f"[{i}] {hit['retrieval']}={hit['score']}  "
            f"[{hit['source_type']}/{tag}]  {location}{marker}"
        )
        print(f"    {hit['heading']}")
        print(f"    {hit['content'][:200].strip()}")
        print("─" * 72)


if __name__ == "__main__":
    _main()
