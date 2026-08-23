"""
links.py — Obsidian wikilink extraction and resolution.

Turns `[[double bracket]]` links into a note-to-note edge list, resolved the way
Obsidian resolves them. The edges feed two things in query.py: a hub prior (a
heavily-linked note is one the vault itself treats as authoritative) and one-hop
expansion (following a note's own "canonical note for X is Y" pointers).

Measured on the current vault: 2,643 raw links → 1,947 edges across 377 of 415
notes, with 10 unresolved.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# Fenced and inline code are stripped before parsing, otherwise JavaScript such as
# `[[...coords, coords[0]]` parses as a link.
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_LINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")


def strip_code(text: str) -> str:
    """Remove fenced and inline code spans so code is not parsed as links."""
    return _INLINE_CODE_RE.sub(" ", _FENCE_RE.sub(" ", text))


def build_index(rel_paths: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Index notes for resolution.

    Returns:
        (by_path, by_name): exact lookups (with and without the .md suffix), and
        basename -> candidate paths for bare-name links.
    """
    by_path: dict[str, str] = {}
    by_name: dict[str, list[str]] = defaultdict(list)

    for rel in rel_paths:
        by_path[rel] = rel
        if rel.endswith(".md"):
            by_path[rel[:-3]] = rel
        by_name[Path(rel).stem].append(rel)

    return by_path, by_name


def _shared_prefix(a: str, b: str) -> int:
    """Count leading path segments two paths share."""
    a_parts, b_parts = a.split("/")[:-1], b.split("/")[:-1]
    count = 0
    for x, y in zip(a_parts, b_parts):
        if x != y:
            break
        count += 1
    return count


def _nearest(candidates: list[str], source_rel: str) -> str:
    """
    Pick the candidate closest to the source note in the folder tree.

    Bare-name links are ambiguous: `auth.md` can exist under two different
    projects' `api/` folders. Obsidian
    resolves by proximity, and so must this: taking the first match sends roughly
    half of the ambiguous links into the wrong project's notes.
    """
    return max(candidates, key=lambda c: (_shared_prefix(c, source_rel), -len(c)))


def _normalise_relative(target: str, source_rel: str) -> str:
    """Resolve a ./ or ../ target against the source note's folder."""
    parts: list[str] = []
    for segment in (Path(source_rel).parent / target).as_posix().split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts:
                parts.pop()
        else:
            parts.append(segment)
    return "/".join(parts)


def resolve(target: str, source_rel: str, by_path, by_name) -> str | None:
    """
    Resolve one link target to a vault-relative note path, or None.

    Handles every form present in the vault: full paths, relative paths, bare
    names, `#heading` anchors, and `|alias` display text. The pipe may arrive
    backslash-escaped, because prettier escapes it inside markdown tables and
    vault-sync.sh runs prettier across the vault on every commit.
    """
    target = target.replace("\\|", "|").split("|", 1)[0].strip()

    # [[#Heading]] points inside the same note — not an edge between notes.
    if target.startswith("#"):
        return None

    target = target.split("#", 1)[0].strip()
    if not target:
        return None

    if target.startswith((".", "/")):
        normalised = _normalise_relative(target, source_rel)
        for form in (normalised, normalised + ".md"):
            if form in by_path:
                return by_path[form]
        return None

    for form in (target, target + ".md"):
        if form in by_path:
            return by_path[form]

    candidates = by_name.get(Path(target).stem)
    if candidates:
        return _nearest(candidates, source_rel)

    return None


def extract_edges(text: str, source_rel: str, by_path, by_name) -> set[str]:
    """Return the set of note paths that `source_rel` links to."""
    targets = set()
    for raw in _LINK_RE.findall(strip_code(text)):
        resolved = resolve(raw, source_rel, by_path, by_name)
        if resolved and resolved != source_rel:
            targets.add(resolved)
    return targets
