"""
sources.py — source configuration and the global secret denylist.

Loads sources.yaml into `Source` records and builds each source's file matcher.
The denylist here is appended to every source's exclude patterns, and CocoIndex's
PatternFilePathMatcher applies exclusions over inclusions — so no include pattern
in sources.yaml can pull a secret into the index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from cocoindex.resources.file import PatternFilePathMatcher

# Applied to every source, always. An embedded secret is unrecoverable: it lands
# in the vector store and then in every future search result.
#
# Patterns target file forms that *carry* secrets, not filenames that merely
# mention them. Loose matching on "*secret*"/"*credentials*" was rejected: it
# excludes source files whose subject is credential handling — e.g.
# apps/api/src/common/payfast/platform-credentials.ts, the PayFast crypto seam —
# which hold logic worth indexing, not keys. Over-exclusion loses knowledge
# silently, which is its own failure mode.
SECRET_DENYLIST = [
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/*.crt",
    "**/*.jks",
    "**/*.keystore",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/*.tfvars",
    "**/*.tfstate*",
    "**/credentials.json",
    "**/*serviceaccount*.json",
    "**/.git/**",
]

# Derived output. Code sources only — the vault is prose, and a note named
# CHANGELOG.md would be dropped.
DERIVED_DENYLIST = [
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/.next/**",
    "**/.turbo/**",
    "**/coverage/**",
    "**/*.d.ts",
    "**/CHANGELOG.md",
    "**/generated/**",
    "**/*.generated.*",
    "**/openwiki/**",
    "**/graphify-out/**",
]

CONFIG_PATH = Path(os.getenv("SOURCES_CONFIG", Path(__file__).parent.parent / "sources.yaml"))


@dataclass(frozen=True)
class Source:
    """
    One ingestion source: a directory plus the tags written onto its chunks.

    Pattern fields are tuples, not lists, so the whole record stays hashable —
    it is passed into memoized CocoIndex functions, which key on their arguments.
    """

    name: str
    path: Path
    project: str
    source_type: str
    repo: str | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    # Path prefix -> project, longest match wins. A vault holding several projects
    # cannot be tagged with one project; without this, a generic question matches
    # every project's equivalent note and there is nothing to disambiguate on.
    project_map: tuple[tuple[str, str], ...] = ()
    # Path prefixes whose content is superseded. Kept and searchable, but demoted.
    archived_prefixes: tuple[str, ...] = ()
    # Whole source superseded, e.g. a repo replaced by a rewrite. Same treatment.
    archived: bool = False

    def project_for(self, rel_path: str) -> str:
        """Resolve a file's project from its path, falling back to `project`."""
        best = ""
        found = self.project
        for prefix, project in self.project_map:
            if rel_path.startswith(prefix) and len(prefix) > len(best):
                best, found = prefix, project
        return found

    def is_archived(self, rel_path: str) -> bool:
        """Whether a file is superseded, by whole source or by path prefix."""
        if self.archived:
            return True
        return rel_path.startswith(self.archived_prefixes) if self.archived_prefixes else False

    def matcher(self) -> PatternFilePathMatcher:
        """Build the file matcher, with the shared denylists appended to excludes."""
        shared = SECRET_DENYLIST
        if self.source_type == "code":
            shared = [*DERIVED_DENYLIST, *shared]

        return PatternFilePathMatcher(
            included_patterns=list(self.include) or ["**/*"],
            excluded_patterns=[*self.exclude, *shared],
        )


def load_sources(config_path: Path = CONFIG_PATH) -> list[Source]:
    """
    Read sources.yaml into Source records.

    Raises:
        FileNotFoundError: If a configured source path does not exist — better to
            fail loudly than to silently index nothing.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    sources = []
    for entry in raw:
        path = Path(os.path.expandvars(entry["path"])).expanduser()
        if "$" in str(path):
            raise ValueError(
                f"source '{entry['name']}' path has an unset variable: {path}. "
                "Set CORTEX_ROOT in .env."
            )
        if not path.is_dir():
            raise FileNotFoundError(f"source '{entry['name']}' path does not exist: {path}")

        sources.append(
            Source(
                name=entry["name"],
                path=path,
                project=entry["project"],
                source_type=entry["source_type"],
                repo=entry.get("repo"),
                include=tuple(entry.get("include", ())),
                exclude=tuple(entry.get("exclude", ())),
                project_map=tuple(sorted(entry.get("project_map", {}).items())),
                archived_prefixes=tuple(entry.get("archived_prefixes", ())),
                archived=bool(entry.get("archived", False)),
            )
        )

    return sources
