"""
SKILL.md indexer — scans directories for SKILL.md files and extracts metadata.

This is the entry point of the pipeline: SKILL.md → parsed metadata → SkillEntry objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class SkillEntry:
    """A single skill parsed from a SKILL.md file."""

    name: str
    description: str = ""
    category: str = "general"
    path: Path | None = None
    related: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    # Runtime fields (populated during retrieval)
    score: float = 0.0
    source: str = "seed"  # "seed" | "graph" | "reranked"

    def to_index_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage (exclude runtime fields)."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "path": str(self.path) if self.path else "",
            "related": self.related,
            "tools": self.tools,
            "platforms": self.platforms,
            "conditions": self.conditions,
        }


# ── Frontmatter Parser ──────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL | re.MULTILINE
)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (metadata_dict, body_text).
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    fm_text, body = match.groups()
    try:
        meta = yaml.safe_load(fm_text) or {}
        if not isinstance(meta, dict):
            return {}, content
    except yaml.YAMLError:
        return {}, content

    return meta, body


# ── Iterators ───────────────────────────────────────────────────────────────


def iter_skill_files(skills_dir: Path) -> list[Path]:
    """Find all SKILL.md files under a directory.

    Supports nested categories: skills_dir/creative/excalidraw/SKILL.md
    """
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []
    return sorted(skills_dir.rglob("SKILL.md"))


def iter_description_files(skills_dir: Path) -> list[Path]:
    """Find all DESCRIPTION.md files (category-level descriptions)."""
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []
    return sorted(skills_dir.rglob("DESCRIPTION.md"))


# ── Indexer ─────────────────────────────────────────────────────────────────


def index_skills(
    skills_dir: Path | str,
    external_dirs: list[Path | str] | None = None,
) -> list[SkillEntry]:
    """Scan a skills directory and return parsed SkillEntry objects.

    Args:
        skills_dir: Primary skills directory (e.g. ~/.hermes/skills).
        external_dirs: Additional read-only skill directories (plugins, etc.).

    Returns:
        List of SkillEntry objects, deduplicated by name (first occurrence wins).
    """
    skills_dir = Path(skills_dir).expanduser()
    all_dirs = [skills_dir] + [Path(d).expanduser() for d in (external_dirs or [])]

    entries: list[SkillEntry] = []
    seen_names: set[str] = set()

    for skill_dir in all_dirs:
        if not skill_dir.exists():
            continue

        for skill_file in iter_skill_files(skill_dir):
            entry = _parse_skill_file(skill_file, skill_dir)
            if entry is None:
                continue
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            entries.append(entry)

    return entries


def _parse_skill_file(path: Path, base_dir: Path) -> SkillEntry | None:
    """Parse a single SKILL.md file into a SkillEntry."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta, _body = parse_frontmatter(content)

    # Skill name: frontmatter `name` field, or directory name as fallback
    name = meta.get("name") or path.parent.name
    if not name:
        return None

    # Category: relative parent path, or frontmatter `category` field
    category = meta.get("category") or "general"
    if category == "general":
        try:
            rel = path.parent.relative_to(base_dir)
            if len(rel.parts) > 1:
                category = "/".join(rel.parts[:-1])
            elif len(rel.parts) == 1:
                category = rel.parts[0]
        except ValueError:
            pass

    related = meta.get("related") or []
    if isinstance(related, str):
        related = [related]
    if not isinstance(related, list):
        related = []

    tools = meta.get("tools") or []
    if isinstance(tools, str):
        tools = [tools]
    if not isinstance(tools, list):
        tools = []

    platforms = meta.get("platforms") or []
    if isinstance(platforms, str):
        platforms = [platforms]
    if not isinstance(platforms, list):
        platforms = []

    conditions = meta.get("conditions") or {}
    if not isinstance(conditions, dict):
        conditions = {}

    description = meta.get("description") or ""
    description = str(description).strip()

    return SkillEntry(
        name=str(name).strip(),
        description=description,
        category=str(category).strip(),
        path=path,
        related=[str(r).strip() for r in related],
        tools=[str(t).strip() for t in tools],
        platforms=[str(p).strip() for p in platforms],
        conditions=conditions,
    )


def parse_category_descriptions(skills_dir: Path | str) -> dict[str, str]:
    """Read DESCRIPTION.md files to get category-level descriptions.

    Returns a dict like {"creative": "Creative content generation..."}.
    """
    skills_dir = Path(skills_dir).expanduser()
    if not skills_dir.exists():
        return {}

    result: dict[str, str] = {}
    for desc_file in iter_description_files(skills_dir):
        try:
            content = desc_file.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _ = parse_frontmatter(content)
        desc = meta.get("description")
        if not desc:
            continue
        try:
            rel = desc_file.relative_to(skills_dir)
            cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
        except ValueError:
            cat = "general"
        result[str(cat)] = str(desc).strip().strip("'\"")

    return result