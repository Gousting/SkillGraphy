"""
Generic adapter — works with any SKILL.md directory layout.

For frameworks not explicitly supported, or custom agent setups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..indexer import SkillEntry
from .base import BaseAdapter


class GenericAdapter(BaseAdapter):
    """Generic adapter for any SKILL.md setup.

    Requires explicit skills_dir parameter.
    """

    framework_name = "generic"

    def __init__(
        self,
        skills_dir: Path | str,
        backend: str = "ollama",
        top_k: int = 8,
        external_dirs: list[Path | str] | None = None,
        **retriever_kwargs: Any,
    ) -> None:
        super().__init__(
            skills_dir=skills_dir,
            backend=backend,
            top_k=top_k,
            external_dirs=external_dirs,
            **retriever_kwargs,
        )
        self._explicit_skills_dir = Path(skills_dir)

    def get_skills_dir(self) -> Path:
        return self._explicit_skills_dir

    def format_prompt(self, skills: list[SkillEntry]) -> str:
        if not skills:
            return ""

        lines: list[str] = ["# Relevant Skills (retrieved by SkillGraph)", ""]

        # Group by category
        by_category: dict[str, list[SkillEntry]] = {}
        for skill in skills:
            by_category.setdefault(skill.category, []).append(skill)

        for category in sorted(by_category.keys()):
            lines.append(f"## {category}")
            for skill in sorted(by_category[category], key=lambda s: -s.score):
                score_str = f" (score={skill.score:.3f})" if skill.score > 0 else ""
                if skill.description:
                    lines.append(f"- **{skill.name}**: {skill.description}{score_str}")
                else:
                    lines.append(f"- **{skill.name}**{score_str}")
            lines.append("")

        return "\n".join(lines)