"""
Hermes Agent adapter — replaces build_skills_system_prompt() output.

Hermes injects all skills as a flat list in the system prompt:
    <available_skills>
      creative:
        - excalidraw: ...
        - sketch: ...
    </available_skills>

This adapter outputs the same format, but only includes the top-K relevant skills.
"""

from __future__ import annotations

from pathlib import Path
import os

from ..indexer import SkillEntry
from .base import BaseAdapter


class HermesAdapter(BaseAdapter):
    """Adapter for Hermes Agent (hermes-agent).

    Skills directory defaults to ~/.hermes/skills/.
    Output format matches Hermes' <available_skills> block.
    """

    framework_name = "hermes"

    def get_skills_dir(self) -> Path:
        hermes_home = os.environ.get(
            "HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes")
        )
        return Path(hermes_home) / "skills"

    def format_prompt(self, skills: list[SkillEntry]) -> str:
        if not skills:
            return ""

        # Group by category for readable output
        by_category: dict[str, list[SkillEntry]] = {}
        for skill in skills:
            by_category.setdefault(skill.category, []).append(skill)

        lines: list[str] = []
        for category in sorted(by_category.keys()):
            lines.append(f"  {category}:")
            for skill in sorted(by_category[category], key=lambda s: (-s.score)):
                if skill.description:
                    lines.append(f"    - {skill.name}: {skill.description}")
                else:
                    lines.append(f"    - {skill.name}")

        return (
            "## Skills (dynamic)\n"
            "Relevant skills retrieved by SkillGraph for this query. "
            "Load matching skills with skill_view(name).\n\n"
            "<available_skills>\n"
            + "\n".join(lines)
            + "\n</available_skills>"
        )