"""
Claude Code adapter — writes retrieved skills to a context file.

Claude Code reads skill context from .claude/ directory.
This adapter writes only the top-K relevant skills to a context file
that Claude Code can pick up.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..indexer import SkillEntry
from .base import BaseAdapter


class ClaudeCodeAdapter(BaseAdapter):
    """Adapter for Claude Code (Anthropic's CLI agent).

    Skills directory defaults to .claude/skills/ in the current project.
    Output format: writes a context file with relevant skills.
    """

    framework_name = "claude-code"

    def get_skills_dir(self) -> Path:
        # Claude Code stores skills in .claude/skills/ within the project
        return Path(os.getcwd()) / ".claude" / "skills"

    def format_prompt(self, skills: list[SkillEntry]) -> str:
        if not skills:
            return ""

        lines: list[str] = []
        lines.append("# Relevant Skills (retrieved by SkillGraph)")
        lines.append("")
        for skill in skills:
            lines.append(f"## {skill.name}")
            if skill.description:
                lines.append(f"  {skill.description}")
            if skill.category != "general":
                lines.append(f"  *Category: {skill.category}*")
            if skill.related:
                lines.append(f"  *Related: {', '.join(skill.related)}*")
            lines.append("")

        return "\n".join(lines)

    def write_context_file(
        self,
        user_message: str,
        top_k: int | None = None,
        output_path: Path | str | None = None,
    ) -> Path:
        """Retrieve skills and write them to a context file.

        Args:
            user_message: User query for semantic matching.
            top_k: Number of skills to retrieve.
            output_path: Path to write context file. Defaults to .claude/skills-context.md.

        Returns:
            Path to the written file.
        """
        k = top_k or self.top_k
        skills = self.retriever.retrieve(user_message, top_k=k)
        content = self.format_prompt(skills)

        if output_path is None:
            output_path = Path(os.getcwd()) / ".claude" / "skills-context.md"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path