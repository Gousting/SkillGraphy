"""
Base adapter protocol — define the interface all framework adapters implement.

Every adapter does two things:
  1. get_skills_dir() — where to find SKILL.md files for this framework
  2. build_prompt(user_message, top_k) — format the retrieved skills for injection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..indexer import SkillEntry
from ..retriever import Retriever


class BaseAdapter(ABC):
    """Abstract base class for framework adapters.

    Subclasses define:
      - `get_skills_dir()`: Where SKILL.md files live for this framework
      - `format_prompt(skills)`: How to format retrieved skills into a prompt
    """

    framework_name: str = "base"

    def __init__(
        self,
        skills_dir: Path | str | None = None,
        backend: str = "ollama",
        top_k: int = 8,
        **retriever_kwargs: Any,
    ) -> None:
        self.top_k = top_k
        self._retriever: Retriever | None = None
        self._skills_dir: Path | str | None = skills_dir
        self._backend = backend
        self._retriever_kwargs = retriever_kwargs

    # ── Abstract methods ──────────────────────────────────────────────

    @abstractmethod
    def get_skills_dir(self) -> Path:
        """Return the path to the skills directory for this framework."""
        ...

    @abstractmethod
    def format_prompt(self, skills: list[SkillEntry]) -> str:
        """Format retrieved skills into a system prompt fragment."""
        ...

    # ── Shared logic ───────────────────────────────────────────────────

    @property
    def retriever(self) -> Retriever:
        """Lazy-initialize the retriever on first access."""
        if self._retriever is None:
            skills_dir = self._skills_dir or self.get_skills_dir()
            self._retriever = Retriever(
                skills_dir=skills_dir,
                backend=self._backend,
                **self._retriever_kwargs,
            )
            self._retriever.build()
        return self._retriever

    def build_prompt(self, user_message: str, top_k: int | None = None) -> str:
        """Retrieve relevant skills and format them as a prompt fragment.

        Args:
            user_message: The user's incoming message (used for semantic matching).
            top_k: Override default top_k for this call.

        Returns:
            Formatted prompt string to inject into the system prompt.
        """
        k = top_k or self.top_k
        skills = self.retriever.retrieve(user_message, top_k=k)
        return self.format_prompt(skills)

    def retrieve(self, user_message: str, top_k: int | None = None) -> list[SkillEntry]:
        """Retrieve relevant skills without formatting."""
        k = top_k or self.top_k
        return self.retriever.retrieve(user_message, top_k=k)

    @property
    def stats(self) -> dict[str, Any]:
        """Return retriever stats."""
        return self.retriever.stats