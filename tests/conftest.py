"""Shared test fixtures for SkillGraph tests."""

import shutil
from pathlib import Path
from typing import Generator

import pytest
import numpy as np


# ── Fixtures: create temporary skill directories ──────────────────────────


SKILL_FIXTURES = {
    "creative/excalidraw": {
        "name": "excalidraw",
        "description": "Hand-drawn Excalidraw JSON diagrams (arch, flow, seq).",
        "related": ["ascii-art", "sketch"],
    },
    "creative/ascii-art": {
        "name": "ascii-art",
        "description": "ASCII art: pyfiglet, cowsay, boxes, image-to-ascii.",
        "related": ["excalidraw"],
    },
    "creative/sketch": {
        "name": "sketch",
        "description": "Throwaway HTML mockups: 2-3 design variants to compare.",
        "related": ["excalidraw"],
    },
    "creative/pixel-art": {
        "name": "pixel-art",
        "description": "Pixel art w/ era palettes (NES, Game Boy, PICO-8).",
    },
    "devops/kanban-orchestrator": {
        "name": "kanban-orchestrator",
        "description": "Decomposition playbook for orchestration system.",
    },
    "devops/webhook-subscriptions": {
        "name": "webhook-subscriptions",
        "description": "Webhook subscriptions: event-driven agent runs.",
    },
    "research/arxiv": {
        "name": "arxiv",
        "description": "Search arXiv papers by keyword, author, category, or ID.",
    },
    "research/blogwatcher": {
        "name": "blogwatcher",
        "description": "Monitor blogs and RSS/Atom feeds via blogwatcher-cli.",
    },
    "software-development/codegraph": {
        "name": "codegraph",
        "description": "Code intelligence — codebase knowledge graph via tree-sitter.",
    },
}


def _write_skill_md(path: Path, meta: dict) -> None:
    """Write a minimal SKILL.md file with YAML frontmatter."""
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
    content = f"---\n{fm}---\n\n# {meta.get('name', 'skill')}\n\nBody text.\n"
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def tmp_skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with fixture SKILL.md files."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for rel_path, meta in SKILL_FIXTURES.items():
        _write_skill_md(skills_dir / rel_path / "SKILL.md", meta)
    return skills_dir


@pytest.fixture
def mock_embeddings() -> np.ndarray:
    """Deterministic embeddings for 9 skills (must match SKILL_FIXTURES order)."""
    np.random.seed(42)
    n = len(SKILL_FIXTURES)
    dim = 64
    emb = np.random.randn(n, dim).astype(np.float32)
    # Normalize to unit length
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns from a fixed embedding pool."""
    from skillgraph.embedder import Embedder
    import numpy as np

    # Pre-generate deterministic embeddings
    np.random.seed(42)
    pool = np.random.randn(200, 64).astype(np.float32)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)

    class MockEmbedder(Embedder):
        def __init__(self):
            self._idx = 0

        @property
        def name(self):
            return "mock"

        @property
        def dim(self):
            return 64

        def embed(self, text: str) -> list[float]:
            self._idx = (self._idx + 1) % len(pool)
            return pool[self._idx].tolist()

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            results = []
            for _ in texts:
                self._idx = (self._idx + 1) % len(pool)
                results.append(pool[self._idx].tolist())
            return results

    return MockEmbedder()