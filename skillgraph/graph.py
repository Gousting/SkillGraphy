"""
Knowledge graph builder for skills.

Constructs an adjacency list from SkillEntry objects using three edge types:
  - `related`: explicitly declared in SKILL.md frontmatter (weight=1.0)
  - `sibling`: same category (weight=sibling_weight, default=0.3)
  - `similar`: embedding cosine similarity > threshold (weight=cosine)

The graph is undirected for sibling/similar edges, but `related` edges are
treated as undirected by default (i.e. A→B implies B→A).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .indexer import SkillEntry


# ── Edge Model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Edge:
    """A directed edge in the skill graph."""

    source: str
    target: str
    edge_type: str  # "related" | "sibling" | "similar"
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "weight": round(self.weight, 4),
        }


# ── Graph ───────────────────────────────────────────────────────────────────


class SkillGraph:
    """Knowledge graph over skill embeddings.

    Stores an adjacency list mapping skill_name -> list[Edge].
    Supports serialization to/from JSON for persistence.
    """

    def __init__(
        self,
        entries: list[SkillEntry] | None = None,
        embeddings: np.ndarray | None = None,
        *,
        sibling_weight: float = 0.3,
        similar_threshold: float = 0.75,
    ) -> None:
        self.entries: dict[str, SkillEntry] = {}
        self.adjacency: dict[str, list[Edge]] = {}
        self.embeddings: np.ndarray | None = embeddings
        self._name_to_idx: dict[str, int] = {}

        self.sibling_weight = sibling_weight
        self.similar_threshold = similar_threshold

        if entries and embeddings is not None:
            self.build(entries, embeddings)

    # ── Build ──────────────────────────────────────────────────────────

    def build(
        self,
        entries: list[SkillEntry],
        embeddings: np.ndarray,
    ) -> None:
        """Build the full graph from entries + their embeddings.

        Args:
            entries: All SkillEntry objects.
            embeddings: (N, D) numpy array of embeddings.
        """
        assert len(entries) == len(embeddings), (
            f"Count mismatch: {len(entries)} entries vs {len(embeddings)} embeddings"
        )

        self.entries = {e.name: e for e in entries}
        self.embeddings = embeddings
        self._name_to_idx = {e.name: i for i, e in enumerate(entries)}
        self.adjacency = {name: [] for name in self.entries}

        self._build_related_edges(entries)
        self._build_sibling_edges(entries)
        self._build_similar_edges(entries, embeddings)

    def _build_related_edges(self, entries: list[SkillEntry]) -> None:
        """Build edges from explicit `related:` frontmatter fields.

        Related edges are bidirectional: if A lists B, A→B and B→A are both added.
        """
        for entry in entries:
            for related_name in entry.related:
                target = related_name.strip()
                if not target or target not in self.entries:
                    continue
                self._add_edge(entry.name, target, "related", 1.0)
                self._add_edge(target, entry.name, "related", 1.0)

    def _build_sibling_edges(self, entries: list[SkillEntry]) -> None:
        """Build edges between skills in the same category.

        Avoids creating duplicate edges if already connected by `related`.
        """
        by_category: dict[str, list[str]] = {}
        for entry in entries:
            by_category.setdefault(entry.category, []).append(entry.name)

        for _cat, names in by_category.items():
            if len(names) < 2:
                continue
            for i, name_a in enumerate(names):
                for name_b in names[i + 1 :]:
                    if self._has_edge(name_a, name_b):
                        continue
                    self._add_edge(name_a, name_b, "sibling", self.sibling_weight)
                    self._add_edge(name_b, name_a, "sibling", self.sibling_weight)

    def _build_similar_edges(
        self,
        entries: list[SkillEntry],
        embeddings: np.ndarray,
    ) -> None:
        """Build edges from embedding cosine similarity above threshold.

        Uses normalized embeddings for cosine similarity. Avoids duplicates
        and self-loops.
        """
        if len(embeddings) < 2:
            return

        # Normalize embeddings (in-place copy)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1, norms)

        # Compute full similarity matrix (N x N)
        sim_matrix = normalized @ normalized.T

        n = len(entries)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim < self.similar_threshold:
                    continue
                name_a = entries[i].name
                name_b = entries[j].name
                if self._has_edge(name_a, name_b):
                    continue
                self._add_edge(name_a, name_b, "similar", sim)
                self._add_edge(name_b, name_a, "similar", sim)

    # ── Edge helpers ───────────────────────────────────────────────────

    def _add_edge(
        self, source: str, target: str, edge_type: str, weight: float
    ) -> None:
        """Add an edge to the adjacency list (no self-loops)."""
        if source == target:
            return
        self.adjacency.setdefault(source, []).append(
            Edge(source=source, target=target, edge_type=edge_type, weight=weight)
        )

    def _has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists (any type) from source to target."""
        return any(
            edge.target == target for edge in self.adjacency.get(source, [])
        )

    # ── Query ─────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        name: str,
        max_weight: float = 0.0,
        edge_types: list[str] | None = None,
    ) -> list[Edge]:
        """Get outgoing edges from a node.

        Args:
            name: Skill name.
            max_weight: Minimum weight threshold (0 = all edges).
            edge_types: Filter by edge types (None = all types).

        Returns:
            List of Edge objects.
        """
        edges = self.adjacency.get(name, [])
        if edge_types:
            edges = [e for e in edges if e.edge_type in edge_types]
        if max_weight > 0:
            edges = [e for e in edges if e.weight >= max_weight]
        return edges

    def get_edge(self, source: str, target: str) -> Edge | None:
        """Get the strongest edge between source and target."""
        edges = [e for e in self.adjacency.get(source, []) if e.target == target]
        if not edges:
            return None
        return max(edges, key=lambda e: e.weight)

    @property
    def num_nodes(self) -> int:
        return len(self.entries)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self.adjacency.values())

    @property
    def edge_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edges in self.adjacency.values():
            for e in edges:
                counts[e.edge_type] = counts.get(e.edge_type, 0) + 1
        return counts

    # ── Serialization ─────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Save the graph (adjacency list only) to JSON."""
        path = Path(path)
        data = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "sibling_weight": self.sibling_weight,
            "similar_threshold": self.similar_threshold,
            "edges": [e.to_dict() for adj in self.adjacency.values() for e in adj],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> dict[str, Any]:
        """Load graph JSON (returns raw dict; use with SkillGraph.from_dict)."""
        return json.loads(Path(path).read_text(encoding="utf-8"))