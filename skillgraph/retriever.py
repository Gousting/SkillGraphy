"""
Retrieval engine — semantic matching + graph traversal.

Pipeline:
  1. Embed the user query
  2. Compute cosine similarity against all skill embeddings → seed candidates
  3. Select top-K seeds
  4. Expand 1-hop neighbors via graph adjacency
  5. Rerank by alpha * semantic + beta * graph_weight
  6. Return top-N skills
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import Embedder, create_embedder
from .graph import SkillGraph
from .indexer import SkillEntry, index_skills


# ── Retriever ───────────────────────────────────────────────────────────────


class Retriever:
    """Semantic + graph skill retriever.

    Usage:
        retriever = Retriever(skills_dir="~/.hermes/skills", backend="ollama")
        retriever.build()
        results = retriever.retrieve("generate a diagram", top_k=8)
    """

    def __init__(
        self,
        skills_dir: Path | str,
        backend: str = "ollama",
        external_dirs: list[Path | str] | None = None,
        embedder: Embedder | None = None,
        # Graph config
        sibling_weight: float = 0.3,
        similar_threshold: float = 0.75,
        # Retrieval config
        seed_count: int = 3,
        expansion_top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> None:
        self.skills_dir = str(skills_dir)
        self.external_dirs = external_dirs or []
        self.embedder = embedder or create_embedder(backend)

        self.seed_count = seed_count
        self.expansion_top_k = expansion_top_k
        self.alpha = alpha
        self.beta = beta

        self.entries: list[SkillEntry] = []
        self.embeddings: np.ndarray | None = None
        self.graph: SkillGraph | None = None

    # ── Build ──────────────────────────────────────────────────────────

    def build(self) -> int:
        """Index skills, compute embeddings, and build the graph.

        Returns:
            Number of skills indexed.
        """
        # 1. Parse all SKILL.md files
        self.entries = index_skills(self.skills_dir, self.external_dirs)
        if not self.entries:
            return 0

        # 2. Embed all skill descriptions
        texts = [
            f"{e.name}: {e.description}" if e.description else e.name
            for e in self.entries
        ]
        self.embeddings = np.array(self.embedder.embed_batch(texts), dtype=np.float32)

        # 3. Build graph
        self.graph = SkillGraph(
            self.entries,
            self.embeddings,
            sibling_weight=self.sibling_weight if hasattr(self, '_sibling_weight') else 0.3,
            similar_threshold=self.similar_threshold if hasattr(self, '_similar_threshold') else 0.75,
        )

        return len(self.entries)

    # ── Retrieve ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        expand_graph: bool = True,
    ) -> list[SkillEntry]:
        """Retrieve the most relevant skills for a query.

        Args:
            query: Natural language query.
            top_k: Number of skills to return.
            expand_graph: Whether to do 1-hop graph expansion.

        Returns:
            List of SkillEntry objects with `score` and `source` fields set.
        """
        if not self.entries or self.embeddings is None:
            return []

        # 1. Embed query and compute cosine similarity
        q_vec = np.array(self.embedder.embed(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) or 1.0
        emb_norms = np.linalg.norm(self.embeddings, axis=1)
        emb_norms = np.where(emb_norms == 0, 1, emb_norms)

        similarities = (self.embeddings @ q_vec) / (emb_norms * q_norm)

        # 2. Select seed candidates (top seed_count by similarity)
        seed_indices = np.argsort(similarities)[::-1][: self.seed_count]
        seeds: dict[str, float] = {}
        for idx in seed_indices:
            name = self.entries[idx].name
            seeds[name] = float(similarities[idx])

        if not expand_graph or self.graph is None:
            # Pure semantic — return top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                entry = self.entries[idx]
                entry.score = float(similarities[idx])
                entry.source = "seed"
                results.append(entry)
            return results

        # 3. Graph expansion — collect 1-hop neighbors
        candidates: dict[str, dict[str, float]] = {}
        # Add seeds
        for name, sim in seeds.items():
            candidates[name] = {"semantic": sim, "graph": 0.0, "source": "seed"}

        # Expand from each seed
        for name, sim in seeds.items():
            edges = self.graph.get_neighbors(name)
            for edge in edges:
                if edge.target not in self.entries:
                    continue
                if edge.target in candidates:
                    # Augment graph score — keep the max edge weight
                    candidates[edge.target]["graph"] = max(
                        candidates[edge.target]["graph"], edge.weight
                    )
                else:
                    candidates[edge.target] = {
                        "semantic": float(similarities[self._idx(edge.target)]),
                        "graph": edge.weight,
                        "source": "graph",
                    }

        # 4. Rerank: score = alpha * semantic + beta * graph_weight
        scored: list[tuple[str, float]] = []
        for name, info in candidates.items():
            score = self.alpha * info["semantic"] + self.beta * info["graph"]
            scored.append((name, score, info))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 5. Return top_k
        results: list[SkillEntry] = []
        for name, score, info in scored[:top_k]:
            entry = self.entries[self._name_list_index(name)]
            entry.score = score
            entry.source = info["source"]
            results.append(entry)

        return results

    def _idx(self, name: str) -> int:
        """Get the embedding index for a skill name."""
        return self._name_to_idx.get(name, 0)

    @property
    def _name_to_idx(self) -> dict[str, int]:
        return {e.name: i for i, e in enumerate(self.entries)}

    def _name_list_index(self, name: str) -> int:
        """Get the list index for a skill name."""
        return self._name_to_idx.get(name, 0)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, output_dir: Path | str) -> None:
        """Save index + embeddings + graph to output directory.

        Files:
          skill_index.json   — skill metadata
          embeddings.npy     — embedding matrix
          adjacency.json     — graph edges
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save skill index
        index_data = [e.to_index_dict() for e in self.entries]
        (output_dir / "skill_index.json").write_text(
            json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Save embeddings
        if self.embeddings is not None:
            np.save(output_dir / "embeddings.npy", self.embeddings)

        # Save graph
        if self.graph is not None:
            self.graph.save(output_dir / "adjacency.json")

    def load(self, input_dir: Path | str) -> None:
        """Load index + embeddings + graph from output directory."""
        input_dir = Path(input_dir)

        # Load skill index
        index_data = json.loads(
            (input_dir / "skill_index.json").read_text(encoding="utf-8")
        )
        self.entries = [
            SkillEntry(
                name=d["name"],
                description=d.get("description", ""),
                category=d.get("category", "general"),
                related=d.get("related", []),
                tools=d.get("tools", []),
                platforms=d.get("platforms", []),
                conditions=d.get("conditions", {}),
            )
            for d in index_data
        ]

        # Load embeddings
        emb_path = input_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)

        # Load graph
        adj_path = input_dir / "adjacency.json"
        if adj_path.exists() and self.embeddings is not None:
            graph_data = SkillGraph.load(adj_path)
            # Rebuild graph from entries + embeddings
            self.graph = SkillGraph(
                self.entries,
                self.embeddings,
            )

    # ── Stats ──────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        categories: dict[str, int] = {}
        for e in self.entries:
            categories[e.category] = categories.get(e.category, 0) + 1
        return {
            "total_skills": len(self.entries),
            "categories": categories,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "embedder": self.embedder.name,
            "graph": {
                "nodes": self.graph.num_nodes if self.graph else 0,
                "edges": self.graph.num_edges if self.graph else 0,
                "edge_types": self.graph.edge_type_counts if self.graph else {},
            },
        }