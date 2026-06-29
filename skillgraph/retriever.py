"""
Retrieval engine — semantic matching + graph traversal.

Pipeline:
  1. Embed the user query (with LRU cache)
  2. Compute cosine similarity against all skill embeddings → seed candidates
  3. Select top-K seeds by semantic similarity
  4. Expand 1-hop neighbors via graph adjacency (independent score, not weighted fusion)
  5. Merge: seeds first, then graph-expanded nodes (deduplicated)
  6. Return top-N skills

Algorithm change (v0.2): seeds and graph nodes are scored independently.
Previously used alpha*semantic + beta*graph_weight which let high seed scores
drown out graph-expanded nodes. Now seeds get their raw semantic score, and
graph nodes get edge_weight as a bonus on top of their own semantic score,
then the two pools are merged with guaranteed slots for each.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
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
        sibling_weight: float = 0.5,
        similar_threshold: float = 0.6,
        # Retrieval config
        seed_count: int = 5,
        graph_slot_ratio: float = 0.4,   # 40% of top_k reserved for graph-expanded
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> None:
        self.skills_dir = str(skills_dir)
        self.external_dirs = external_dirs or []
        self.embedder = embedder or create_embedder(backend)

        self.seed_count = seed_count
        self.graph_slot_ratio = graph_slot_ratio
        self.alpha = alpha
        self.beta = beta
        self._sibling_weight = sibling_weight
        self._similar_threshold = similar_threshold

        self.entries: list[SkillEntry] = []
        self.embeddings: np.ndarray | None = None
        self.graph: SkillGraph | None = None

        # Query embedding cache (text hash → vector)
        self._query_cache: dict[str, np.ndarray] = {}
        self._query_cache_max = 256

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
            sibling_weight=self._sibling_weight,
            similar_threshold=self._similar_threshold,
        )

        return len(self.entries)

    # ── Query embedding with cache ─────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string with LRU caching."""
        # Use hash as cache key (handles long queries)
        key = hashlib.md5(query.encode("utf-8")).hexdigest()

        if key in self._query_cache:
            return self._query_cache[key]

        vec = np.array(self.embedder.embed(query), dtype=np.float32)

        # Add to cache
        self._query_cache[key] = vec
        if len(self._query_cache) > self._query_cache_max:
            # Evict oldest (FIFO)
            oldest = next(iter(self._query_cache))
            del self._query_cache[oldest]

        return vec

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

        # 1. Embed query (cached) and compute cosine similarity
        q_vec = self._embed_query(query)
        q_norm = np.linalg.norm(q_vec) or 1.0
        emb_norms = np.linalg.norm(self.embeddings, axis=1)
        emb_norms = np.where(emb_norms == 0, 1, emb_norms)

        similarities = (self.embeddings @ q_vec) / (emb_norms * q_norm)

        # 2. Select seed candidates (top seed_count by similarity)
        seed_indices = np.argsort(similarities)[::-1][: self.seed_count]
        seeds: dict[str, float] = {}
        seed_set: set[str] = set()
        for idx in seed_indices:
            name = self.entries[idx].name
            seeds[name] = float(similarities[idx])
            seed_set.add(name)

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

        # 3. Graph expansion — collect 1-hop neighbors with edge weights
        # Graph nodes get: their own semantic score + max edge weight from any seed
        graph_candidates: dict[str, dict[str, float]] = {}
        for name, sim in seeds.items():
            edges = self.graph.get_neighbors(name)
            for edge in edges:
                target = edge.target
                if target in seed_set:
                    continue  # Don't expand to nodes already in seeds
                if target not in self._name_to_idx:
                    continue
                target_semantic = float(similarities[self._name_to_idx[target]])
                graph_score = sim * edge.weight + target_semantic * 0.5  # seed_sim × edge_weight + own_semantic × 0.5
                if target in graph_candidates:
                    graph_candidates[target]["graph_score"] = max(
                        graph_candidates[target]["graph_score"], graph_score
                    )
                    graph_candidates[target]["edge_weight"] = max(
                        graph_candidates[target]["edge_weight"], edge.weight
                    )
                else:
                    graph_candidates[target] = {
                        "semantic": target_semantic,
                        "graph_score": graph_score,
                        "edge_weight": edge.weight,
                        "edge_type": edge.edge_type,
                        "via": name,
                    }

        # 4. Build seed results (raw semantic score)
        seed_results: list[tuple[str, float, str]] = [
            (name, sim, "seed") for name, sim in seeds.items()
        ]
        seed_results.sort(key=lambda x: x[1], reverse=True)

        # 5. Build graph results (sorted by graph_score)
        graph_results: list[tuple[str, float, str]] = [
            (name, info["graph_score"], "graph") for name, info in graph_candidates.items()
        ]
        graph_results.sort(key=lambda x: x[1], reverse=True)

        # 6. Merge with guaranteed slots
        # seed_slots = top_k - graph_slots, graph_slots = ceil(top_k * graph_slot_ratio)
        graph_slots = max(1, int(top_k * self.graph_slot_ratio))
        seed_slots = top_k - graph_slots

        merged: list[tuple[str, float, str]] = []
        seen: set[str] = set()

        # Fill seed slots first
        for name, score, source in seed_results[:seed_slots]:
            if name not in seen:
                merged.append((name, score, source))
                seen.add(name)

        # Fill graph slots
        for name, score, source in graph_results:
            if len(merged) >= top_k:
                break
            if name not in seen:
                merged.append((name, score, source))
                seen.add(name)

        # If still not enough, backfill from remaining seeds
        for name, score, source in seed_results[seed_slots:]:
            if len(merged) >= top_k:
                break
            if name not in seen:
                merged.append((name, score, source))
                seen.add(name)

        # 7. Convert to SkillEntry
        results: list[SkillEntry] = []
        for name, score, source in merged:
            entry = self.entries[self._name_to_idx[name]]
            entry.score = score
            entry.source = source
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
                sibling_weight=self._sibling_weight,
                similar_threshold=self._similar_threshold,
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