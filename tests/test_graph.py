"""Tests for the knowledge graph builder."""

import numpy as np

from skillgraph.indexer import SkillEntry
from skillgraph.graph import SkillGraph, Edge


class TestRelatedEdges:
    def test_related_edges_are_bidirectional(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        exc_edges = graph.get_neighbors("excalidraw")
        targets = {e.target for e in exc_edges}
        assert "ascii-art" in targets
        assert "sketch" in targets

        # Bidirectional
        ascii_edges = graph.get_neighbors("ascii-art")
        ascii_targets = {e.target for e in ascii_edges}
        assert "excalidraw" in ascii_targets

    def test_related_edge_weight(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        exc_edges = graph.get_neighbors("excalidraw", edge_types=["related"])
        for edge in exc_edges:
            assert edge.weight == 1.0
            assert edge.edge_type == "related"


class TestSiblingEdges:
    def test_same_category_creates_sibling_edge(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        # pixel-art and excalidraw are in creative category
        pixel_edges = graph.get_neighbors("pixel-art")
        targets = {e.target: e for e in pixel_edges}
        assert "excalidraw" in targets
        assert targets["excalidraw"].edge_type in ("sibling", "related")

    def test_sibling_weight_default(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        # arxiv and blogwatcher are in research, not related
        arxiv_edges = graph.get_neighbors("arxiv")
        sibling_edges = [e for e in arxiv_edges if e.edge_type == "sibling"]
        for edge in sibling_edges:
            assert edge.weight == 0.3


class TestSimilarEdges:
    def test_similar_edges_above_threshold(self):
        # Create two entries with high cosine similarity, different categories
        # (same category would create sibling edges first, blocking similar)
        entries = [
            SkillEntry(name="a", description="aaa", category="cat1"),
            SkillEntry(name="b", description="bbb", category="cat2"),
            SkillEntry(name="c", description="ccc", category="cat3"),
        ]
        # a and b have similar embeddings, c is different
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],   # ~1.0 cosine with a
            [0.0, 0.0, 1.0],     # orthogonal
        ], dtype=np.float32)

        graph = SkillGraph(entries, emb, similar_threshold=0.5)
        a_edges = graph.get_neighbors("a")
        similar_targets = {e.target for e in a_edges if e.edge_type == "similar"}
        assert "b" in similar_targets
        assert "c" not in similar_targets

    def test_no_similar_edges_below_threshold(self):
        entries = [
            SkillEntry(name="a", description="aaa", category="cat"),
            SkillEntry(name="b", description="bbb", category="cat"),
        ]
        emb = np.array([
            [1.0, 0.0],
            [0.0, 1.0],  # orthogonal
        ], dtype=np.float32)

        graph = SkillGraph(entries, emb, similar_threshold=0.5)
        similar_edges = [
            e for adj in graph.adjacency.values()
            for e in adj if e.edge_type == "similar"
        ]
        assert len(similar_edges) == 0


class TestSerialization:
    def test_save_and_load(self, tmp_path):
        entries = [
            SkillEntry(name="a", description="aaa", category="cat"),
            SkillEntry(name="b", description="bbb", category="cat"),
        ]
        emb = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        graph = SkillGraph(entries, emb, similar_threshold=0.5)

        save_path = tmp_path / "adj.json"
        graph.save(save_path)
        loaded = SkillGraph.load(save_path)

        assert loaded["num_nodes"] == 2
        assert loaded["num_edges"] > 0


class TestGraphStats:
    def test_node_and_edge_counts(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        assert graph.num_nodes == 9
        assert graph.num_edges > 0

    def test_edge_type_counts(self, tmp_skills_dir):
        from skillgraph.indexer import index_skills

        entries = index_skills(tmp_skills_dir)
        embeddings = np.random.randn(len(entries), 64).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        graph = SkillGraph(entries, embeddings, similar_threshold=0.99)
        counts = graph.edge_type_counts
        assert "related" in counts
        assert "sibling" in counts