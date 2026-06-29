"""Tests for the retriever."""

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from skillgraph.retriever import Retriever
from skillgraph.indexer import SkillEntry
from skillgraph.embedder import Embedder


class TestRetriever:
    def test_build_with_mock_embedder(self, tmp_skills_dir, mock_embedder):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        count = retriever.build()
        assert count == 9
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == (9, 64)
        assert retriever.graph is not None

    def test_retrieve_returns_results(self, tmp_skills_dir, mock_embedder):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever.build()
        results = retriever.retrieve("generate a diagram", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SkillEntry) for r in results)

    def test_retrieve_scores_are_sorted(self, tmp_skills_dir, mock_embedder):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever.build()
        results = retriever.retrieve("test query", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_without_graph(self, tmp_skills_dir, mock_embedder):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever.build()
        results = retriever.retrieve("test", top_k=3, expand_graph=False)
        assert len(results) <= 3
        # All should be seed source (no graph expansion)
        assert all(r.source == "seed" for r in results)

    def test_retrieve_empty_index(self, mock_embedder):
        retriever = Retriever(
            skills_dir="/nonexistent",
            embedder=mock_embedder,
        )
        retriever.build()
        results = retriever.retrieve("anything")
        assert results == []

    def test_save_and_load(self, tmp_skills_dir, mock_embedder, tmp_path):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever.build()
        retriever.save(tmp_path / "index")

        # Create new retriever and load
        retriever2 = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever2.load(tmp_path / "index")
        assert len(retriever2.entries) == 9
        assert retriever2.embeddings is not None
        assert retriever2.embeddings.shape == (9, 64)

    def test_stats(self, tmp_skills_dir, mock_embedder):
        retriever = Retriever(
            skills_dir=tmp_skills_dir,
            embedder=mock_embedder,
        )
        retriever.build()
        stats = retriever.stats
        assert stats["total_skills"] == 9
        assert "embedding_dim" in stats
        assert "graph" in stats
        assert stats["graph"]["nodes"] == 9