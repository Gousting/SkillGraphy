"""Tests for the embedder factory and backends."""

import pytest
from unittest.mock import patch, MagicMock

from skillgraph.embedder import (
    Embedder,
    OllamaEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class TestFactory:
    def test_create_ollama(self):
        emb = create_embedder("ollama")
        assert isinstance(emb, OllamaEmbedder)

    def test_create_local(self):
        emb = create_embedder("local")
        assert isinstance(emb, LocalEmbedder)

    def test_create_openai_no_key(self):
        import os
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError):
                create_embedder("openai")
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            create_embedder("invalid")

    def test_case_insensitive(self):
        emb = create_embedder("OLLAMA")
        assert isinstance(emb, OllamaEmbedder)


class TestOllamaEmbedder:
    def test_name(self):
        emb = OllamaEmbedder()
        assert "ollama" in emb.name

    def test_custom_url(self):
        emb = OllamaEmbedder(url="http://localhost:12345")
        assert "localhost:12345" in emb.url

    @patch("httpx.post")
    def test_embed_returns_vector(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        emb = OllamaEmbedder()
        vec = emb.embed("test text")
        assert len(vec) == 3
        assert vec == [0.1, 0.2, 0.3]

    @patch("httpx.post")
    def test_embed_batch(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        emb = OllamaEmbedder()
        vecs = emb.embed_batch(["text1", "text2"])
        assert len(vecs) == 2
        assert vecs[0] == [0.1, 0.2]
        assert vecs[1] == [0.3, 0.4]


class TestOpenAIEmbedder:
    def test_missing_key_raises(self):
        import os
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError):
                OpenAIEmbedder()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_name(self):
        emb = OpenAIEmbedder(api_key="sk-fake")
        assert "openai" in emb.name