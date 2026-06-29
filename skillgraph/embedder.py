"""
Embedding backends for SkillGraph.

Three backends supported, in priority order:
  1. `ollama`  — local Ollama API (default, zero-cost, offline-capable)
  2. `local`   — sentence-transformers (all-MiniLM-L6-v2, fully offline)
  3. `openai`  — OpenAI text-embedding-3-small (API key required)

All backends share a common interface: embed(text) -> list[float].
Batch embedding is supported via embed_batch(texts) -> list[list[float]].
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


# ── Base Protocol ───────────────────────────────────────────────────────────


class Embedder(ABC):
    """Abstract base class for embedding backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default: sequential embed()."""
        return [self.embed(t) for t in texts]


# ── Ollama Backend ───────────────────────────────────────────────────────────


class OllamaEmbedder(Embedder):
    """Ollama-based embedding backend.

    Requires Ollama running on localhost:11434 with an embedding model pulled:
        ollama pull nomic-embed-text
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str | None = None,
        url: str | None = None,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.url = (url or os.environ.get("OLLAMA_BASE_URL") or self.DEFAULT_URL).rstrip("/")
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"ollama({self.model})"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def embed(self, text: str) -> list[float]:
        import httpx

        resp = httpx.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding") or []
        if isinstance(embeddings[0] if embeddings else None, list):
            return embeddings[0]
        return embeddings

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        resp = httpx.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding") or []
        # Ollama returns a list of vectors for batch input
        if embeddings and isinstance(embeddings[0], list):
            return embeddings
        # Single vector returned — replicate for each text
        return [embeddings] * len(texts) if embeddings else []


# ── sentence-transformers Backend ───────────────────────────────────────────


class LocalEmbedder(Embedder):
    """Local sentence-transformers embedding backend (fully offline).

    Requires: pip install "skillgraph[local]"
    Default model: all-MiniLM-L6-v2 (90MB, 384-dim, runs on CPU).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or self.DEFAULT_MODEL
        self._model: Any = None
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"local({self.model_name})"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    'Run: pip install "skillgraph[local]"'
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


# ── OpenAI Backend ───────────────────────────────────────────────────────────


class OpenAIEmbedder(Embedder):
    """OpenAI embedding backend.

    Requires: pip install "skillgraph[openai]"
    Requires OPENAI_API_KEY environment variable or --api-key argument.
    """

    DEFAULT_MODEL = "text-embedding-3-small"  # 1536-dim, $0.02/1M tokens

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required: set OPENAI_API_KEY or pass --api-key"
            )
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"openai({self.model})"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                'openai not installed. Run: pip install "skillgraph[openai]"'
            ) from e
        return OpenAI(api_key=self._api_key)

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


# ── Factory ──────────────────────────────────────────────────────────────────


def create_embedder(
    backend: str = "ollama",
    *,
    model: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
) -> Embedder:
    """Create an embedder backend by name.

    Args:
        backend: "ollama" | "local" | "openai"
        model: Override default model for the backend
        url: Override Ollama URL (ollama only)
        api_key: OpenAI API key (openai only, or set OPENAI_API_KEY env var)

    Returns:
        An Embedder instance.
    """
    backend = backend.lower().strip()

    if backend == "ollama":
        return OllamaEmbedder(model=model, url=url)
    elif backend == "local":
        return LocalEmbedder(model=model)
    elif backend == "openai":
        return OpenAIEmbedder(model=model, api_key=api_key)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Choose from: ollama, local, openai"
        )