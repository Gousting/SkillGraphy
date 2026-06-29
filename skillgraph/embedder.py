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
import math
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


def _sanitize_vec(vec: list[float]) -> list[float]:
    """Replace NaN/Inf values with 0.0 (Ollama bge-m3 bug workaround)."""
    if any(math.isnan(v) or math.isinf(v) for v in vec):
        return [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in vec]
    return vec


# ── Ollama Backend ───────────────────────────────────────────────────────────


class OllamaEmbedder(Embedder):
    """Ollama-based embedding backend.

    Requires Ollama running on localhost:11434 with an embedding model pulled:
        ollama pull bge-m3
    """

    DEFAULT_MODEL = "bge-m3"
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
            return _sanitize_vec(embeddings[0])
        return _sanitize_vec(embeddings)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        # Batch in chunks of 4 to avoid Ollama timeout on large models (e.g. bge-m3)
        # Some texts may trigger Ollama 500 (NaN bug in bge-m3) — fall back to zero vectors
        all_vecs: list[list[float]] = []
        batch_size = 4
        dim = 1024  # bge-m3 default; corrected after first success
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                resp = httpx.post(
                    f"{self.url}/api/embed",
                    json={"model": self.model, "input": chunk},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings") or data.get("embedding") or []
                if embeddings and isinstance(embeddings[0], list):
                    for vec in embeddings:
                        all_vecs.append(_sanitize_vec(vec))
                    dim = len(embeddings[0])
                else:
                    all_vecs.append(embeddings if embeddings else [0.0] * dim)
            except Exception:
                # Batch failed — try embeddings one by one, use zero vector for failures
                for text in chunk:
                    try:
                        resp2 = httpx.post(
                            f"{self.url}/api/embed",
                            json={"model": self.model, "input": text},
                            timeout=60.0,
                        )
                        if resp2.status_code == 200:
                            data2 = resp2.json()
                            embs = data2.get("embeddings") or data2.get("embedding") or []
                            vec = embs[0] if embs and isinstance(embs[0], list) else embs
                            if vec:
                                vec = _sanitize_vec(vec)
                                dim = len(vec)
                                all_vecs.append(vec)
                                continue
                        all_vecs.append([0.0] * dim)
                    except Exception:
                        all_vecs.append([0.0] * dim)
        return all_vecs


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