"""Utility helpers for embedding generation used across the RAG pipeline.

This module now provides a resilient embedding backend that prefers the
`sentence-transformers` model when it is available locally.  When the model
files cannot be resolved (for example in an offline environment) we gracefully
fall back to a deterministic hashing based vectorizer that runs entirely
offline.  The fallback keeps the embedding dimensionality identical to the
former model so that downstream components (guardrails, Chroma collections,
etc.) continue to operate without additional changes.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import HashingVectorizer

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled via fallback
    SentenceTransformer = None  # type: ignore[assignment]

DEFAULT_MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
FALLBACK_DIMENSION = 384

# Ensure the cache directory resolves to a writable location.  If the
# environment already specifies a cache directory we keep it untouched.
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface_cache")


class EmbeddingBackend:
    """Embedding generator with transparent fallback support.

    The backend first tries to instantiate the requested sentence-transformer
    locally.  If the model files cannot be found (or the library itself is not
    installed) a lightweight hashing vectorizer is used instead.  The hashing
    approach is deterministic which guarantees that embeddings for identical
    inputs remain stable across runs.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.requested_model = model_name
        self._model_name: str = model_name
        self._model: SentenceTransformer | None = None
        self._uses_sentence_transformer = False
        self._dimension = FALLBACK_DIMENSION
        self._hash_vectorizer = HashingVectorizer(
            n_features=FALLBACK_DIMENSION, alternate_sign=False, norm="l2"
        )
        self._initialise_model()

    # ------------------------------------------------------------------
    # Public attributes
    @property
    def embedding_name(self) -> str:
        """Human readable name of the active embedding strategy."""

        return self._model_name if self._uses_sentence_transformer else "hashing-vectorizer-384"

    # ------------------------------------------------------------------
    def uses_sentence_transformer(self) -> bool:
        """Return ``True`` when the real sentence-transformer is available."""

        return self._uses_sentence_transformer

    # ------------------------------------------------------------------
    def encode(self, documents: Sequence[str] | str) -> NDArray[np.float32]:
        """Return embeddings for the provided documents.

        Args:
            documents: A single string or an iterable of documents to embed.

        Returns:
            A ``(n_documents, embedding_dim)`` float32 numpy array.
        """

        if isinstance(documents, str):
            documents = [documents]

        texts: list[str] = list(documents)
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(texts)
            return np.asarray(embeddings, dtype=np.float32)

        hashed = self._hash_vectorizer.transform(texts)
        dense = hashed.astype(np.float32).toarray()
        return dense

    # ------------------------------------------------------------------
    def __call__(self, documents: Sequence[str] | str) -> list[list[float]]:
        """Make the backend compatible with Chroma embedding hooks."""

        return self.encode(documents).tolist()

    # ------------------------------------------------------------------
    def _initialise_model(self) -> None:
        """Attempt to load the requested sentence-transformer locally."""

        if SentenceTransformer is None:
            logging.warning(
                "sentence-transformers is not available; falling back to hashing vectorizer embeddings."
            )
            self._uses_sentence_transformer = False
            self._model_name = "hashing-vectorizer-384"
            return

        try:
            self._model = SentenceTransformer(self.requested_model)
            self._model_name = self.requested_model
            self._uses_sentence_transformer = True
            try:
                self._dimension = int(self._model.get_sentence_embedding_dimension())
            except AttributeError:  # pragma: no cover - older versions
                self._dimension = FALLBACK_DIMENSION
            logging.info("Loaded SentenceTransformer model '%s'", self.requested_model)
        except (OSError, RuntimeError, ValueError) as exc:
            logging.warning(
                "Unable to load SentenceTransformer model '%s' locally (%s). "
                "Falling back to hashing vectorizer embeddings.",
                self.requested_model,
                exc,
            )
            self._model = None
            self._uses_sentence_transformer = False
            self._model_name = "hashing-vectorizer-384"
            self._dimension = FALLBACK_DIMENSION


@lru_cache(maxsize=None)
def _get_cached_backend(model_name: str) -> EmbeddingBackend:
    return EmbeddingBackend(model_name=model_name)


def get_embedding_backend(model_name: str | None = None) -> EmbeddingBackend:
    """Return a cached :class:`EmbeddingBackend` instance.

    Using a cached backend avoids repeatedly instantiating the underlying model
    (either the sentence-transformer or the hashing vectorizer), which is
    important for the Streamlit runtime where modules may be reloaded.
    """

    target_model = model_name or DEFAULT_MODEL_NAME
    return _get_cached_backend(target_model)


def get_embedding(
    documents: Sequence[str] | str, model_name: str | None = None
) -> NDArray[np.float32]:
    """Generate embeddings for the provided ``documents``.

    Args:
        documents: Either a single string or an iterable of strings.
        model_name: Optionally override the model used for the embedding.

    Returns:
        A NumPy ``float32`` matrix containing one embedding per input document.
    """

    backend = get_embedding_backend(model_name)
    return backend.encode(documents)
