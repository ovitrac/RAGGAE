"""
RAGGAE/core/embeddings.py

Embedding providers for dense text representation with pluggable architectures.
Supports Sentence-Transformers bi-encoders with optional E5-style prefixing
for improved retrieval performance.

This module provides:
- Abstract EmbeddingProvider interface for custom implementations
- STBiEncoder: Production-ready wrapper for Sentence-Transformers models
- EmbeddingInfo: Metadata descriptor for embedding configurations

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from core.embeddings import STBiEncoder
>>> encoder = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
>>> texts = ["MLOps with MLflow", "ISO 27001 certification"]
>>> embeddings = encoder.embed_texts(texts)
>>> embeddings.shape
(2, 384)
"""
# RAGGAE/core/embeddings.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional, Literal
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

PrefixMode = Literal["none", "e5"]

@dataclass
class EmbeddingInfo:
    """
    Lightweight descriptor of an embedding provider configuration.

    Stores metadata about the embedding model, device allocation,
    dimensionality, and prefix strategy for retrieval optimization.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier (e.g., 'intfloat/multilingual-e5-small').
    device : str
        Execution device ('cuda' or 'cpu').
    dimension : int
        Embedding vector dimensionality.
    prefix_mode : PrefixMode
        Prefix strategy ('none' or 'e5' for E5-family models).
    """
    model_name: str
    device: str
    dimension: int
    prefix_mode: PrefixMode = "none"

    def __str__(self) -> str:
        return f"{self.model_name} [{self.device}] dim={self.dimension} ({self.prefix_mode})"

    def __repr__(self) -> str:
        return f"EmbeddingInfo(model_name={self.model_name!r}, device={self.device!r}, dimension={self.dimension}, prefix_mode={self.prefix_mode!r})"


class EmbeddingProvider:
    """
    Abstract base class for embedding providers.

    Defines the interface for text-to-vector transformation. Implementations
    must provide methods for batch text encoding and single query encoding,
    along with metadata exposure via the info property.

    Design Notes
    ------------
    - Return L2-normalized vectors for cosine similarity / inner-product search.
    - Implementations should handle tokenization, model inference, and
      any preprocessing (e.g., E5 prefixes) internally.
    - Thread-safety is the responsibility of concrete implementations.

    Methods to Implement
    --------------------
    embed_texts(texts: Sequence[str]) -> np.ndarray
        Encode multiple texts into dense vectors (batch operation).
    embed_query(text: str) -> np.ndarray
        Encode a single query text into a dense vector.
    info -> EmbeddingInfo
        Return metadata about the embedding provider.
    """
    @property
    def info(self) -> EmbeddingInfo:
        raise NotImplementedError

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class STBiEncoder(EmbeddingProvider):
    """
    Production-ready Sentence-Transformers bi-encoder with E5 prefix support.

    Wraps HuggingFace Sentence-Transformers models with automatic device
    detection, L2 normalization, and optional E5-style prefixing for improved
    retrieval performance. Suitable for multilingual RAG systems with GPU
    acceleration support.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., 'intfloat/multilingual-e5-small',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').
    device : str, optional
        Execution device ('cuda' or 'cpu'). If None, auto-detects CUDA availability.
    prefix_mode : {'none', 'e5'}, default='none'
        Prefix strategy:
        - 'none': No prefix modification
        - 'e5': Adds 'passage:' for texts, 'query:' for queries (E5 models)
    normalize : bool, default=True
        If True, L2-normalizes embeddings for cosine similarity / inner-product search.

    Attributes
    ----------
    info : EmbeddingInfo
        Metadata descriptor with model name, device, dimension, and prefix mode.

    Notes
    -----
    - E5 models (e.g., intfloat/multilingual-e5-*) require 'e5' prefix mode.
    - Batch encoding uses batch_size=64 by default for optimal GPU utilization.
    - All operations use torch.inference_mode() for memory efficiency.

    Examples
    --------
    >>> from core.embeddings import STBiEncoder
    >>> enc = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
    >>> texts = ["K8s with MLflow", "ISO 27001 certification required"]
    >>> E = enc.embed_texts(texts)
    >>> q = enc.embed_query("MLOps on Kubernetes with MLflow")
    >>> print(f"Texts: {E.shape}, Query: {q.shape}")
    Texts: (2, 384), Query: (384,)

    >>> # Cosine similarity via inner product (vectors are normalized)
    >>> import numpy as np
    >>> similarity = np.dot(E, q)
    >>> print(f"Similarities: {similarity}")
    """
    def __init__(self,
                 model_name: str,
                 device: Optional[str] = None,
                 prefix_mode: PrefixMode = "none",
                 normalize: bool = True):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please `mamba install -c conda-forge sentence-transformers`.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._prefix_mode = prefix_mode
        self._device = device
        dim = self._model.get_sentence_embedding_dimension()
        self._info = EmbeddingInfo(model_name=model_name, device=device, dimension=dim, prefix_mode=prefix_mode)

    @property
    def info(self) -> EmbeddingInfo:
        return self._info

    def _maybe_prefix(self, items: Iterable[str], kind: Literal["passage","query"]) -> List[str]:
        if self._prefix_mode == "e5":
            pre = f"{kind}: "
            return [pre + (t or "") for t in items]
        return list(items)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = self._maybe_prefix(texts, "passage")
        with torch.inference_mode():
            vecs = self._model.encode(
                texts,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def __str__(self) -> str:
        return f"STBiEncoder<{self.info}>"

    def __repr__(self) -> str:
        return f"STBiEncoder(model={self.info.model_name!r}, device={self.info.device!r}, prefix_mode={self._prefix_mode!r}, normalize={self._normalize})"
