"""
RAGGAE/core/retriever.py

Hybrid retrieval combining dense (FAISS) and sparse (BM25) search with
linear score fusion. Optimized for multilingual document retrieval with
provenance tracking.

This module provides:
- HybridRetriever: Dense + sparse fusion with configurable weights
- Hit: Dataclass for search results with decomposed scores
- Build patterns: Factory methods for easy index construction

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from core.embeddings import STBiEncoder
>>> from core.retriever import HybridRetriever
>>> encoder = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
>>> texts = ["MLOps with MLflow on K8s", "ISO 27001 certification required"]
>>> retriever = HybridRetriever.build(encoder, texts)
>>> hits = retriever.search("MLflow on Kubernetes", k=10, alpha=0.6)
>>> for h in hits:
...     print(f"{h.score:.4f}: {h.text[:50]}")
"""
# RAGGAE/core/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingProvider
from .index_faiss import FaissIndex, Record

@dataclass
class Hit:
    """
    Search result with hybrid scoring breakdown.

    Attributes
    ----------
    id : int
        Document ID in the index.
    score : float
        Final fused score (α·dense + (1-α)·sparse).
    text : str
        Full text content of the hit.
    metadata : Dict[str, Any]
        Provenance metadata (page, block, file, etc.).
    dense : float
        Dense (semantic) similarity score.
    sparse : float
        Sparse (BM25) score (min-max normalized).
    """
    id: int
    score: float
    text: str
    metadata: Dict[str, Any]
    dense: float
    sparse: float

    def __str__(self) -> str:
        loc = ""
        if "page" in self.metadata and "block" in self.metadata:
            loc = f" (p.{self.metadata['page']},b{self.metadata['block']})"
        return f"Hit[{self.id}] score={self.score:.4f} dense={self.dense:.4f} sparse={self.sparse:.4f}{loc}: {self.text[:80]!r}"

class HybridRetriever:
    """
    Production-grade hybrid retriever fusing dense and sparse retrieval.

    Combines FAISS inner-product search (semantic similarity) with BM25
    term matching (lexical similarity) using configurable linear fusion.
    Optimized for multilingual document retrieval with provenance tracking.

    Pipeline
    --------
    1. Build Phase:
       - Encode texts → FAISS index (dense vectors)
       - Tokenize texts → BM25 index (term frequencies)
    2. Query Phase:
       - Encode query → FAISS search (top-k_dense)
       - Tokenize query → BM25 scores (all docs)
       - Fuse: score = α·dense + (1-α)·sparse (min-max normalized)
       - Rank and return top-k

    Parameters
    ----------
    encoder : EmbeddingProvider
        Text embedding provider (e.g., STBiEncoder).
    index : FaissIndex
        Dense vector index.
    bm25 : BM25Okapi
        Sparse term-based retriever.
    texts : List[str]
        Original texts (for provenance).
    metadatas : List[Dict[str, Any]]
        Metadata for each text (page, block, file, etc.).

    Examples
    --------
    >>> from core.embeddings import STBiEncoder
    >>> from core.retriever import HybridRetriever
    >>> encoder = STBiEncoder("intfloat/multilingual-e5-small", prefix_mode="e5")
    >>> texts = ["MLOps with MLflow on K8s", "ISO 27001 certification required"]
    >>> metas = [{"page": 1, "block": 0}, {"page": 2, "block": 1}]
    >>> retriever = HybridRetriever.build(encoder, texts, metadatas=metas)
    >>> hits = retriever.search("MLflow on Kubernetes", k=5, alpha=0.6)
    >>> for h in hits:
    ...     print(f"{h.score:.4f}: {h.text[:50]}")

    Notes
    -----
    - Default α=0.6 favors dense retrieval (semantic > lexical).
    - BM25 scores are min-max normalized to [0,1] before fusion.
    - For best results with E5 models, use prefix_mode='e5' in encoder.
    """
    def __init__(self, encoder: EmbeddingProvider, index: FaissIndex, bm25: BM25Okapi, texts: List[str], metadatas: List[Dict[str, Any]]):
        self.encoder = encoder
        self.index = index
        self._bm25 = bm25
        self._texts = texts
        self._metas = metadatas

    @classmethod
    def build(cls,
              encoder: EmbeddingProvider,
              texts: Sequence[str],
              metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> "HybridRetriever":
        texts = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts]
        metadatas = list(metadatas)

        # Dense index
        vecs = encoder.embed_texts(texts).astype("float32")
        index = FaissIndex(dim=vecs.shape[1])
        index.add(vecs, texts, list(metadatas))

        # BM25
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)

        return cls(encoder, index, bm25, texts, list(metadatas))

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-9)

    def search(self, query: str, k_dense: int = 100, k: int = 20, alpha: float = 0.6) -> List[Hit]:
        qv = self.encoder.embed_query(query).astype("float32")[None, :]
        D, I, _ = self.index.search(qv, min(k_dense, len(self._texts)))
        dense_map = {int(i): float(s) for i, s in zip(I[0], D[0])}

        # BM25 for all documents, normalize
        bm = self._bm25.get_scores(query.split())
        bm = self._minmax(bm)

        fused: List[Hit] = []
        for i, ds in dense_map.items():
            ss = float(bm[i])
            fs = alpha * ds + (1 - alpha) * ss
            fused.append(Hit(
                id=i,
                score=fs,
                dense=ds,
                sparse=ss,
                text=self._texts[i],
                metadata=self._metas[i]
            ))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused[:k]

    def __len__(self) -> int:
        return len(self._texts)

    def __str__(self) -> str:
        return f"HybridRetriever(n={len(self)}, encoder={self.encoder.info})"

    def __repr__(self) -> str:
        return f"HybridRetriever(n={len(self)}, encoder={self.encoder!r})"
