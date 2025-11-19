# RAGGAE/core/reranker.py
from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass, replace

import numpy as np

from .retriever import Hit

try:
    # Two good defaults users may have:
    # 1) sentence-transformers CrossEncoder
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

@dataclass
class RerankConfig:
    """
    Configuration for Cross-Encoder style reranking.
    """
    model_name: str = "jinaai/jina-reranker-v1-base-multilingual"
    batch_size: int = 16

    def __str__(self) -> str:
        return f"RerankConfig(model={self.model_name}, batch={self.batch_size})"

class CrossEncoderReranker:
    """
    Thin wrapper around a cross-encoder that scores (query, passage) pairs.

    If `sentence-transformers` is not available, a helpful ImportError is raised.

    Example
    -------
    >>> from RAGGAE.core.reranker import CrossEncoderReranker
    >>> rr = CrossEncoderReranker()
    >>> hits_rr = rr.rerank("MLflow on K8s", hits)  # where hits come from HybridRetriever
    """
    def __init__(self, config: Optional[RerankConfig] = None):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers (CrossEncoder) not installed. Install via `mamba install -c conda-forge sentence-transformers`.")
        self.config = config or RerankConfig()
        self._model = CrossEncoder(self.config.model_name)

    def rerank(self, query: str, hits: List[Hit], top_k: Optional[int] = None) -> List[Hit]:
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs, batch_size=self.config.batch_size)
        scores = np.asarray(scores, dtype=np.float32)
        # Replace final score with reranker score, keep components
        new_hits = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
        new_hits.sort(key=lambda h: h.score, reverse=True)
        return new_hits[:top_k] if top_k else new_hits

    def __str__(self) -> str:
        return f"CrossEncoderReranker<{self.config}>"

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.config.model_name!r})"
