"""
RAGGAE/core/index_faiss.py

FAISS-based vector index with metadata sidecar for document provenance tracking.
Provides minimal wrapper around FAISS IndexFlatIP for inner-product (cosine)
similarity search with persistent storage of texts and metadata.

This module provides:
- FaissIndex: Vector store with synchronized text/metadata storage
- Record: Dataclass for indexed documents with provenance
- Persistence: Save/load to disk (.faiss + .jsonl)

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from core.index_faiss import FaissIndex
>>> import numpy as np
>>> # Create index
>>> idx = FaissIndex(dim=384)
>>> vecs = np.random.rand(10, 384).astype('float32')
>>> texts = [f"doc {i}" for i in range(10)]
>>> idx.add(vecs, texts)
>>> # Search
>>> query = np.random.rand(1, 384).astype('float32')
>>> scores, ids, records = idx.search(query, k=5)
>>> # Persist
>>> idx.save("./my_index")  # Creates my_index.faiss + my_index.jsonl
>>> idx2 = FaissIndex.load("./my_index")
"""
# RAGGAE/core/index_faiss.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
import faiss
from pathlib import Path

@dataclass
class Record:
    """
    Indexed document record with text content and metadata.

    Attributes
    ----------
    id : int
        Unique identifier within the index.
    text : str
        Full text content of the indexed chunk.
    metadata : Dict[str, Any]
        Provenance metadata (e.g., page, block, bbox, file).
    """
    id: int
    text: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return f"Record(id={self.id}, text={self.text[:50]!r}..., meta_keys={list(self.metadata)})"

class FaissIndex:
    """
    FAISS inner-product index with synchronized text/metadata storage.

    Wraps FAISS IndexFlatIP for exact cosine similarity search (assuming
    L2-normalized vectors). Maintains a sidecar list of Record objects
    for provenance tracking. Supports persistence to disk via dual files:
    .faiss (binary index) and .jsonl (text + metadata).

    Design Notes
    ------------
    - Uses inner product (IP) metric: assumes input vectors are L2-normalized.
    - Exact search (no approximation): suitable for <100K documents.
    - For larger collections, consider IndexIVFFlat or IndexHNSWFlat.
    - Thread-safe for read operations; add() should be externally synchronized.

    Parameters
    ----------
    dim : int
        Embedding dimensionality (must match vectors added).

    Attributes
    ----------
    dim : int
        Vector dimension.
    _index : faiss.IndexFlatIP
        Underlying FAISS index.
    _records : List[Record]
        Synchronized list of indexed documents.

    Examples
    --------
    >>> from core.index_faiss import FaissIndex
    >>> import numpy as np
    >>> idx = FaissIndex(dim=384)
    >>> vecs = np.random.rand(100, 384).astype('float32')
    >>> texts = [f"Document {i}" for i in range(100)]
    >>> metas = [{"page": i % 10 + 1} for i in range(100)]
    >>> idx.add(vecs, texts, metas)
    >>> query = np.random.rand(1, 384).astype('float32')
    >>> scores, ids, records = idx.search(query, k=5)
    >>> print(f"Top result: {records[0][0].text[:50]}")
    >>> idx.save("./my_index")
    >>> loaded = FaissIndex.load("./my_index")
    >>> assert len(loaded) == 100

    Notes
    -----
    - Expects embeddings to be float32 and L2-normalized for cosine similarity.
    - Persistence creates two files: <path>.faiss (binary) and <path>.jsonl (metadata).
    - Load restores both index and records in original order.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._records: List[Record] = []
        self._next_id = 0

    def add(self, vectors: np.ndarray, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> List[int]:
        assert vectors.dtype == np.float32, "vectors must be float32"
        assert vectors.shape[0] == len(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        assert len(metadatas) == len(texts)

        ids = list(range(self._next_id, self._next_id + len(texts)))
        self._next_id += len(texts)

        for i, t, m in zip(ids, texts, metadatas):
            self._records.append(Record(id=i, text=t, metadata=m))
        self._index.add(vectors)
        return ids

    def search(self, query_vecs: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, List[List[Record]]]:
        """
        Returns (scores, ids, records_per_query)
        """
        assert query_vecs.dtype == np.float32
        D, I = self._index.search(query_vecs, k)
        all_records: List[List[Record]] = []
        for row in I:
            all_records.append([self._records[i] for i in row if i != -1])
        return D, I, all_records

    # ---- Persistence ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps({"id": r.id, "text": r.text, "metadata": r.metadata}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "FaissIndex":
        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        dim = index.d
        inst = cls(dim)
        inst._index = index
        inst._records.clear()
        with open(path.with_suffix(".jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                inst._records.append(Record(id=int(obj["id"]), text=obj["text"], metadata=obj.get("metadata", {})))
        inst._next_id = (max([r.id for r in inst._records]) + 1) if inst._records else 0
        return inst

    def __len__(self) -> int:
        return len(self._records)

    def __str__(self) -> str:
        return f"FaissIndex(dim={self.dim}, ntotal={len(self)}, type=IP)"

    def __repr__(self) -> str:
        return f"FaissIndex(dim={self.dim}, records={len(self)})"
