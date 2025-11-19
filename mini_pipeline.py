#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:41:29 2025

@author: olivi
"""

# pip install sentence-transformers faiss-cpu rank-bm25 pypdf pymupdf tqdm
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from rank_bm25 import BM25Okapi
import fitz  # PyMuPDF

# 1) Parse & chunk
def parse_pdf(path):
    doc = fitz.open(path)
    chunks = []
    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("blocks")  # retains block order
        for i, (_, _, _, _, t, _, _) in enumerate(text):
            t = (t or "").strip()
            if len(t) > 40:
                chunks.append({"text": t, "page": pno+1, "block": i})
    return chunks

chunks = parse_pdf("examples/SNCF Réseau_PACS_Accord Cadre nova.pdf")
texts = [c["text"] for c in chunks]

# 2) Dense embeddings
model = SentenceTransformer("intfloat/multilingual-e5-small")
# e5 expects "query: ..." vs "passage: ..." prefixes for best results
passages = [f"passage: {t}" for t in texts]
E = np.vstack(model.encode(passages, normalize_embeddings=True))

# 3) FAISS index
index = faiss.IndexFlatIP(E.shape[1])
index.add(E.astype("float32"))

# 4) BM25
bm25 = BM25Okapi([t.split() for t in texts])

# 5) Hybrid search
def hybrid_search(q, k_dense=100, k=20, alpha=0.6):
    q_dense = model.encode([f"query: {q}"], normalize_embeddings=True)
    D, I = index.search(q_dense.astype("float32"), k_dense)
    dense_scores = {i: float(s) for i, s in zip(I[0], D[0])}
    # BM25 scores
    bm = bm25.get_scores(q.split())
    # Normalize BM25
    bm = (bm - bm.min()) / (bm.ptp() + 1e-9)
    # Fuse
    fused = []
    for i, ds in dense_scores.items():
        fs = alpha*ds + (1-alpha)*float(bm[i])
        fused.append((i, fs))
    fused.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] | {"score": s} for i, s in fused[:k]]

hits = hybrid_search("PCA, PRA, gestion de crise, certification, environnement, cloud")
for h in hits[:5]:
    print(h["score"], h["page"], h["text"][:120], "…")
