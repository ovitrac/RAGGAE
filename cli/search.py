# RAGGAE/cli/search.py
"""
Search an existing FAISS index with a query and show provenance.

Command-line tool for semantic search over indexed documents with
detailed provenance display (page, block, score).

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Usage:
    python -m RAGGAE.cli.search --index tender.idx --model intfloat/multilingual-e5-small --e5 \
        --query "Plateforme MLOps avec MLflow sur Kubernetes"
"""
from __future__ import annotations
import argparse
import numpy as np

from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="prefix used in index_doc.py")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--e5", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
    idx = FaissIndex.load(args.index)

    qv = enc.embed_query(args.query).astype("float32")[None, :]
    D, I, recs = idx.search(qv, args.k)

    print(f"Top-{args.k} for: {args.query!r}\n")
    for score, rec in zip(D[0], recs[0]):
        meta = rec.metadata
        loc = f"(p.{meta.get('page','?')}, b{meta.get('block','?')})"
        snippet = rec.text.replace("\n", " ")[:140]
        print(f"• {score:.4f} {loc} {snippet}…")

if __name__ == "__main__":
    main()
