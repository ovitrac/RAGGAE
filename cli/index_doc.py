# RAGGAE/cli/index_doc.py
"""
Index a PDF into a FAISS store using a chosen embedding model.

Command-line tool for building dense vector indices from PDF documents
with configurable embedding models and E5 prefix support.

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Usage:
    python -m RAGGAE.cli.index_doc --pdf tender.pdf --out tender.idx --model intfloat/multilingual-e5-small --e5
"""
from __future__ import annotations
import argparse
import numpy as np

from RAGGAE.io.pdf import extract_blocks, to_texts_and_meta
from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True, help="prefix path for index: writes .faiss and .jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--e5", action="store_true", help="Use E5 prefixes")
    args = ap.parse_args()

    blocks = extract_blocks(args.pdf)
    texts, metas = to_texts_and_meta(blocks)

    enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
    vecs = enc.embed_texts(texts).astype("float32")

    idx = FaissIndex(dim=vecs.shape[1])
    idx.add(vecs, texts, metas)
    idx.save(args.out)

    print(f"Indexed {len(texts)} chunks → {args.out}.faiss + {args.out}.jsonl")
    print(enc.info)

if __name__ == "__main__":
    main()
