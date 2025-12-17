# RAGGAE/cli/search.py
"""
Search an existing FAISS index with a query and show provenance.

Command-line tool for semantic search over indexed documents with
detailed provenance display (page, block, score).

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: December 17, 2025
License: MIT

Usage:
    python -m RAGGAE.cli.search --index tender --e5 \\
        --query "Plateforme MLOps avec MLflow sur Kubernetes"

    # JSON output for automation
    python -m RAGGAE.cli.search --index tender --e5 --format json \\
        --query "ISO 27001 certification"
"""
from __future__ import annotations
import argparse
import json
import sys

from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex


def main():
    ap = argparse.ArgumentParser(
        description="Semantic search over indexed documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python -m RAGGAE.cli.search --index tender --e5 --query "ISO 27001"

  # JSON output for automation
  python -m RAGGAE.cli.search --index tender --e5 --format json --query "MLflow"

  # Multiple results
  python -m RAGGAE.cli.search --index tender --e5 --k 20 --query "Kubernetes"
        """
    )

    ap.add_argument("--index", required=True, help="Path to FAISS index (prefix)")
    ap.add_argument("--query", required=True, help="Search query")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small",
                    help="Embedding model (default: intfloat/multilingual-e5-small)")
    ap.add_argument("--e5", action="store_true", help="Use E5-style prefixes")
    ap.add_argument("--k", type=int, default=10, help="Number of results (default: 10)")
    ap.add_argument("--format", choices=["text", "json"], default="text",
                    help="Output format (default: text)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show full text instead of snippets")

    args = ap.parse_args()

    # Load index and encoder
    try:
        enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
        idx = FaissIndex.load(args.index)
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)

    # Search
    qv = enc.embed_query(args.query).astype("float32")[None, :]
    D, I, recs = idx.search(qv, args.k)

    # Build results
    results = []
    for rank, (score, rec) in enumerate(zip(D[0], recs[0]), 1):
        meta = rec.metadata
        results.append({
            "rank": rank,
            "score": float(score),
            "text": rec.text if args.verbose else rec.text[:200],
            "file": meta.get("file", ""),
            "page": meta.get("page", 0),
            "block": meta.get("block", 0),
            "ext": meta.get("ext", "")
        })

    # Output
    if args.format == "json":
        output = {
            "query": args.query,
            "index": args.index,
            "model": args.model,
            "k": args.k,
            "count": len(results),
            "results": results
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"\nTop-{args.k} results for: {args.query!r}")
        print(f"Index: {args.index}")
        print("-" * 70)
        for r in results:
            loc = f"(p.{r['page']}, b{r['block']})"
            snippet = r["text"].replace("\n", " ")
            if not args.verbose and len(snippet) > 140:
                snippet = snippet[:140] + "..."
            file_info = f" [{r['file']}]" if r['file'] else ""
            print(f"{r['rank']:2}. [{r['score']:.4f}] {loc}{file_info}")
            print(f"    {snippet}")
            print()


if __name__ == "__main__":
    main()
