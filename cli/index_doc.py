# RAGGAE/cli/index_doc.py
"""
Index documents into a FAISS store using a chosen embedding model.

Command-line tool for building dense vector indices from documents
with configurable embedding models and E5 prefix support.

Supports multiple formats:
- PDF, DOCX, PPTX, XLSX, ODT, ODP, ODS
- TXT, MD, CSV, JSON, XML, HTML
- Code files (py, java, js, ts, etc.)

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: December 17, 2025
License: MIT

Usage:
    # Single PDF
    python -m RAGGAE.cli.index_doc --input tender.pdf --out tender --e5

    # Multiple files
    python -m RAGGAE.cli.index_doc --input doc1.pdf --input doc2.docx --out combined --e5

    # Directory (all supported files)
    python -m RAGGAE.cli.index_doc --input ./documents/ --out corpus --e5

    # JSON output for automation
    python -m RAGGAE.cli.index_doc --input tender.pdf --out tender --e5 --format json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex
from RAGGAE.io.ingest import ingest_any, get_supported_extensions


def collect_files(inputs: List[str], recursive: bool = True) -> List[Path]:
    """Collect files from input paths (files or directories)."""
    supported = get_supported_extensions()
    files = []

    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            if p.suffix.lower().lstrip(".") in supported:
                files.append(p)
            else:
                print(f"Warning: Skipping unsupported file: {p}", file=sys.stderr)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for f in p.glob(pattern):
                if f.is_file() and f.suffix.lower().lstrip(".") in supported:
                    files.append(f)
        else:
            print(f"Warning: Path not found: {p}", file=sys.stderr)

    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(
        description="Index documents into a FAISS vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported formats: {', '.join(sorted(get_supported_extensions()))}

Examples:
  # Single PDF
  python -m RAGGAE.cli.index_doc --input tender.pdf --out tender --e5

  # Multiple files
  python -m RAGGAE.cli.index_doc --input doc1.pdf --input doc2.docx --out combined --e5

  # Directory (recursive)
  python -m RAGGAE.cli.index_doc --input ./documents/ --out corpus --e5

  # JSON output
  python -m RAGGAE.cli.index_doc --input tender.pdf --out tender --e5 --format json
        """
    )

    ap.add_argument("--input", "-i", action="append", required=True,
                    help="Input file or directory (repeat for multiple)")
    ap.add_argument("--out", "-o", required=True,
                    help="Output index prefix (creates .faiss and .jsonl)")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small",
                    help="Embedding model (default: intfloat/multilingual-e5-small)")
    ap.add_argument("--e5", action="store_true", help="Use E5-style prefixes")
    ap.add_argument("--min-chars", type=int, default=40,
                    help="Minimum characters per chunk (default: 40)")
    ap.add_argument("--no-recursive", action="store_true",
                    help="Don't search directories recursively")
    ap.add_argument("--format", choices=["text", "json"], default="text",
                    help="Output format (default: text)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show detailed progress")

    args = ap.parse_args()

    # Collect files
    files = collect_files(args.input, recursive=not args.no_recursive)
    if not files:
        print("Error: No supported files found", file=sys.stderr)
        sys.exit(1)

    if args.format == "text":
        print(f"Found {len(files)} file(s) to index")

    # Ingest documents
    all_texts = []
    all_metas = []
    file_stats = []

    for f in files:
        try:
            blocks = ingest_any(f, min_chars=args.min_chars)
            texts = [b.text for b in blocks]
            metas = [{"file": str(f.name), "page": b.page, "block": b.block_id, "ext": f.suffix.lower()} for b in blocks]

            all_texts.extend(texts)
            all_metas.extend(metas)

            file_stats.append({
                "file": str(f),
                "chunks": len(texts),
                "status": "ok"
            })

            if args.verbose and args.format == "text":
                print(f"  {f.name}: {len(texts)} chunks")

        except Exception as e:
            file_stats.append({
                "file": str(f),
                "chunks": 0,
                "status": f"error: {e}"
            })
            if args.format == "text":
                print(f"Warning: Failed to process {f}: {e}", file=sys.stderr)

    if not all_texts:
        print("Error: No text extracted from files", file=sys.stderr)
        sys.exit(1)

    # Create embeddings
    if args.format == "text":
        print(f"Creating embeddings for {len(all_texts)} chunks...")

    enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
    vecs = enc.embed_texts(all_texts).astype("float32")

    # Build and save index
    idx = FaissIndex(dim=vecs.shape[1])
    idx.add(vecs, all_texts, all_metas)
    idx.save(args.out)

    # Output results
    result = {
        "index": args.out,
        "model": args.model,
        "e5": args.e5,
        "files_processed": len([s for s in file_stats if s["status"] == "ok"]),
        "files_failed": len([s for s in file_stats if s["status"] != "ok"]),
        "total_chunks": len(all_texts),
        "dimension": vecs.shape[1],
        "files": file_stats if args.verbose else None
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"\nIndexed {len(all_texts)} chunks from {result['files_processed']} file(s)")
        print(f"  → {args.out}.faiss")
        print(f"  → {args.out}.jsonl")
        print(f"Model: {args.model} (dim={vecs.shape[1]})")
        if result['files_failed'] > 0:
            print(f"Warning: {result['files_failed']} file(s) failed to process")


if __name__ == "__main__":
    main()
