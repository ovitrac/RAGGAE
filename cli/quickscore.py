# RAGGAE/cli/quickscore.py
"""
Quick NLI-based fit score over requirements.

Command-line tool for requirement-based compliance scoring using NLI models.
Supports both local Ollama (default, sovereign) and Claude API backends.

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: December 17, 2025
License: MIT

Usage (Ollama - default, sovereign):
    python -m RAGGAE.cli.quickscore --index tender --e5 \\
        --req "Provider must be ISO 27001 certified" \\
        --req "Deployments on Kubernetes with GitOps"

Usage (Claude API):
    python -m RAGGAE.cli.quickscore --index tender --e5 \\
        --backend claude --claude-model claude-sonnet-4-20250514 \\
        --req "Provider must be ISO 27001 certified"

Output formats:
    --format text   Human-readable (default)
    --format json   JSON for automation
    --format csv    CSV for spreadsheets
"""
from __future__ import annotations
import argparse
import json
import sys

from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex
from RAGGAE.core.nli_ollama import NLIClient, NLIConfig
from RAGGAE.core.nli_claude import ClaudeNLIClient, ClaudeNLIConfig, load_api_key
from RAGGAE.core.scoring import FitScorer, RequirementVerdict


def create_nli_client(args):
    """Create NLI client based on command-line arguments."""
    if args.backend == "claude":
        # Resolve API key: explicit > env > config file
        api_key = load_api_key(args.api_key)
        if not api_key:
            print("Error: Anthropic API key not found.", file=sys.stderr)
            print("Provide via --api-key, ANTHROPIC_API_KEY env var, or ~/.config/raggae/config.json", file=sys.stderr)
            sys.exit(1)

        return ClaudeNLIClient(
            api_key=api_key,
            config=ClaudeNLIConfig(
                model=args.claude_model,
                temperature=0.0,
                lang=args.nli_lang
            )
        )
    else:
        # Default: Ollama (local, sovereign)
        return NLIClient(NLIConfig(
            model=args.ollama_model,
            temperature=0.0,
            lang=args.nli_lang
        ))


def main():
    ap = argparse.ArgumentParser(
        description="NLI-based compliance scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama (local, default):
  python -m RAGGAE.cli.quickscore --index tender --e5 --req "ISO 27001 certified"

  # Claude API:
  python -m RAGGAE.cli.quickscore --index tender --e5 --backend claude --req "ISO 27001"

  # JSON output for automation:
  python -m RAGGAE.cli.quickscore --index tender --e5 --format json --req "ISO 27001"
        """
    )

    # Required arguments
    ap.add_argument("--index", required=True, help="Path to FAISS index (prefix)")
    ap.add_argument("--req", action="append", required=True,
                    help="Requirement string (repeat for multiple)")

    # Embedding model
    ap.add_argument("--model", default="intfloat/multilingual-e5-small",
                    help="Embedding model (default: intfloat/multilingual-e5-small)")
    ap.add_argument("--e5", action="store_true", help="Use E5-style prefixes")

    # NLI backend selection
    ap.add_argument("--backend", choices=["ollama", "claude"], default="ollama",
                    help="NLI backend: ollama (local, default) or claude (API)")

    # Ollama options
    ap.add_argument("--ollama-model", default="mistral",
                    help="Ollama model (default: mistral)")

    # Claude options
    ap.add_argument("--claude-model", default="claude-sonnet-4-20250514",
                    choices=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-3-5-20241022"],
                    help="Claude model (default: claude-sonnet-4-20250514)")
    ap.add_argument("--api-key", default=None,
                    help="Anthropic API key (or use ANTHROPIC_API_KEY env var)")

    # Common NLI options
    ap.add_argument("--nli-lang", default="auto", choices=["auto", "en", "fr"],
                    help="Language hint for NLI (default: auto)")
    ap.add_argument("--topk", type=int, default=5,
                    help="Number of candidate clauses per requirement (default: 5)")

    # Output format
    ap.add_argument("--format", choices=["text", "json", "csv"], default="text",
                    help="Output format (default: text)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show detailed output including rationales")

    args = ap.parse_args()

    # Load index and encoder
    try:
        enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
        idx = FaissIndex.load(args.index)
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)

    # Create NLI client
    nli = create_nli_client(args)
    use_batch = isinstance(nli, ClaudeNLIClient)

    # Process requirements
    verdicts = []
    details = []

    for req in args.req:
        qv = enc.embed_query(req).astype("float32")[None, :]
        D, I, recs = idx.search(qv, args.topk)

        best_label = "No"
        best_rationale = ""
        best_evidence = ""
        evaluated = []

        if use_batch and recs[0]:
            # Claude: batch all clause checks
            pairs = [(r.text, req) for r in recs[0]]
            try:
                results = nli.check_batch(pairs)
                for rec, result, score in zip(recs[0], results, D[0]):
                    evaluated.append({
                        "clause": rec.text[:200],
                        "score": float(score),
                        "label": result.label,
                        "rationale": result.rationale,
                        "file": rec.meta.get("file", ""),
                        "page": rec.meta.get("page", 0)
                    })
                    # Update best
                    if result.label == "Yes" and best_label != "Yes":
                        best_label = "Yes"
                        best_rationale = result.rationale
                        best_evidence = rec.text[:300]
                    elif result.label == "Partial" and best_label == "No":
                        best_label = "Partial"
                        best_rationale = result.rationale
                        best_evidence = rec.text[:300]
            except Exception as e:
                best_rationale = f"Batch error: {e}"
        else:
            # Ollama: sequential with early stopping
            for rec, score in zip(recs[0], D[0]):
                result = nli.check(rec.text, req)
                evaluated.append({
                    "clause": rec.text[:200],
                    "score": float(score),
                    "label": result.label,
                    "rationale": result.rationale,
                    "file": rec.meta.get("file", ""),
                    "page": rec.meta.get("page", 0)
                })
                if result.label == "Yes":
                    best_label = "Yes"
                    best_rationale = result.rationale
                    best_evidence = rec.text[:300]
                    break
                elif result.label == "Partial" and best_label == "No":
                    best_label = "Partial"
                    best_rationale = result.rationale
                    best_evidence = rec.text[:300]

        verdicts.append(RequirementVerdict(
            requirement=req, label=best_label, rationale=best_rationale, weight=1.0
        ))
        details.append({
            "requirement": req,
            "verdict": best_label,
            "rationale": best_rationale,
            "evidence": best_evidence,
            "evaluated": evaluated if args.verbose else []
        })

    # Compute score
    scorer = FitScorer()
    raw_score = scorer.fit_score(verdicts)
    percent_score = scorer.to_percent(raw_score)

    # Output results
    if args.format == "json":
        output = {
            "score": percent_score,
            "backend": args.backend,
            "model": args.claude_model if args.backend == "claude" else args.ollama_model,
            "requirements": details,
            "summary": {
                "total": len(verdicts),
                "yes": sum(1 for v in verdicts if v.label == "Yes"),
                "partial": sum(1 for v in verdicts if v.label == "Partial"),
                "no": sum(1 for v in verdicts if v.label == "No")
            }
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    elif args.format == "csv":
        print("requirement,verdict,rationale")
        for d in details:
            req_escaped = d["requirement"].replace('"', '""')
            rat_escaped = d["rationale"].replace('"', '""')
            print(f'"{req_escaped}",{d["verdict"]},"{rat_escaped}"')

    else:  # text
        print(f"\nFit Score: {percent_score}/100")
        print(f"Backend: {args.backend} ({args.claude_model if args.backend == 'claude' else args.ollama_model})")
        print(f"\nResults ({len(verdicts)} requirements):")
        print("-" * 60)
        for v, d in zip(verdicts, details):
            icon = {"Yes": "✓", "Partial": "~", "No": "✗"}[v.label]
            print(f" {icon} [{v.label:7}] {v.requirement}")
            if args.verbose and v.rationale:
                print(f"           Rationale: {v.rationale}")
                if d.get("evidence"):
                    print(f"           Evidence: {d['evidence'][:100]}...")
        print("-" * 60)
        yes_count = sum(1 for v in verdicts if v.label == "Yes")
        partial_count = sum(1 for v in verdicts if v.label == "Partial")
        no_count = sum(1 for v in verdicts if v.label == "No")
        print(f"Summary: {yes_count} Yes, {partial_count} Partial, {no_count} No")


if __name__ == "__main__":
    main()
