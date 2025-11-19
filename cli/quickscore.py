# RAGGAE/cli/quickscore.py
"""
Quick NLI-based fit score over a few canned requirements.

Command-line tool for requirement-based compliance scoring using local
NLI models (Ollama). Computes weighted fit scores from Yes/Partial/No verdicts.

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Usage:
    python -m RAGGAE.cli.quickscore --index tender.idx --model intfloat/multilingual-e5-small --e5 \
        --req "Provider must be ISO 27001 certified" \
        --req "Deployments on Kubernetes with GitOps" \
        --topk 5
"""
from __future__ import annotations
import argparse
import numpy as np

from RAGGAE.core.embeddings import STBiEncoder
from RAGGAE.core.index_faiss import FaissIndex
from RAGGAE.core.nli_ollama import NLIClient, NLIConfig
from RAGGAE.core.scoring import FitScorer, RequirementVerdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--e5", action="store_true")
    ap.add_argument("--req", action="append", help="Requirement string; repeat to add more", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ollama-model", default="mistral")
    ap.add_argument("--nli-lang", default="auto", help="auto | en | fr")
    args = ap.parse_args()

    enc = STBiEncoder(args.model, prefix_mode="e5" if args.e5 else "none")
    idx = FaissIndex.load(args.index)
    nli = NLIClient(NLIConfig(model=args.ollama_model, temperature=0.0, lang=args.nli_lang))

    # Get topk clauses by dense search for each requirement, then NLI them
    verdicts = []
    for req in args.req:
        qv = enc.embed_query(req).astype("float32")[None, :]
        D, I, recs = idx.search(qv, args.topk)
        label = "No"
        rationale = ""
        # first positive hit wins (Yes > Partial > No)
        for r in recs[0]:
            res = nli.check(r.text, req)
            if res.label == "Yes":
                label, rationale = res.label, res.rationale
                break
            elif res.label == "Partial" and label != "Yes":
                label, rationale = res.label, res.rationale
        verdicts.append(RequirementVerdict(requirement=req, label=label, rationale=rationale, weight=1.0))

    scorer = FitScorer()
    s = scorer.fit_score(verdicts)
    print(f"Fit score: {scorer.to_percent(s)}/100")
    for v in verdicts:
        print(f" - {v.requirement}: {v.label}")

if __name__ == "__main__":
    main()
