"""
RAGGAE/core/nli_ollama.py

Natural Language Inference client for local compliance checking via Ollama.
Provides robust NLI with automatic language detection, JSON parsing, and
label sanitization for requirement satisfaction evaluation.

This module provides:
- NLIClient: Ollama-based NLI with retry logic
- NLIResult: Dataclass for inference results
- NLIConfig: Configuration for model and inference parameters

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from core.nli_ollama import NLIClient, NLIConfig
>>> nli = NLIClient(NLIConfig(model="mistral", lang="auto"))
>>> result = nli.check(
...     clause="Le prestataire est certifié ISO/IEC 27001:2022.",
...     requirement="Provider must be ISO 27001 certified"
... )
>>> print(f"{result.label}: {result.rationale}")
"""
# RAGGAE/core/nli_ollama.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import json, re

try:
    import ollama
except Exception:
    ollama = None

@dataclass
class NLIResult:
    label: str          # "Yes" | "No" | "Partial"
    rationale: str

    def __str__(self) -> str:
        return f"{self.label}: {self.rationale[:60]}..."

ALLOWED = {"Yes","No","Partial"}

def _sanitize_label(lbl: str) -> str:
    if not isinstance(lbl, str):
        return "No"
    lbl = lbl.strip().title()
    return lbl if lbl in ALLOWED else "No"

def _looks_invalid(s: str) -> bool:
    if s is None: return True
    s = str(s).strip()
    return (s == "" or s.lower() == "nan" or "nan" in s.lower())

@dataclass
class NLIConfig:
    model: str = "mistral"
    temperature: float = 0.0
    num_ctx: int = 4096
    lang: str = "auto"  # "fr" | "en" | "auto"

    def __str__(self) -> str:
        return f"NLIConfig(model={self.model}, temp={self.temperature}, ctx={self.num_ctx}, lang={self.lang})"

SYS = (
  "You are a strict compliance checker. "
  "Return ONLY compact JSON with keys: label, rationale. "
  "label ∈ ['Yes','No','Partial']."
)

def _parse_json_loose(s: str) -> Optional[Dict]:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I|re.M)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class NLIClient:
    """
    Deterministic local NLI (Ollama) for requirement satisfaction checks.

    Example
    -------
    >>> from RAGGAE.core.nli_ollama import NLIClient
    >>> nli = NLIClient()
    >>> nli.check("Le prestataire est certifié ISO/IEC 27001:2022.",
    ...           "Provider must be ISO 27001 certified")
    NLIResult(label='Yes', rationale='...')

    Batch matrix:
    >>> hits = ["Uses MLflow on Kubernetes", "Data hosted in EU region"]
    >>> reqs = ["MLflow used", "Data hosted in EU"]
    >>> df, verdicts = nli.matrix(hits, reqs, topk=2)  # requires pandas
    """
    def __init__(self, config: Optional[NLIConfig] = None):
        if ollama is None:
            raise ImportError("ollama python client not installed. `pip install ollama`.")
        self.config = config or NLIConfig()

    def check(self, clause: str, requirement: str) -> NLIResult:
        def _ask(lang: str) -> NLIResult:
            prompt = (
                f"Language: {lang}. "
                f'Clause: "{clause}"\n'
                f'Requirement: "{requirement}"\n'
                'Respond as JSON: {"label":"Yes|No|Partial","rationale":"..."}'
            )
            r = ollama.chat(
                model=self.config.model,
                options={"temperature": self.config.temperature, "num_ctx": self.config.num_ctx},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt}]
            )
            out = _parse_json_loose(r["message"]["content"]) or {"label":"No","rationale":"Invalid or non-JSON output"}
            label = _sanitize_label(out.get("label","No"))
            rationale = str(out.get("rationale","")).strip()
            if _looks_invalid(rationale):
                rationale = "LLM rationale invalid/empty (possibly wrong language)."
            return NLIResult(label=label, rationale=rationale)

        # first pass with configured language (often "auto")
        res = _ask(self.config.lang)
        if res.label == "No" and "wrong language" in res.rationale.lower():
            # direct fallback
            fallback = "en" if self.config.lang != "en" else "fr"
            res = _ask(fallback)
        return res

    # Optional helper if pandas is available
    def matrix(self, clauses: Iterable[str], requirements: Iterable[str], topk: int = 5):
        try:
            import pandas as pd
        except Exception:
            raise ImportError("`pandas` required for matrix(); install via `mamba install -c conda-forge pandas`.")
        rows = []
        for req in requirements:
            for i, clause in enumerate(list(clauses)[:topk]):
                res = self.check(clause, req)
                rows.append({
                    "requirement": req,
                    "hit_rank": i + 1,
                    "label": res.label,
                    "rationale": res.rationale,
                    "snippet": clause[:160].replace("\n", " ")
                })
        df = pd.DataFrame(rows)
        order = {"Yes": 2, "Partial": 1, "No": 0}
        verdict = (df.assign(score=df["label"].map(order))
                     .groupby("requirement")["score"].max()
                     .map({2: "Yes", 1: "Partial", 0: "No"}))
        return df, verdict

    def __str__(self) -> str:
        return f"NLIClient<{self.config}>"

    def __repr__(self) -> str:
        return f"NLIClient(model={self.config.model!r}, lang={self.config.lang!r})"
