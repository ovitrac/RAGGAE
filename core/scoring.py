"""
RAGGAE/core/scoring.py

Fit scoring system for requirement-based compliance evaluation.
Provides weighted scoring from NLI verdicts with support for extra signals.

This module provides:
- RequirementVerdict: Dataclass for requirement evaluation results
- FitScorer: Configurable scorer with weighted aggregation

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from core.scoring import FitScorer, RequirementVerdict
>>> verdicts = [
...     RequirementVerdict("ISO 27001", "Yes", weight=1.5),
...     RequirementVerdict("MLflow on K8s", "Partial", weight=1.0),
...     RequirementVerdict("Data in EU", "No", weight=1.0),
... ]
>>> scorer = FitScorer()
>>> score = scorer.fit_score(verdicts)
>>> print(f"Fit: {scorer.to_percent(score)}/100")
"""
# RAGGAE/core/scoring.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

LABEL_WEIGHT = {"Yes": 1.0, "Partial": 0.5, "No": 0.0}

@dataclass
class RequirementVerdict:
    requirement: str
    label: str               # "Yes" | "Partial" | "No"
    rationale: str = ""
    weight: float = 1.0      # importance/priority (1.0 default)

    def score(self) -> float:
        return LABEL_WEIGHT.get(self.label, 0.0) * self.weight

    def __str__(self) -> str:
        return f"{self.requirement[:40]!r}: {self.label} (w={self.weight}, s={self.score():.2f})"

class FitScorer:
    """
    Simple weighted scoring from NLI verdicts + optional additive signals.

    - Base score: mean of weighted requirement scores (0..1)
    - Extra signals: any keyed floats (0..1) added with weights

    Example
    -------
    >>> from RAGGAE.core.scoring import FitScorer, RequirementVerdict
    >>> verdicts = [
    ...   RequirementVerdict("ISO 27001", "Yes", weight=1.5),
    ...   RequirementVerdict("MLflow on K8s", "Partial", weight=1.0),
    ...   RequirementVerdict("Data in EU", "No", weight=1.0),
    ... ]
    >>> fs = FitScorer()
    >>> fs.fit_score(verdicts)
    0.67  # (example)
    """
    def __init__(self, extra_signal_weights: Optional[Mapping[str, float]] = None):
        self.extra_signal_weights = dict(extra_signal_weights or {})

    def fit_score(self,
                  verdicts: Iterable[RequirementVerdict],
                  extra_signals: Optional[Mapping[str, float]] = None) -> float:
        verdicts = list(verdicts)
        if not verdicts:
            base = 0.0
        else:
            total_w = sum(v.weight for v in verdicts) or 1e-9
            base = sum(v.score() for v in verdicts) / total_w  # 0..1

        add = 0.0
        if extra_signals:
            for key, val in extra_signals.items():
                w = self.extra_signal_weights.get(key, 1.0)
                add += w * float(val)  # assume 0..1

        # clamp to [0,1]
        final = max(0.0, min(1.0, base + add))
        return final

    def to_percent(self, value: float) -> float:
        return round(100.0 * float(value), 1)

    def __str__(self) -> str:
        return f"FitScorer(extra={self.extra_signal_weights})"

    def __repr__(self) -> str:
        return f"FitScorer(extra_signal_weights={self.extra_signal_weights!r})"
