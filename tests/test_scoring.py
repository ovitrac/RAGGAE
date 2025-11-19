# tests/test_scoring.py
from RAGGAE.core.scoring import FitScorer, RequirementVerdict

def test_fit_scorer_weights():
    verdicts = [
        RequirementVerdict("ISO 27001", "Yes", weight=2.0),
        RequirementVerdict("MLflow on K8s", "Partial", weight=1.0),
        RequirementVerdict("Data in EU", "No", weight=1.0),
    ]
    fs = FitScorer()
    s = fs.fit_score(verdicts)
    assert 0.49 <= s <= 1.0
    p = fs.to_percent(s)
    assert 0 <= p <= 100
