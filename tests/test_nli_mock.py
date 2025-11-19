# tests/test_nli_mock.py
"""
We mock the NLI client to avoid requiring an Ollama daemon in CI.
"""
from RAGGAE.core.nli_ollama import NLIClient, NLIConfig, NLIResult

class DummyNLI(NLIClient):
    def __init__(self):
        # don't call super(); we don't want to import ollama in tests
        self.config = NLIConfig(model="dummy", temperature=0.0, lang="en")

    def check(self, clause: str, requirement: str) -> NLIResult:
        ok = requirement.lower().split()[0] in clause.lower()
        return NLIResult(label="Yes" if ok else "No", rationale="mocked")

def test_dummy_nli():
    nli = DummyNLI()
    assert nli.check("Provider ISO 27001 certified", "ISO 27001").label == "Yes"
    assert nli.check("Something else", "ISO 27001").label == "No"
