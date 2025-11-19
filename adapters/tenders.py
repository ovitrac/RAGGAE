# RAGGAE/adapters/tenders.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..io.pdf import extract_blocks, to_texts_and_meta, PDFBlock

@dataclass
class TenderDoc:
    """Parsed tender document with blocks and convenient accessors."""
    path: str
    blocks: List[PDFBlock]

    @property
    def texts(self) -> List[str]:
        return [b.text for b in self.blocks]

    @property
    def metadatas(self) -> List[Dict]:
        return [b.as_metadata() for b in self.blocks]

    def __len__(self) -> int:
        return len(self.blocks)

    def __str__(self) -> str:
        return f"TenderDoc<{self.path}> blocks={len(self)}"

    def __repr__(self) -> str:
        return f"TenderDoc(path={self.path!r}, blocks={len(self.blocks)})"

class TenderAdapter:
    """
    Adapter for tenders: parse, chunk, and provide field hints.

    Methods
    -------
    parse(path) -> TenderDoc
    extract_requirements(texts) -> list[str] (heuristic MUST/SHALL mining)
    key_hints(texts) -> map of common tender fields (deadline, budget, lots...)

    Examples
    --------
    >>> from RAGGAE.adapters.tenders import TenderAdapter
    >>> doc = TenderAdapter().parse("tender.pdf")
    >>> reqs = TenderAdapter.extract_requirements(doc.texts)[:5]
    >>> reqs
    ['The provider SHALL ...', 'The solution MUST ...', ...]
    """
    REQ_PAT = r"(?i)\\b(must|shall|should|doit|exigence|obligatoire)\\b"

    def parse(self, path: str, min_chars: int = 40) -> TenderDoc:
        blocks = extract_blocks(path, min_chars=min_chars, keep_headers=True)
        return TenderDoc(path=path, blocks=blocks)

    @staticmethod
    def extract_requirements(texts: Iterable[str]) -> List[str]:
        import re
        reqs: List[str] = []
        pat = re.compile(TenderAdapter.REQ_PAT)
        for t in texts:
            if pat.search(t):
                reqs.append(t)
        return reqs

    @staticmethod
    def key_hints(texts: Iterable[str]) -> Dict[str, List[str]]:
        """
        Very light heuristics to find frequent tender fields.
        """
        import re
        keys = {
            "deadline": [],
            "budget": [],
            "lot": [],
            "certifications": [],
            "security": [],
        }
        for t in texts:
            low = t.lower()
            if any(w in low for w in ["deadline", "date limite", "remise des offres"]):
                keys["deadline"].append(t)
            if any(w in low for w in ["budget", "montant", "enveloppe"]):
                keys["budget"].append(t)
            if "lot" in low:
                keys["lot"].append(t)
            if any(w in low for w in ["iso 27001", "soc2", "iso/iec"]):
                keys["certifications"].append(t)
            if any(w in low for w in ["rgpd", "gdpr", "souveraineté", "security", "sécurité"]):
                keys["security"].append(t)
        return {k: v[:10] for k, v in keys.items()}
