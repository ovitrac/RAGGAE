# RAGGAE/adapters/cv.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from ..io.pdf import extract_blocks, PDFBlock

@dataclass
class CVExperience:
    organization: str
    title: str
    period: str
    snippet: str
    page: int

    def __str__(self) -> str:
        return f"{self.title} @ {self.organization} ({self.period})"

@dataclass
class CVDoc:
    path: str
    blocks: List[PDFBlock]
    experiences: List[CVExperience]

    @property
    def texts(self) -> List[str]:
        return [b.text for b in self.blocks]

    def __len__(self) -> int:
        return len(self.blocks)

    def __str__(self) -> str:
        return f"CVDoc<{self.path}> blocks={len(self)} exp={len(self.experiences)}"

class CVAdapter:
    """
    Heuristic CV parser: extract experiences by simple patterning.
    Fine-tune later with regex/NLP and layout detection.

    Examples
    --------
    >>> from RAGGAE.adapters.cv import CVAdapter
    >>> cv = CVAdapter().parse("resume.pdf")
    >>> cv.experiences[0]
    Data Scientist @ Adservio (2023–now)
    """
    def parse(self, path: str, min_chars: int = 30) -> CVDoc:
        blocks = extract_blocks(path, min_chars=min_chars, keep_headers=True)
        exps = self._extract_experiences(blocks)
        return CVDoc(path=path, blocks=blocks, experiences=exps)

    @staticmethod
    def _extract_experiences(blocks: List[PDFBlock]) -> List[CVExperience]:
        import re
        exps: List[CVExperience] = []
        pat_period = re.compile(r"(?i)(\d{4})(?:\s*[–-]\s*|\s+to\s+|\s*-\s*)(present|\d{4}|now)")
        for b in blocks:
            txt = b.text.replace("\n", " ")
            if pat_period.search(txt):
                # naive split: role @ org (period)
                title = txt[:60]
                org = "Unknown"
                period = pat_period.search(txt).group(0)
                exps.append(CVExperience(organization=org, title=title, period=period, snippet=txt[:180], page=b.page))
        return exps
