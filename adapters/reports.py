# RAGGAE/adapters/reports.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from ..io.pdf import extract_blocks, PDFBlock

@dataclass
class ReportSection:
    title: str
    page: int
    snippet: str

    def __str__(self) -> str:
        return f"[p.{self.page}] {self.title}: {self.snippet[:50]}..."

@dataclass
class ReportDoc:
    path: str
    blocks: List[PDFBlock]
    sections: List[ReportSection]

    def __len__(self) -> int:
        return len(self.blocks)

    def __str__(self) -> str:
        return f"ReportDoc<{self.path}> blocks={len(self.blocks)} sections={len(self.sections)}"

class ReportAdapter:
    """
    Minimal scientific/technical report adapter.
    Heuristics:
      - Detect uppercase / numbered headings as sections
      - Keep first sentence as a snippet
    """
    def parse(self, path: str, min_chars: int = 40) -> ReportDoc:
        blocks = extract_blocks(path, min_chars=min_chars, keep_headers=True)
        sections = self._sections(blocks)
        return ReportDoc(path=path, blocks=blocks, sections=sections)

    @staticmethod
    def _sections(blocks: List[PDFBlock]) -> List[ReportSection]:
        import re
        out: List[ReportSection] = []
        head = re.compile(r"^(\d+(\.\d+)*)\s+|^[A-Z][A-Z0-9\s\-\.:]{6,}$")
        for b in blocks:
            first_line = b.text.splitlines()[0].strip()
            if head.match(first_line):
                snippet = b.text.split(".")[0][:160]
                out.append(ReportSection(title=first_line[:80], page=b.page, snippet=snippet))
        return out
