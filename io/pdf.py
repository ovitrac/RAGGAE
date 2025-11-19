"""
RAGGAE/io/pdf.py

PDF parsing with layout-aware block extraction using PyMuPDF.
Provides fast and robust text extraction with provenance tracking
(page, block, bounding box) for RAG applications.

This module provides:
- PDFBlock: Dataclass for text blocks with spatial metadata
- extract_blocks: Main extraction function with filtering
- Helper functions for PDF processing

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: October 31, 2025
License: MIT

Examples
--------
>>> from io.pdf import extract_blocks
>>> blocks = extract_blocks("tender.pdf", min_chars=40)
>>> for b in blocks[:3]:
...     print(f"Page {b.page}, Block {b.block}: {b.text[:50]}...")
"""
# RAGGAE/io/pdf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path

import re
import fitz  # PyMuPDF

@dataclass
class PDFBlock:
    """One text block with provenance anchors."""
    text: str
    page: int
    block: int
    bbox: Tuple[float, float, float, float]

    def as_metadata(self) -> Dict:
        return {"page": self.page, "block": self.block, "bbox": self.bbox}

    def __str__(self) -> str:
        t = self.text.replace("\n", " ")
        return f"PDFBlock(p.{self.page},b{self.block}, {t[:60]!r}...)"

    def __repr__(self) -> str:
        return f"PDFBlock(page={self.page}, block={self.block}, bbox={tuple(round(v,1) for v in self.bbox)})"

def load_pdf(path: str | Path) -> fitz.Document:
    """Open a PDF with PyMuPDF."""
    return fitz.open(path)

def extract_blocks(path: str | Path,
                   min_chars: int = 40,
                   keep_headers: bool = True,
                   max_blocks: Optional[int] = None) -> List[PDFBlock]:
    """
    Extract simple text blocks from a PDF (fast and robust).

    Parameters
    ----------
    min_chars : int
        Ignore tiny blocks (page numbers, footers) below this length.
    keep_headers : bool
        If False, try to drop very short, repeated header/footer lines.
    max_blocks : int | None
        Cap total number of blocks extracted.

    Returns
    -------
    list[PDFBlock]

    Examples
    --------
    >>> from RAGGAE.io.pdf import extract_blocks
    >>> blocks = extract_blocks("tender.pdf")
    >>> blocks[0]
    PDFBlock(p.1,b0, 'Objet : Mise en place d'une plateforme MLOps ...')
    """
    path = Path(path)
    doc = fitz.open(path)
    out: List[PDFBlock] = []
    header_footer_cache = set()

    for pno in range(len(doc)):
        page = doc[pno]
        for bi, blk in enumerate(page.get_text("blocks")):
            x0, y0, x1, y1, text, *_ = blk
            txt = (text or "").strip()
            if not txt:
                continue
            if not keep_headers:
                short = re.sub(r"\s+", " ", txt)
                if len(short) < 30:
                    key = (short, pno)
                    if key in header_footer_cache:
                        continue
                    header_footer_cache.add(key)
            if len(txt) >= min_chars:
                out.append(PDFBlock(text=txt, page=pno+1, block=bi, bbox=(x0, y0, x1, y1)))
                if max_blocks and len(out) >= max_blocks:
                    return out
    return out

def to_texts_and_meta(blocks: Iterable[PDFBlock]) -> Tuple[List[str], List[Dict]]:
    """Split blocks into parallel lists for indexing."""
    texts, metas = [], []
    for b in blocks:
        texts.append(b.text)
        metas.append(b.as_metadata())
    return texts, metas
