# RAGGAE/io/tables.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class TableCell:
    row: int
    col: int
    text: str

    def __str__(self) -> str:
        return f"({self.row},{self.col})={self.text[:30]!r}"

@dataclass
class TableExtract:
    page: int
    cells: List[TableCell]

    def as_key_values(self) -> List[Dict[str, Any]]:
        """
        Heuristic: interpret 2-column tables as key-value pairs.
        """
        kv = []
        rows: Dict[int, Dict[int, str]] = {}
        for c in self.cells:
            rows.setdefault(c.row, {})[c.col] = c.text.strip()
        for rid in sorted(rows):
            cols = rows[rid]
            if len(cols) == 2:
                k = cols.get(0, "")
                v = cols.get(1, "")
                if k and v:
                    kv.append({"key": k, "value": v, "page": self.page, "row": rid})
        return kv

    def __str__(self) -> str:
        return f"TableExtract(page={self.page}, cells={len(self.cells)})"

try:
    import camelot  # type: ignore
except Exception:
    camelot = None

def extract_tables_camelot(path: str | Path, pages: str = "all", flavor: str = "lattice") -> List[TableExtract]:
    """
    Extract tables using Camelot if available.

    Returns empty list if Camelot isn't installed.

    Examples
    --------
    >>> from RAGGAE.io.tables import extract_tables_camelot
    >>> tables = extract_tables_camelot("tender.pdf")
    >>> tables[0].as_key_values()[:3]
    [{'key': 'RPO', 'value': '15 minutes', 'page': 3, 'row': 4}, ...]
    """
    if camelot is None:
        return []
    path = str(Path(path))
    tlist = camelot.read_pdf(path, pages=pages, flavor=flavor)
    out: List[TableExtract] = []
    for t in tlist:
        cells: List[TableCell] = []
        df = t.df  # pandas DataFrame
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                txt = str(df.iat[r, c]).strip()
                if txt:
                    cells.append(TableCell(row=r, col=c, text=txt))
        out.append(TableExtract(page=getattr(t, "page", -1), cells=cells))
    return out
