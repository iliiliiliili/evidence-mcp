"""PDF text extraction (pymupdf / PyMuPDF), with a pure-python fallback.

pymupdf is an optional dependency; if absent we still emit an error that tells
the user how to enable it. Page rendering for the visual index lives in
`render_pages.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PdfDocument:
    path: Path
    text: str
    pages: list[str]  # per-page text
    n_pages: int
    title: str = ""
    meta: dict = field(default_factory=dict)


def extract_pdf(path: str | Path) -> PdfDocument:
    """Extract per-page text from a PDF via pymupdf.

    Raises `RuntimeError` with install guidance when pymupdf is missing.
    """
    path = Path(path)
    try:
        import fitz  # pymupdf
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "pymupdf is not installed. Install with `pip install pymupdf` "
            "or `pip install evidence[pdf]` to ingest PDFs."
        ) from exc

    pages: list[str] = []
    title = ""
    with fitz.open(str(path)) as doc:
        title = doc.metadata.get("title", "") or path.stem
        for page in doc:
            pages.append(page.get_text("text"))
    text = "\n\n".join(pages)
    return PdfDocument(
        path=path,
        text=text,
        pages=pages,
        n_pages=len(pages),
        title=title,
        meta={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]},
    )


def doc_id_for(path: str | Path) -> str:
    """Stable, content-independent id for a document (path-derived)."""
    p = Path(path)
    h = hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{p.stem}__{h}"
