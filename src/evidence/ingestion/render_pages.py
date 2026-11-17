"""Render PDF pages to PNG for the visual index (PLAN.MD Phase 4).

Optional; requires pymupdf. Produces one PNG per page into an output dir and
returns a mapping of page number -> image path so the visual retrieval node
can later embed them with ColSmol / CLIP.
"""

from __future__ import annotations

from pathlib import Path


def render_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 150,
    max_pages: int | None = None,
) -> dict[int, str]:
    """Render each page of `pdf_path` to `<out_dir>/page_<n>.png`.

    Returns `{page_number (1-based): png_path}`.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # pymupdf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pymupdf is required to render pages. "
            "Install with `pip install pymupdf` or `pip install evidence[pdf]`."
        ) from exc

    mapping: dict[int, str] = {}
    with fitz.open(str(pdf_path)) as doc:
        pages = doc.page_count
        if max_pages is not None:
            pages = min(pages, max_pages)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = out_dir / f"page_{i+1}.png"
            pix.save(str(out_path))
            mapping[i + 1] = str(out_path)
    return mapping


def pages_for_corpus(corpus_dir: str | Path, out_root: str | Path) -> dict[str, dict[int, str]]:
    """Render every PDF in a corpus directory. Returns {paper_path: {page: png}}."""
    corpus_dir = Path(corpus_dir)
    out_root = Path(out_root)
    result: dict[str, dict[int, str]] = {}
    for pdf in sorted(corpus_dir.rglob("*.pdf")):
        out_dir = out_root / pdf.stem
        try:
            result[str(pdf)] = render_pages(pdf, out_dir)
        except Exception:  # noqa: BLE001 - skip a bad file, keep going
            continue
    return result
