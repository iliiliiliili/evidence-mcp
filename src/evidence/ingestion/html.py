"""HTML document ingestion — a dependency-free text projection.

We convert an HTML file into readable text (headings, paragraphs, list
items, table cells) using the stdlib `html.parser`, so no `beautifulsoup4` /
`lxml` / `html2text` dependency is required. This is intentionally
conservative: the goal is a clean, chunkable projection of the document body
— not full semantic parsing. Inline tags (`<b>`, `<i>`, `<a>`, …) are dropped
but their text is kept. `<script>` / `<style>` / `<noscript>` / `<head>`
content is stripped.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from .pdf import PdfDocument

# Tags whose *text* should be discarded entirely (noise).
_SKIP_CONTENT = {"script", "style", "noscript", "template", "head", "svg", "iframe"}

# Block-level tags that should introduce a newline in the output.
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "hr", "li", "ul", "ol", "dl", "dt", "dd",
    "tr", "td", "th", "table", "blockquote", "pre",
    "figure", "figcaption", "nav", "header", "footer",
}

# Heading level → markdown heading depth (h1 → #, h2 → ##, …).
_HEADING_TAGS = {
    "h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6,
}

# Tags whose text should be space-separated rather than line-broken.
_INLINE_BREAK = {"li", "td", "th", "dt", "dd"}


class _TextExtractor(HTMLParser):
    """Stream HTML into a list of text blocks (one per block-level element)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._skip_depth = 0
        self._heading_prefix = ""

    # -- helpers ---------------------------------------------------------
    def _flush_buf(self) -> None:
        text = "".join(self._buf).strip()
        if text:
            self.blocks.append(text)
        self._buf = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS:
            # Close the current line, then prefix the heading.
            self._flush_buf()
            self._heading_prefix = "#" * _HEADING_TAGS[tag] + " "
        elif tag in _BLOCK_TAGS:
            self._flush_buf()
        elif tag in _INLINE_BREAK:
            # New cell / item on its own line.
            self._flush_buf()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS or tag in _BLOCK_TAGS or tag in _INLINE_BREAK:
            self._flush_buf()
        self._heading_prefix = ""

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if not data:
            return
        prefix = self._heading_prefix if self._heading_prefix else ""
        self._buf.append(prefix + data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        self._flush_buf()


def _extract_title(html_text: str) -> str:
    """Best-effort `<title>` grab (case-insensitive), else ""."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()
        # Unescape a few common entities.
        for ent, ch in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                        ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
            title = title.replace(ent, ch)
        return title
    return ""


def extract_html(path: str | Path) -> PdfDocument:
    """Convert an HTML file into a single-page `PdfDocument` of readable text.

    Returns a `PdfDocument` (single page) so it slots into the same
    chunking / storage path as PDF / LaTeX / markdown sources.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML should not abort ingestion
        pass

    # Collapse runs of blank lines and trailing whitespace per block.
    blocks: list[str] = []
    for b in parser.blocks:
        cleaned = " ".join(b.split())
        if cleaned:
            blocks.append(cleaned)

    text = "\n\n".join(blocks)
    title = _extract_title(raw) or path.stem

    # Surface the "PDF viewer wrapper" dead-end explicitly instead of
    # silently indexing an empty chunk.
    if not text.strip():
        if re.search(r"<\s*(iframe|embed|object)\b", raw, re.IGNORECASE) or ".pdf" in raw.lower():
            raise ValueError(
                f"{path.name} is a PDF-viewer wrapper (an <iframe>/<embed> that "
                "loads an external PDF) with no extractable text. Save the actual "
                "PDF file (not the 'page as HTML' viewer) and re-run "
                "`evidence ingest`."
            )
        raise ValueError(
            f"{path.name} contained no extractable text (it may be a "
            "JavaScript-rendered page). Save a PDF or plain-text version instead."
        )

    return PdfDocument(
        path=path,
        text=text,
        pages=[text],
        n_pages=1,
        title=title,
        meta={"source": "html"},
    )
