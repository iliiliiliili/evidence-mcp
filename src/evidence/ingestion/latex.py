"""LaTeX source ingestion (preferred when available — clean math/macros).

We extract a plain-text projection of the LaTeX document body: section
commands are converted to markdown headings, math is left inline, and
common macros are stripped. This is intentionally conservative — the goal is
a readable chunkable projection, not full semantic parsing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .pdf import PdfDocument


@dataclass
class LatexDocument:
    path: Path
    text: str
    pages: list[str] = field(default_factory=list)
    n_pages: int = 1
    title: str = ""
    meta: dict = field(default_factory=dict)  # type: ignore[assignment]


# Commands we drop entirely (comments handled per-line).
_DROP_CMD = re.compile(
    r"\\(?:bibliography|nocite|cite|url|href|footnote|footnote|includegraphics|vspace|hline|clearpage|pagebreak|newpage|bigskip|medskip|smallskip|input|include|sectionmark)"
)
# Section-ish commands → markdown headings.
_SECTION = re.compile(
    r"\\(section|subsection|subsubsection|paragraph|part)\*?\{([^{}]+)\}"
)
# Inline math: leave as-is (chunker keeps it).
# Strip comments (a % not preceded by a backslash to the end of line).
_COMMENT = re.compile(r"(?<!\\)%.*$")
# Collapse display-environment wrappers, keeping their body.
_ENVIR = re.compile(
    r"\\begin\{(?P<env>equation|align|align\*|equation\*|abstract|itemize|enumerate|center|list|tabular)\}.*?\\end\{(?P=env)\}",
    re.DOTALL,
)


def _strip_environment(text: str) -> str:
    # Collapse display-environment wrappers, keeping their contents.
    def _keep(match: re.Match) -> str:
        inner = match.group(0)
        env = match.group("env")
        # Remove the begin/end markers but keep the body.
        inner = re.sub(r"\\begin\{" + re.escape(env) + r"\}", "", inner, count=1)
        inner = re.sub(r"\\end\{" + re.escape(env) + r"\}\s*$", "", inner)
        return inner

    return _ENVIR.sub(_keep, text)


def _to_markdown_sections(text: str) -> str:
    depth_map = {
        "part": 1,
        "section": 1,
        "subsection": 2,
        "subsubsection": 3,
        "paragraph": 4,
    }

    def _sub(match: re.Match) -> str:
        kind, title = match.group(1), match.group(2).strip()
        level = depth_map.get(kind, 2)
        return "\n\n" + "#" * level + " " + title + "\n\n"

    return _SECTION.sub(_sub, text)


def extract_latex(path: str | Path) -> LatexDocument:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Drop comments.
    lines = [
        _COMMENT.sub("", ln)
        for ln in raw.splitlines()
    ]
    text = "\n".join(lines)

    # Drop \begin{document}…\end{document} wrapper if present, keep body.
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # Otherwise, drop the preamble (everything before the first \section
        # or \title, if any) to reduce noise.
        m2 = re.search(r"\\(?:section|title|begin\{abstract\})", text)
        if m2:
            text = text[m2.start():]

    # Grab a title if present.
    title_match = re.search(r"\\title\{([^{}]+)\}", text)
    title = title_match.group(1).strip() if title_match else path.stem

    # Drop graphic/cite noise, keep math.
    text = _DROP_CMD.sub(" ", text)
    text = _strip_environment(text)
    text = _to_markdown_sections(text)

    # Collapse excess blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return LatexDocument(
        path=path,
        text=text,
        n_pages=1,
        title=title,
        meta={"sha256": hashlib.sha256(raw.encode()).hexdigest()[:16]},
    )


def as_pdf_doc(doc: LatexDocument) -> PdfDocument:
    """Adapt a LatexDocument to the PdfDocument shape for the shared pipeline."""
    return PdfDocument(
        path=doc.path,
        text=doc.text,
        pages=[doc.text],
        n_pages=1,
        title=doc.title,
        meta=doc.meta,
    )
