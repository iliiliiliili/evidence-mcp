"""Structure-aware chunking (PLAN.MD offline pipeline, step 2).

Split text on section headings / blank-line paragraphs, then pack into
token-budgeted chunks with overlap. Falls back to a tiktoken tokenizer when
available, else a whitespace heuristic so chunking works in any environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import ChunkingConfig


@dataclass
class Section:
    heading: str
    text: str
    level: int = 1
    section_id: str = ""  # stable id within a paper (assigned post-hoc)


@dataclass
class Chunk:
    text: str
    char_start: int
    char_end: int
    heading: str = ""
    section_level: int = 1
    tokens: int = 0
    # Filled by the caller with corpus-level ids.
    paper_id: str = ""
    chunk_id: str = ""
    page: int | None = None
    meta: dict = field(default_factory=dict)
    # Hierarchical span fields (filled by split_sections):
    section_id: str = ""            # stable id of the section this chunk belongs to
    is_first_in_section: bool = False
    is_last_in_section: bool = False


def count_tokens(text: str) -> int:
    """Best-effort token count: tiktoken if present, else whitespace heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 - optional dependency
        # ~4 chars/token heuristic for English-ish technical text.
        return max(1, len(text) // 4)


_KNOWN_SECTION_LABEL = re.compile(
    r"^\s*"
    r"(abstract|background|introduction|related\s+work|contributions?|method(ology)?s?|"
    r"materials?\s+and\s+methods?|dataset|data\s+and\s+methods?|experimental\s+setup|"
    r"results?|experiments?|discussion|conclusions?|limitations?|acknowledg(e?ment)s?|"
    r"references|bibliography|appendix|appendices|availability\s+of\s+data|"
    r"ethical\s+considerations|impact\s+statement)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3}\.?)\s+([A-Z][A-Za-z0-9&\-\u2019'() /]{1,60})\s*$"
)
_ALLCAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9&\-\u2019'() /]{2,50})\s*$")
# Tokens that are abbreviations, not section titles — never treat as headings.
_ACRONYMS = {
    "LLM", "LLMS", "RAG", "KG", "AI", "ML", "NLP", "SQL", "API", "CPU", "GPU",
    "TPU", "SDK", "CLI", "URL", "JSON", "YAML", "CSV", "PDF", "DOI", "ARXIV",
    "OPENAI", "S2", "CROSSREF", "EMBEDDING", "EMBEDDINGS", "TOKEN", "TOKENS",
    "CONTEXT", "WINDOW", "MODEL", "MODELS", "PROMPT", "PROMPTS", "AGENT",
    "AGENTS", "BENCHMARK", "BENCHMARKS", "EVALUATION", "DATA", "GRAPH", "GRAPHS",
}


def _plain_section_markers(lines: list[str]) -> list[tuple[int, str]]:
    """Detect section-heading lines in plain text (no Markdown/LaTeX markers).

    Returns (line_index, heading_text) pairs in document order.
    Heuristics (in priority order):
      1. Known section label (case-insensitive): Abstract, Methods, Results…
      2. Numbered heading: "3.2  Evaluation Metric" or "1.1.2 Fine tuning"
      3. All-caps short phrase with at least one space (avoids bare acronyms)
    """
    hits: list[tuple[int, str]] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or len(line) > 80:
            continue
        if _KNOWN_SECTION_LABEL.match(line):
            hits.append((i, line))
            continue
        m = _NUMBERED_HEADING.match(line)
        if m and m.group(2).strip().upper() not in _ACRONYMS:
            hits.append((i, m.group(2).strip()))
            continue
        m = _ALLCAPS_HEADING.match(line)
        if m and " " in m.group(1) and m.group(1).strip().upper() not in _ACRONYMS:
            hits.append((i, m.group(1).strip()))
    return hits


def _build_sections_from_lines(lines: list[str]) -> list[Section]:
    """Build (heading, body) sections from raw lines, preferring Markdown, then
    LaTeX, then plain-text section markers. Returns one heading-less section
    if nothing structured is found."""
    # Pass 1: Markdown + LaTeX.
    sections: list[Section] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(Section(heading=current_heading, text=body))

    structured = False
    for raw_line in lines:
        line = raw_line.rstrip()
        m_md = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if m_md:
            structured = True
            flush()
            current_lines = []
            current_heading = m_md.group(2).strip()
            continue
        m_latex = re.match(r"^\\(section|subsection|subsubsection|paragraph)\*?\{(.*)\}", line)
        if m_latex:
            structured = True
            flush()
            current_lines = []
            current_heading = m_latex.group(2).strip()
            continue
        current_lines.append(raw_line)
    flush()
    if structured and sections:
        return sections

    # Pass 2: plain-text section markers.
    markers = _plain_section_markers(lines)
    if markers:
        sections: list[Section] = []
        n = len(lines)
        # Each marker at line index `i` starts a section that spans
        # lines[i : next_marker_index] (or lines[i:] for the last one).
        for k, (idx, heading) in enumerate(markers):
            end = markers[k + 1][0] if k + 1 < len(markers) else n
            # Skip the marker line itself; it becomes the heading, not body.
            body = "\n".join(lines[idx + 1: end]).strip()
            if body:
                sections.append(Section(heading=heading, text=body))
            else:
                # Empty-bodied section (marker directly followed by another
                # marker or EOF). Still record it — it may contain content
                # after further processing (e.g. a figure caption).
                sections.append(Section(heading=heading, text=""))
        # Preamble (anything before the first marker) — keep if non-trivial.
        first_idx = markers[0][0]
        preamble = "\n".join(lines[:first_idx]).strip()
        if preamble:
            sections.insert(0, Section(heading="", text=preamble))
        return sections
    # Nothing found: single heading-less section.
    return [Section(heading="", text="\n".join(lines).strip())]


def split_sections(text: str) -> list[Section]:
    """Split raw text into (heading, body) sections.

    Detects Markdown (#), LaTeX (\\section), and plain-text section markers
    (known labels, "N." numbering, all-caps phrases). Falls back to a single
    heading-less section when nothing structured is found.
    """
    return _build_sections_from_lines(text.splitlines())


def _pack_into_chunks(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    heading: str,
    level: int,
    section_id: str = "",
) -> list[Chunk]:
    """Pack a section's text into token-budgeted chunks with overlap.

    We split on sentence/paragraph boundaries so we never cut mid-sentence.
    """
    # Split into "soft" units: paragraphs first, then sentences.
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # Break long paragraphs into sentences to avoid oversized units.
        if count_tokens(para) <= max_tokens:
            units.append(para)
        else:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", para)
            units.extend(s.strip() for s in sentences if s.strip())

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    offset = 0  # char offset in the original text (approx)

    def push() -> None:
        nonlocal buf, buf_tokens
        joined = "\n\n".join(buf).strip()
        if joined and count_tokens(joined) >= min_tokens:
            chunks.append(
                Chunk(
                    text=joined,
                    char_start=offset,
                    char_end=offset + len(joined),
                    heading=heading,
                    section_level=level,
                    tokens=count_tokens(joined),
                    section_id=section_id,
                )
            )
        buf = []
        buf_tokens = 0

    for unit in units:
        ut = count_tokens(unit)
        if ut > max_tokens:
            # Unit alone exceeds budget: flush and emit it standalone.
            if buf:
                push()
            chunks.append(
                Chunk(
                    text=unit,
                    char_start=offset,
                    char_end=offset + len(unit),
                    heading=heading,
                    section_level=level,
                    tokens=ut,
                    section_id=section_id,
                )
            )
            offset += len(unit)
            continue
        if buf_tokens + ut > max_tokens and buf:
            # Overlap: carry the tail of the buffer into the next chunk.
            push()
            # Re-seed with roughly `overlap_tokens` of trailing text.
            tail = text_offset_tail("\n\n".join(units), overlap_tokens)
            if tail:
                buf = [tail, unit]
                buf_tokens = count_tokens(tail) + ut
                offset += len(tail)
            else:
                buf = [unit]
                buf_tokens = ut
        else:
            buf.append(unit)
            buf_tokens += ut
        offset += len(unit)
    if buf:
        push()

    # Fallback: if we produced nothing (e.g. very short text below min_tokens),
    # emit the whole section as one chunk so we never lose content.
    if not chunks and text.strip():
        chunks.append(
            Chunk(
                text=text.strip(),
                char_start=0,
                char_end=len(text.strip()),
                heading=heading,
                section_level=level,
                tokens=count_tokens(text),
                section_id=section_id,
            )
        )
    return chunks


def text_offset_tail(text: str, target_tokens: int) -> str:
    """Return the trailing portion of `text` that is ~`target_tokens` long."""
    if not text:
        return ""
    words = text.split()
    # Approx tokens ~ words * 1.3
    keep_words = max(1, int(target_tokens / 1.3))
    tail = " ".join(words[-keep_words:])
    return tail


def chunk_document(
    text: str,
    cfg: ChunkingConfig,
    *,
    page: int | None = None,
) -> list[Chunk]:
    """Chunk a full document into budgeted, overlap-carrying chunks.

    Chunks carry `section_id` (a stable per-paper identifier) and
    `is_first_in_section` / `is_last_in_section` flags so downstream code
    can stitch chunks of a section back together.
    """
    sections = split_sections(text)
    # Prefix with the page so section ids stay unique across multi-page docs.
    prefix = f"p{page}_" if page is not None else ""
    all_chunks: list[Chunk] = []
    for i, sec in enumerate(sections):
        sid = f"{prefix}sec{i}"
        sec.section_id = sid
        sec_chunks = _pack_into_chunks(
            sec.text,
            max_tokens=cfg.max_tokens,
            overlap_tokens=cfg.overlap_tokens,
            min_tokens=cfg.min_chunk_tokens,
            heading=sec.heading,
            level=sec.level,
            section_id=sid,
        )
        if sec_chunks:
            sec_chunks[0].is_first_in_section = True
            sec_chunks[-1].is_last_in_section = True
        all_chunks.extend(sec_chunks)
    if page is not None:
        for c in all_chunks:
            c.page = page
    return all_chunks
