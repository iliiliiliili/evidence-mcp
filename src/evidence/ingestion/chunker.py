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


def count_tokens(text: str) -> int:
    """Best-effort token count: tiktoken if present, else whitespace heuristic."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001 - optional dependency
        # ~4 chars/token heuristic for English-ish technical text.
        return max(1, len(text) // 4)


def split_sections(text: str) -> list[Section]:
    """Split raw text into (heading, body) sections on common heading markers."""
    sections: list[Section] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(Section(heading=current_heading, text=body))

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m_md = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
        if m_md:
            flush()
            current_lines = []
            current_heading = m_md.group(2).strip()
            continue
        m_latex = re.match(r"^\\(section|subsection|subsubsection|paragraph)\*?\{(.*)\}", line)
        if m_latex:
            flush()
            current_lines = []
            current_heading = m_latex.group(2).strip()
            continue
        current_lines.append(raw_line)
    flush()

    # If we found no headings at all, wrap the whole thing as one section.
    if not sections:
        sections.append(Section(heading="", text=text.strip()))
    return sections


def _pack_into_chunks(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    heading: str,
    level: int,
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
    """Chunk a full document into budgeted, overlap-carrying chunks."""
    sections = split_sections(text)
    all_chunks: list[Chunk] = []
    for sec in sections:
        all_chunks.extend(
            _pack_into_chunks(
                sec.text,
                max_tokens=cfg.max_tokens,
                overlap_tokens=cfg.overlap_tokens,
                min_tokens=cfg.min_chunk_tokens,
                heading=sec.heading,
                level=sec.level,
            )
        )
    if page is not None:
        for c in all_chunks:
            c.page = page
    return all_chunks
