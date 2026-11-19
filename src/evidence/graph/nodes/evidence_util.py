"""Shared helpers for building `EvidenceItem` records from the store.

Every node that emits evidence uses these so provenance + dedupe-by-hash are
consistent across dense/sparse/graph/visual/external paths (PLAN.MD stage 10).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ...indexes.store import Store
from ..state import EvidenceItem, SourceKind


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


# Multi-letter function words (case-insensitive). Used to tell a TITLE (which
# almost always contains one — "…FOR FINANCIAL SENTIMENT ANALYSIS…",
# "…and Re-Identification in Dynamic Scenes") from an AUTHOR NAME LIST (which
# rarely does — "Giorgos Iacovides, Oliver Hannibal"). Single-letter tokens are
# excluded so middle initials ("Alice A. Smith") don't count as function words.
_TITLE_FN_WORDS = {
    "for", "the", "a", "an", "and", "or", "of", "to", "in", "on", "by", "from",
    "with", "at", "as", "is", "are", "was", "were", "be", "its", "our", "their",
    "this", "that", "these", "those", "we", "you", "all", "any", "no", "not",
    "via", "per", "each",
}

# Watermark / journal-header substrings (matched case-insensitively) to skip.
_TITLE_BOILERPLATE = (
    "for review only", "review only", "draft manuscript", "draft copy",
    "final copy", "copyright", "all rights reserved", "camera-ready",
    "camera ready", "accepted manuscript", "preprint", "proof copy",
    "reproduced with permission", "confidential", "internal use",
    "not for distribution", "under review", "in review", "submitted to",
    "submitted for", "to appear in", "appeared in", "published in",
    "accepted for", "journal of", "transactions on", "proceedings of",
    "in proceedings", "proceedings", "manuscript", "proof",
)

# Section labels (matched case-insensitively, as a whole line or line-prefix).
_TITLE_SECTIONS = (
    "abstract", "introduction", "conclusion", "conclusions", "references",
    "bibliography", "keywords", "index terms", "indexing terms", "biographies",
    "notes", "acknowledgments", "acknowledgements", "appendix", "appendices",
    "citation", "citations", "funding", "data availability", "supplementary",
    "footnotes", "cited references",
)

# Affiliation / institutional keywords — a line containing one is never the
# title (it's an author affiliation or an address).
_TITLE_AFFIL = (
    "university", "institute", "college", "school of", "department", "faculty",
    "laboratory", "corporation", "limited", "hospital", "clinic", "company",
    "division", "institute of", "research center", "research centre",
    "center for", "centre for", "department of", "branch of",
)

# Author markers: superscript digits and common affiliation/correspondence
# symbols that appear on author/affiliation lines but essentially never in a
# title.
_TITLE_AUTHOR_SYMBOLS = "∗¹²³⁴⁵⁶⁷⁸⁹⁰†‡§¶"

# A "name pair" — two consecutive words each starting with a capital
# ("Giorgos Iacovides", "Alice Smith"). Two or more on one line is a strong
# author-list signal.
_NAME_PAIR_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)\b")


def _is_weak_title_stem(title: str) -> bool:
    """True if `title` looks like a fallback (filename stem), not a real title.

    Heuristics (cheap, no model):
      - empty, or
      - no whitespace (i.e. a bare id / filename like `11299`), or
      - all-alphanumeric (no punctuation) AND ≤ 3 words.

    Anything with spaces + real words (e.g. "FinLLM-KG: Knowledge …") passes
    as a real title.
    """
    t = (title or "").strip()
    if not t:
        return True
    if " " not in t:
        return True
    if t.isalnum():
        return len(t.split()) <= 3
    return False


def _title_fn_count(line: str) -> int:
    """Number of multi-letter function words in `line`.

    Single-letter tokens are skipped so a middle initial ("Alice A. Smith")
    does not count as the function word "a".
    """
    return sum(
        1
        for w in line.split()
        if len(w.strip(" .,;:()[]{}'\"-")) > 1
        and w.strip(" .,;:()[]{}'\"-").lower() in _TITLE_FN_WORDS
    )


def _title_digit_tokens(line: str) -> int:
    return sum(1 for w in line.split() if w.strip(" .,;:").isdigit())


def _line_is_boilerplate(line: str) -> bool:
    low = " " + line.lower() + " "
    if any(p in low for p in _TITLE_BOILERPLATE):
        return True
    # Volume / issue lines ("Vol. 12, Issue 3, 2024") carry two or more bare
    # digit tokens.
    if _title_digit_tokens(line) >= 2:
        return True
    return False


def _line_is_section_label(line: str) -> bool:
    s = line.strip().rstrip(".:").lower()
    return any(
        s == lbl or s.startswith(lbl + " ") or s.startswith(lbl + ":")
        for lbl in _TITLE_SECTIONS
    )


def _line_is_affiliation(line: str) -> bool:
    low = line.lower()
    return any(k in low for k in _TITLE_AFFIL)


def _line_is_author_list(line: str) -> bool:
    """Multi-name author line: ≥1 comma, ≥2 name pairs, no function word.

    The "no function word" requirement is the key discriminator that lets a
    title with commas ("Object Detection, Tracking, and Re-Identification")
    survive while an author list ("Alice Smith, Bob Jones") is detected.
    """
    if any(c in line for c in _TITLE_AUTHOR_SYMBOLS):
        return True
    return (
        line.count(",") >= 1
        and len(_NAME_PAIR_RE.findall(line)) >= 2
        and _title_fn_count(line) == 0
    )


def _has_lower(line: str) -> bool:
    return any(c.islower() for c in line if c.isalpha())


def _is_short_name_like(line: str) -> bool:
    """A 2-3 word line with no function word and no digits — an author NAME.

    Used only to *stop* title collection once we've started, and only when the
    line is mixed-case (an ALL-CAPS short line like "FLOW FIELDS" is a title
    continuation in an all-caps paper, not a name).
    """
    words = [w for w in line.split() if w.strip()]
    return (
        2 <= len(words) <= 3
        and _title_fn_count(line) == 0
        and _title_digit_tokens(line) == 0
    )


def derive_title_from_text(text: str) -> str:
    """Derive a paper title from the leading text of a document (pure).

    Line-based: walk the first-page lines in order, skipping watermarks
    (e.g. "FOR REVIEW ONLY"), journal headers, section labels, and
    volume/issue lines, then collect the contiguous run of title lines. Stop
    at the first author line (≥2 name pairs, no function word), affiliation
    line, section label, or mixed-case short name line.

    Works for both all-caps (IEEE, etc.) and title-case titles, and for
    titles that contain commas. Returns `""` when nothing plausible is found;
    callers should then keep their existing fallback (stored title / filename
    stem). This is the single source of the heuristic, shared by both the
    store-side and the ingest-time derivation.
    """
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    parts: list[str] = []
    for line in lines:
        if (
            _line_is_boilerplate(line)
            or _line_is_section_label(line)
            or _line_is_affiliation(line)
        ):
            if parts:
                break
            continue
        if _line_is_author_list(line):
            break
        # A mixed-case short name-like line (2-3 words, no function word, no
        # digits) is an author NAME ("John Doe") — stop. All-caps short lines
        # (e.g. "FLOW FIELDS") are kept as title continuations.
        if parts and _is_short_name_like(line) and _has_lower(line):
            break
        parts.append(line)
        if len(" ".join(parts).split()) > 18:
            break
    title = " ".join(parts).strip(" .,;:")
    return title if len(title.split()) >= 2 else ""


def _derived_paper_title(paper, store) -> str:
    """Best-effort paper title derived from the paper's first chunk text.

    Thin wrapper over `derive_title_from_text` for the store side: pulls the
    paper's first chunk and runs the shared heuristic. Returns `""` when the
    paper has no chunks or nothing looks like a title.
    """
    if paper is None or store is None:
        return ""
    chunks = store.chunks_for_paper(paper.paper_id)
    if not chunks:
        return ""
    return derive_title_from_text(chunks[0].text)


def group_from_path(path: str | None) -> str:
    """Derive a paper's *group* from its on-disk path.

    Convention: a paper's group is the name of its immediate parent folder,
    relative to the corpus root (which we don't track here — the corpus root
    is where the user dropped the files, and we don't want to hard-code it).
    So we simply use the immediate parent folder's basename. Papers that
    sit directly in the corpus root get "(root)" — i.e. "ungrouped".

    Examples (corpus at `~/langchain/review/corpus`):
      corpus/papers/a.pdf           -> "papers"
      corpus/ml/finance/x.pdf       -> "finance"
      corpus/robotics/y.pdf         -> "robotics"
      corpus/loner.pdf              -> "(root)"

    This is purely structural — no metadata change required. To group papers
    for comparative analysis, the user simply moves them into subfolders.
    """
    if not path:
        return ""
    try:
        parent = Path(path).parent
        name = parent.name
        return name or "(root)"
    except Exception:  # noqa: BLE001 - never let a bad path break retrieval
        return ""


def _best_title(paper, store: Store) -> str:
    """Resolve a display title for a paper.

    Uses the stored title when it looks like a real title; otherwise (a bare
    filename stem) derives one from the paper's first chunk text. Falls back
    to the stored title if nothing plausible is found.
    """
    if paper is None:
        return ""
    if _is_weak_title_stem(paper.title):
        derived = _derived_paper_title(paper, store)
        if derived:
            return derived
    return paper.title or ""


def evidence_from_chunk(
    store: Store,
    chunk_id: str,
    *,
    source: SourceKind,
    relevance: float,
    provenance: dict[str, Any] | None = None,
) -> EvidenceItem | None:
    """Build a fully-populated EvidenceItem record for a stored chunk (or None)."""
    rec = store.get_chunk(chunk_id)
    if rec is None:
        return None
    paper = store.get_paper(rec.paper_id)
    ev: EvidenceItem = {
        "chunk_id": rec.chunk_id,
        "paper_id": rec.paper_id,
        "title": _best_title(paper, store),
        "page": rec.page or 0,
        "text": rec.text,
        "source": source,
        "relevance": float(relevance),
        "provenance": dict(provenance or {}),
        "doi": paper.doi if paper else None,
        "url": None,
        "year": paper.year if paper else None,
        "authors": list(paper.authors) if paper else [],
        "verified": False,
        "group": group_from_path(paper.path if paper else None),
    }
    return ev


def dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Dedupe by chunk hash (near-identical text) keeping the highest relevance.

    This is the "chunk hash" dedupe from PLAN.MD stage 10. Semantic near-dupe
    filtering can be layered on later; hash dedupe is deterministic and free.
    """
    best: dict[str, EvidenceItem] = {}
    for ev in items:
        text = (ev.get("text") or "").strip()
        key = ev.get("chunk_id") or (chunk_hash(text) if text else id(ev))
        cur = best.get(key)
        if cur is None or ev.get("relevance", 0.0) > cur.get("relevance", 0.0):
            best[key] = ev
    # Highest relevance first for downstream stable ordering.
    return sorted(
        best.values(),
        key=lambda e: e.get("relevance", 0.0),
        reverse=True,
    )


def external_evidence(
    *,
    work_id: str,
    title: str,
    abstract: str,
    source: SourceKind,
    relevance: float = 0.6,
    url: str | None = None,
    doi: str | None = None,
    year: int | None = None,
    authors: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    group: str = "external",
) -> EvidenceItem:
    """Build an EvidenceItem record from an external work (OpenAlex/S2/Crossref).

    External works have no on-disk path, so their default group is the literal
    `"external"`. Override via the `group` kwarg if you want to bucket them
    under a named group (e.g. "related-work").
    """
    return {
        "chunk_id": f"ext::{source}::{work_id}",
        "paper_id": f"ext::{work_id}",
        "title": title,
        "page": 0,
        "text": abstract or title,
        "source": source,
        "relevance": float(relevance),
        "provenance": dict(provenance or {}),
        "doi": doi,
        "url": url,
        "year": year,
        "authors": list(authors or []),
        "verified": False,
        "group": group,
    }
