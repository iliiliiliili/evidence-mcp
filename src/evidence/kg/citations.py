"""Intra-corpus citation resolution (PLAN.MD offline pipeline, step 5).

Match each paper's reference list to *other* corpus papers via DOI fuzzy
matching (and, as a fallback, title similarity). This produces the CITES edges
that enable co-citation traversal at query time (PLAN.MD stage 5 + intent
"bibliometric").

Design: pure / local-first. We parse candidate DOIs + reference titles from the
raw paper text, then match them against the known corpus (by DOI or normalized
title). No network is required; Crossref/OpenAlex lookups are optional and only
used to *disambiguate* DOIs when a match is ambiguous.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DOI = re.compile(r"10\.\d{4,9}/[^\s\]\},;\"'<>]+", re.IGNORECASE)
# A reference line: "Author(s). Title. Venue. Year." — we capture the title.
_REF_LINE = re.compile(
    r"^\s*[\[\(]?\d{1,3}[)\.\s]*"  # leading numeric marker (optional)
    r"([\w][\w.,\-'\s]{15,160})\."  # the title (15-160 chars, ends with a period)
    r"\s*(\d{4})?\s*$",  # optional trailing year
)


@dataclass
class CitationLink:
    source_paper_id: str
    target_paper_id: str
    target_doi: str | None = None
    target_title: str | None = None
    confidence: float = 0.0
    method: str = "doi"  # "doi" | "title" | "doi+title"
    evidence: dict = field(default_factory=dict)


def norm_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    doi = doi.rstrip(".,;:)\u2019'\"")
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return doi.strip()


def extract_dois(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _DOI.finditer(text):
        d = norm_doi(m.group(0))
        if d and d not in seen:
            seen[d] = None
    return list(seen)


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _candidate_titles(text: str) -> list[str]:
    """Heuristically pull reference-like titles out of a paper's body.

    We look at the trailing 'References' / 'Bibliography' section if present,
    else scan all lines for reference-shaped sentences. This is intentionally
    conservative to avoid false positives in the body text.
    """
    # Prefer the explicit references section.
    m = re.search(
        r"(?im)^\s*(references|bibliography)\s*:?\s*$(.*?)(?=^\s*(appendix|acknowledg|conclusion|conclusion)\b|$)",
        text,
        re.DOTALL,
    )
    scope = m.group(2) if m else text[-8000:]  # else the tail of the document

    titles: list[str] = []
    seen: set[str] = set()
    for line in scope.splitlines():
        line = line.strip()
        if len(line) < 20 or len(line) > 260:
            continue
        rm = _REF_LINE.match(line)
        title = (rm.group(1).strip() if rm else line)
        key = normalize_title(title)
        if len(key) < 12 or key in seen:
            continue
        seen.add(key)
        titles.append(title)
    return titles


def match_corpus(
    *,
    source_paper_id: str,
    source_dois: list[str],
    source_titles: list[str],
    corpus: list[dict],
    similarity_threshold: float = 0.82,
) -> list[CitationLink]:
    """Match a source paper's references against the corpus of known papers.

    `corpus` is a list of dicts, each with at least:
        paper_id, title, doi (optional)
    Returns the resolved CitationLink list (possibly empty).
    """
    links: list[CitationLink] = []
    seen_targets: set[str] = set()
    corpus_by_doi: dict[str, dict] = {}
    corpus_by_title: dict[str, dict] = {}
    for c in corpus:
        if c.get("doi"):
            corpus_by_doi[norm_doi(c["doi"])] = c
        key = normalize_title(c.get("title", ""))
        if key:
            corpus_by_title[key] = c

    # 1) DOI matching (high confidence).
    for doi in source_dois:
        nd = norm_doi(doi)
        target = corpus_by_doi.get(nd)
        if target is None:
            continue
        if target["paper_id"] == source_paper_id:
            continue  # self-cite
        if target["paper_id"] in seen_targets:
            continue
        seen_targets.add(target["paper_id"])
        links.append(
            CitationLink(
                source_paper_id=source_paper_id,
                target_paper_id=target["paper_id"],
                target_doi=nd,
                target_title=target.get("title"),
                confidence=1.0,
                method="doi",
                evidence={"doi": nd},
            )
        )

    # 2) Title-similarity matching (fill in any references that have no DOI hit).
    matched_titles = {normalize_title(link.target_title or "") for link in links}
    for title in source_titles:
        if normalize_title(title) in matched_titles:
            continue
        key = normalize_title(title)
        if not key:
            continue
        best: dict | None = None
        best_ratio = 0.0
        for tkey, target in corpus_by_title.items():
            if target["paper_id"] == source_paper_id:
                continue
            ratio = difflib.SequenceMatcher(None, key, tkey).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = target
        if best is not None and best_ratio >= similarity_threshold:
            if best["paper_id"] in seen_targets:
                continue
            seen_targets.add(best["paper_id"])
            links.append(
                CitationLink(
                    source_paper_id=source_paper_id,
                    target_paper_id=best["paper_id"],
                    target_doi=norm_doi(best.get("doi") or "") or None,
                    target_title=best.get("title"),
                    confidence=round(float(best_ratio), 3),
                    method="title",
                    evidence={"query_title": title, "ratio": best_ratio},
                )
            )
    return links


def resolve_citations_for_corpus(
    papers: list[dict], *, threshold: float = 0.82
) -> list[CitationLink]:
    """Resolve citations for a whole corpus (each paper vs. the rest).

    `papers`: list of dicts with `paper_id`, `title`, `text`, `doi` (optional),
    `references` (optional raw reference-list string).
    """
    corpus = [
        {"paper_id": p["paper_id"], "title": p.get("title", ""), "doi": p.get("doi")}
        for p in papers
    ]
    all_links: list[CitationLink] = []
    for p in papers:
        text = p.get("text", "")
        ref_text = p.get("references") or text
        dois = extract_dois(ref_text)
        titles = _candidate_titles(ref_text)
        links = match_corpus(
            source_paper_id=p["paper_id"],
            source_dois=dois,
            source_titles=titles,
            corpus=corpus,
            similarity_threshold=threshold,
        )
        all_links.extend(links)
    return all_links


def co_citation_pairs(links: list[CitationLink]) -> dict[tuple[str, str], int]:
    """Count shared-reference (co-citation) pairs across the corpus.

    Two papers are co-cited when they both cite the same target. This powers
    the "bibliometric" intent (PLAN.MD stage 9) and graph traversal.
    """
    cites: dict[str, list[str]] = {}
    for link in links:
        cites.setdefault(link.source_paper_id, []).append(link.target_paper_id)
    pairs: dict[tuple[str, str], int] = {}
    sources = list(cites)
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = set(cites[sources[i]]), set(cites[sources[j]])
            shared = a & b
            if shared:
                key = tuple(sorted((sources[i], sources[j])))
                pairs[key] = pairs.get(key, 0) + len(shared)
    return pairs
