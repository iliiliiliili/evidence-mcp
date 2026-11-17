"""Shared helpers for building `EvidenceItem` records from the store.

Every node that emits evidence uses these so provenance + dedupe-by-hash are
consistent across dense/sparse/graph/visual/external paths (PLAN.MD stage 10).
"""

from __future__ import annotations

import hashlib
from typing import Any

from ...indexes.store import Store
from ..state import EvidenceItem, SourceKind


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


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
        "title": (paper.title if paper else ""),
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
) -> EvidenceItem:
    """Build an EvidenceItem record from an external work (OpenAlex/S2/Crossref)."""
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
    }
