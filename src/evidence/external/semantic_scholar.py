"""Semantic Scholar provider (rate-limited ~1 RPS without a key).

Batch endpoints (500 IDs) are used where possible to stay under the cap.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.semanticscholar.org/graph/v1"
_DEFAULT_FIELDS = (
    "paperId,title,year,abstract,authors,externalIds,citationCount,referenceCount"
)


def _norm_paper(p: dict[str, Any]) -> dict[str, Any]:
    year = p.get("year")
    if year is None and p.get("publicationDate"):
        year = int(str(p["publicationDate"])[:4])
    return {
        "id": p.get("paperId", ""),
        "doi": p.get("externalIds", {}).get("DOI"),
        "title": p.get("title") or "",
        "year": year,
        "authors": [a.get("name") for a in p.get("authors", []) if a.get("name")],
        "abstract": p.get("abstract"),
        "citations": p.get("citations", 0),
        "referenceCount": p.get("referenceCount", 0),
    }


def search_papers(
    query: str,
    *,
    limit: int = 10,
    api_key: str | None = None,
    fields: str = _DEFAULT_FIELDS,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": limit, "fields": fields}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.get(f"{BASE}/paper/search", params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    return [_norm_paper(p) for p in data]


def get_paper(
    paper_id: str,
    *,
    api_key: str | None = None,
    fields: str = _DEFAULT_FIELDS + ",tldr",
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"fields": fields}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.get(f"{BASE}/paper/{paper_id}", params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _norm_paper(resp.json())


def get_citations(
    paper_id: str,
    *,
    limit: int = 100,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Citation contexts (PLAN.MD stage 9: "S2 citation contexts")."""
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"limit": limit, "fields": "paperId,title,year,abstract,authors"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.get(f"{BASE}/paper/{paper_id}/citations", params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    out = []
    for c in data:
        p = c.get("citingPaper") or {}
        if p:
            rec = _norm_paper(p)
            rec["context"] = c.get("context")
            out.append(rec)
    return out


def citation_contexts(
    paper_id: str,
    *,
    limit: int = 10,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Alias for `get_citations` — works that cite `paper_id` plus the citation
    sentence (the `context` field), used as the AI/ML fallback in
    `expand_citations` (new plan)."""
    return get_citations(paper_id, limit=limit, api_key=api_key, timeout=timeout)
