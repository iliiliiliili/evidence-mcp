"""OpenAlex provider (no API key; uses the polite pool via `mailto`).

OpenAlex is the preferred bulk provider (PLAN.MD stage 9: "OpenAlex first —
no key needed"). Returns normalized work records.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.openalex.org"


def _norm_work(w: dict[str, Any]) -> dict[str, Any]:
    authors = [a["author"]["display_name"] for a in w.get("authorships", [])
               if a.get("author", {}).get("display_name")]
    year = w.get("publication_year")
    if year is None and w.get("from_publication_date"):
        year = int(str(w["from_publication_date"])[:4])
    return {
        "id": w.get("id", ""),
        "doi": w.get("doi"),
        "title": w.get("title") or w.get("display_name") or "",
        "year": year,
        "authors": authors,
        "oa_url": (w.get("open_access") or {}).get("oa_url"),
        "pdf_url": (w.get("open_access") or {}).get("pdf_url"),
        "cited_by_count": w.get("cited_by_count", 0),
    }


def _reconstruct_abstract(inv: dict[str, list[str]] | None) -> str:
    """OpenAlex stores abstracts as an inverted index; reconstruct plain text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, indices in inv.items():
        for i in indices:
            positions.append((i, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def search_works(
    query: str,
    *,
    per_page: int = 10,
    mailto: str | None = None,
    sort: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Search works. `sort` follows OpenAlex, e.g.
    ``cited_by_count:desc`` (most cited) or ``publication_date:desc`` (newest)."""
    params: dict[str, Any] = {"per-page": per_page, "search": query}
    if sort:
        params["sort"] = sort
    if mailto:
        params["mailto"] = mailto
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{BASE}/works", params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    out = []
    for w in results:
        rec = _norm_work(w)
        rec["abstract"] = _reconstruct_abstract(w.get("abstract_inverted_index"))
        out.append(rec)
    return out


def get_work(work_id: str, *, mailto: str | None = None) -> dict[str, Any] | None:
    url = work_id if work_id.startswith("http") else f"{BASE}/works/{work_id}"
    params = {"mailto": mailto} if mailto else None
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        w = resp.json()
    rec = _norm_work(w)
    rec["abstract"] = _reconstruct_abstract(w.get("abstract_inverted_index"))
    return rec

def works_citing(
    doi_or_id: str,
    *,
    limit: int = 10,
    mailto: str | None = None,
    sort: str = "cited_by_count:desc",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Works that CITE `doi_or_id` (forward citations, OpenAlex)."""
    return _cites_query(doi_or_id, direction="citing", limit=limit, mailto=mailto,
                        sort=sort, timeout=timeout)


def referenced_by(
    doi_or_id: str,
    *,
    limit: int = 10,
    mailto: str | None = None,
    sort: str = "publication_date:desc",
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Works that `doi_or_id` CITES (backward references, OpenAlex)."""
    return _cites_query(doi_or_id, direction="cited", limit=limit, mailto=mailto,
                        sort=sort, timeout=timeout)


def _cites_query(
    doi_or_id: str,
    *,
    direction: str,
    limit: int,
    mailto: str | None,
    sort: str,
    timeout: float,
) -> list[dict[str, Any]]:
    """Query OpenAlex for works citing / cited-by, resolving a DOI to a work PK.

    Strategy: if we have a DOI, use the DOI-based cite filter
    (``filter=cites.id:doi:<doi>``). Otherwise, fetch the work to learn its
    OpenAlex PK, then filter by ``cites.id:W<pk>``.
    """
    doi = doi_or_id.strip()
    if doi.startswith("http"):
        if doi.lower().startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        else:
            # A full URL that is not a DOI — try to use as-is.
            doi = doi
    is_doi = doi.startswith("10.")

    with httpx.Client(timeout=timeout) as client:
        if is_doi:
            prefix = "cites.id" if direction == "citing" else "references.id"
            params: dict[str, Any] = {
                "per-page": limit,
                "sort": sort,
                "filter": f"{prefix}:doi:{doi}",
            }
            if mailto:
                params["mailto"] = mailto
            resp = client.get(f"{BASE}/works", params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        else:
            # Need the OpenAlex PK: fetch the work.
            work = get_work(doi_or_id, mailto=mailto)
            if not work:
                return []
            pk = work.get("id", "")
            if pk and not pk.startswith("http"):
                pk = pk  # may already be W-pk
            # Derive a usable id for the filter (W-pk form).
            pk_field = pk
            if pk.startswith("http"):
                pk_field = pk.rstrip("/").split("/")[-1]
            prefix = "cites.id" if direction == "citing" else "references.id"
            params = {"per-page": limit, "sort": sort, "filter": f"{prefix}:{pk_field}"}
            if mailto:
                params["mailto"] = mailto
            resp = client.get(f"{BASE}/works", params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])

    out = []
    for w in results:
        rec = _norm_work(w)
        rec["abstract"] = _reconstruct_abstract(w.get("abstract_inverted_index"))
        out.append(rec)
    return out