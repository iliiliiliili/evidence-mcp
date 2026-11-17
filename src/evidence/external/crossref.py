"""Crossref provider (generous rate limits, no key required).

Useful for DOI resolution and bibliometric intent (citations / references).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE = "https://api.crossref.org"


def _norm_item(item: dict[str, Any]) -> dict[str, Any]:
    raw_title = item.get("title")
    if isinstance(raw_title, list):
        title = raw_title[0] if raw_title else ""
    else:
        title = raw_title or ""
    authors = [
        (a.get("family") or "") + " " + (a.get("given") or "").strip()
        for a in item.get("author", [])
    ]
    authors = [a.strip() for a in authors if a.strip()]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        date = (item.get(key) or {}).get("date-parts") or [[]]
        if date and date[0] and date[0][0]:
            year = int(date[0][0])
            break
    return {
        "id": item.get("DOI", ""),
        "doi": item.get("DOI"),
        "title": title,
        "year": year,
        "authors": authors,
        "is-referenced-by-count": item.get("is-referenced-by-count", 0),
        "reference-count": item.get("reference-count", 0),
    }


def search(
    query: str,
    *,
    rows: int = 10,
    mailto: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"query": query, "rows": rows}
    if mailto:
        params["mailto"] = mailto
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{BASE}/works", params=params)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
    return [_norm_item(i) for i in items]


def get_work(doi: str, *, mailto: str | None = None) -> dict[str, Any] | None:
    params = {"mailto": mailto} if mailto else None
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{BASE}/works/{doi}", params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _norm_item(resp.json().get("message", {}))
