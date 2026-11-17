"""OpenAlex author validation (PLAN.MD offline pipeline, step 4).

Resolve author names against OpenAlex author IDs to dedupe "J. Smith"
collisions. Local-first: works offline by normalizing + grouping, and only
touches the network (best-effort) when online enrichment is requested.

OpenAlex author endpoint:
    GET https://api.openalex.org/authors?search=<name>
Returns a normalized `AuthorIdentity` per distinct author.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BASE = "https://api.openalex.org"

_STOP = {"jr", "sr", "ii", "iii", "iv", "junior", "senior", "phd", "prof", "dr"}


@dataclass
class AuthorIdentity:
    """A deduplicated author identity (PLAN.MD offline pipeline, step 4)."""

    display_name: str
    normalized: str
    openalex_id: str | None = None
    affiliations: list[str] = field(default_factory=list)
    work_count: int | None = None
    matches: list[str] = field(default_factory=list)  # raw names folded into it


def _norm_name(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(r"[^a-z\s\-']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    parts = [p for p in n.split(" ") if p and p not in _STOP]
    return " ".join(parts)


def group_by_identity(names: list[str]) -> list[AuthorIdentity]:
    """Fold a list of raw author names into deduplicated identities (offline).

    Groups by normalized name so "J. Smith", "John Smith", and "J Smith" land
    in the same bucket when their normalized forms match.
    """
    buckets: dict[str, AuthorIdentity] = {}
    order: list[str] = []
    for name in names:
        if not name or not name.strip():
            continue
        key = _norm_name(name)
        if not key:
            continue
        if key in buckets:
            ident = buckets[key]
            if name.strip() not in ident.matches:
                ident.matches.append(name.strip())
        else:
            ident = AuthorIdentity(
                display_name=name.strip(),
                normalized=key,
                matches=[name.strip()],
            )
            buckets[key] = ident
            order.append(key)
    return [buckets[k] for k in order]


def _search_openalex(name: str, mailto: str | None) -> dict | None:
    """Best-effort OpenAlex lookup for one author. Returns None on any failure."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - optional
        return None
    params: dict = {"per-page": 1, "search": name}
    if mailto:
        params["mailto"] = mailto
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{BASE}/authors", params=params)
            if resp.status_code != 200:
                return None
            results = resp.json().get("results") or []
            if not results:
                return None
            return results[0]
    except Exception as exc:  # noqa: BLE001 - never block the pipeline on this
        logger.debug("OpenAlex author lookup failed for %r: %s", name, exc)
        return None


def validate_authors(
    names: list[str],
    *,
    mailto: str | None = None,
    online: bool = False,
) -> list[AuthorIdentity]:
    """Deduplicate author names; optionally enrich with OpenAlex IDs.

    With `online=False` (default) this is fully local. With `online=True` it
    makes a best-effort OpenAlex call per distinct identity to attach a stable
    `openalex_id`, affiliation, and work count.
    """
    identities = group_by_identity(names)
    if not online:
        return identities

    for ident in identities:
        rec = _search_openalex(ident.display_name, mailto)
        if rec is None:
            continue
        ident.openalex_id = rec.get("id")
        ident.work_count = rec.get("works_count")
        insts = rec.get("institutions") or rec.get("last_known_institutions") or []
        for inst in insts[:3]:
            display = inst.get("display_name")
            if display and display not in ident.affiliations:
                ident.affiliations.append(display)
    return identities
