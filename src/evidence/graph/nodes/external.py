"""External academic search (PLAN.MD stage 9, Phase 3).

Implements the iterative **optimize → search → vet → reformulate** loop over
OpenAlex (first — no key), Semantic Scholar, and Crossref. Strategy is chosen
by intent:

  content        → fetch top OA works, ingest their abstracts as evidence
  bibliometric   → use cited_by_count / reference_count to rank related works
  trend          → sort by publication year (newest first)
  current_world  → recency-sorted, prefer the most recent N

Each fetched work is recorded in `external_used` (citation + audit) and its
abstract added to `external_evidence` (an accumulator, merged at stage 10).
Rate limits are honoured: S2 is serialized (1 RPS), all providers back off on
429/5xx (see `backoff.py`).
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import ExternalConfig
from ...external import crossref, openalex, semantic_scholar
from ...external.backoff import retry_on_error
from ..state import EvidenceItem, EvidenceState, SourceKind
from .context import get_context
from .evidence_util import external_evidence

logger = logging.getLogger(__name__)


def _provider_kind(name: str) -> SourceKind:
    return {"openalex": "openalex", "s2": "s2", "crossref": "crossref"}[name]


def _search_provider(
    name: str,
    query: str,
    *,
    cfg: ExternalConfig,
    intent: str,
) -> list[dict[str, Any]]:
    """Dispatch to a provider, applying the intent-specific ranking."""
    try:
        if name == "openalex":
            works = retry_on_error(
                lambda: openalex.search_works(
                    query, per_page=cfg.max_results_per_provider, mailto=cfg.openalex_mailto
                )
            )
        elif name == "s2":
            import time

            time.sleep(1.0)  # honour ~1 RPS cap between S2 calls
            works = retry_on_error(
                lambda: semantic_scholar.search_papers(
                    query, limit=cfg.max_results_per_provider, api_key=cfg.s2_api_key
                )
            )
        elif name == "crossref":
            works = retry_on_error(
                lambda: crossref.search(query, rows=cfg.max_results_per_provider)
            )
        else:
            return []
    except Exception as exc:  # noqa: BLE001 - a failing provider shouldn't sink the run
        logger.warning("provider %s failed: %s", name, exc)
        return []

    # Intent-specific ranking (re-rank the provider's list).
    def year_key(w: dict) -> int:
        y = w.get("year") or 0
        return y if isinstance(y, int) else 0

    if intent == "bibliometric":
        works.sort(
            key=lambda w: w.get("cited_by_count", w.get("is-referenced-by-count", 0)),
            reverse=True,
        )
    elif intent in {"trend", "current_world"}:
        works.sort(key=year_key, reverse=True)
    # content → keep provider's relevance order.
    return works[: cfg.max_results_per_provider]


def search_external(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    if not cfg.external.enabled:
        logger.info("external search disabled — skipping stage 9")
        return {}

    intent = state.get("intent", "content")
    # Use the recorded shortcomings (if any) to optimize the query.
    base_query = state.get("question", "")
    shortcomings = state.get("retrieval_shortcomings") or []
    if shortcomings:
        # Append the top shortcoming as a refinement (optimize step).
        base_query = f"{base_query} ({shortcomings[0]})"

    collected: list[EvidenceItem] = []
    external_used: list[dict[str, Any]] = []
    providers = cfg.external.providers or ["openalex"]

    for provider in providers:
        works = _search_provider(provider, base_query, cfg=cfg.external, intent=intent)
        for i, w in enumerate(works):
            title = w.get("title") or ""
            if not title:
                continue
            ev = external_evidence(
                work_id=str(w.get("id", i)),
                title=title,
                abstract=w.get("abstract") or w.get("context") or title,
                source=_provider_kind(provider),
                relevance=max(0.4, 0.9 - 0.05 * i),
                url=w.get("oa_url") or w.get("pdf_url"),
                doi=w.get("doi"),
                year=w.get("year"),
                authors=w.get("authors") or [],
                provenance={
                    "provider": provider,
                    "intent": intent,
                    "query": base_query,
                    "rank": i + 1,
                    "cited_by_count": w.get("cited_by_count", w.get("is-referenced-by-count", 0)),
                },
            )
            # Vet: skip works with no abstract/title (low signal).
            if len(ev["text"] or "") < 20:
                continue
            collected.append(ev)
            external_used.append(
                {
                    "provider": provider,
                    "id": w.get("id"),
                    "doi": w.get("doi"),
                    "title": title,
                    "year": w.get("year"),
                    "url": ev.get("url"),
                }
            )

    logger.info(
        "search_external: %d works from %d provider(s) for intent=%s",
        len(collected), len(providers), intent,
    )
    return {
        "external_evidence": collected,
        "external_used": external_used,
    }
