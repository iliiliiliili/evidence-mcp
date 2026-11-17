"""Stage 4 — Cross-encoder reranking.

Re-scores the accumulated evidence against the (primary) question using the
best available reranker (cross-encoder → LLM → heuristic) and truncates to
`retrieval.keep_after_rerank`. Relevance values are rewritten to the rerank
scores so downstream gates see a calibrated signal.
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise into [0,1] (robust to absolute scale of the scorer)."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def rerank(state: EvidenceState) -> dict:
    ctx = get_context()
    # Rerank operates on the main text-path evidence (pre-merge).
    evidence: list[EvidenceItem] = list(state.get("main_evidence") or [])
    if not evidence:
        return {}

    # Use the strongest query as the rerank anchor: prefer the original question.
    anchor = state.get("question") or (state.get("text_queries") or [""])[0]

    passages = [ev.get("text", "") for ev in evidence]
    raw_scores = ctx.models.rerank(anchor, passages)
    if len(raw_scores) != len(passages):
        logger.warning("rerank score length mismatch; using uniform scores")
        raw_scores = [0.5] * len(passages)

    norm = _normalise(raw_scores)

    # Re-score + re-order, keep top-K.
    paired = list(zip(evidence, norm, strict=True))
    paired.sort(key=lambda p: p[1], reverse=True)
    keep = ctx.cfg.retrieval.keep_after_rerank
    kept = paired[:keep]

    out: list[EvidenceItem] = []
    for ev, score in kept:
        new_ev = dict(ev)
        new_ev["relevance"] = float(score)
        prov = dict(ev.get("provenance") or {})
        prov["rerank_score"] = float(score)
        new_ev["provenance"] = prov
        out.append(new_ev)

    logger.debug("rerank: kept %d/%d chunks", len(out), len(evidence))
    return {
        "evidence": out,
        "reranked": [ev["chunk_id"] for ev in out],
    }
