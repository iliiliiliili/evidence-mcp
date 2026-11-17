"""Stage 3 — Hybrid text retrieval (FAISS + BM25 + RRF).

Reads `text_queries`, runs the hybrid retriever, and converts the top
chunk_ids into `EvidenceItem` records (with rank-derived relevance). The rerank
node will re-order and re-score these next.
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context
from .evidence_util import evidence_from_chunk

logger = logging.getLogger(__name__)


def retrieve_hybrid(state: EvidenceState) -> dict:
    ctx = get_context()
    queries = state.get("text_queries") or [state.get("question", "")]
    queries = [q for q in queries if q]
    if not queries:
        return {"main_evidence": []}

    result = ctx.retriever.hybrid(queries)
    ranked = result.ranked

    # Rank-derived relevance: top item ~1.0, tail ~0. Use a smooth decay.
    n = max(1, len(ranked))
    evidence: list[EvidenceItem] = []
    for i, (chunk_id, score) in enumerate(ranked):
        # Normalise the fused score into [0,1] relative to the top.
        top = ranked[0][1] if ranked else 0.0
        relevance = (score / top) if top > 0 else (1.0 - i / (n + 1))
        relevance = max(0.0, min(1.0, relevance))
        ev = evidence_from_chunk(
            ctx.store,
            chunk_id,
            source="rrf",
            relevance=relevance,
            provenance={"rank": i + 1, "score": float(score), "queries": queries},
        )
        if ev is not None:
            evidence.append(ev)

    logger.debug("retrieve_hybrid: %d chunks for %d queries", len(evidence), len(queries))
    return {
        "main_evidence": evidence,
        "dense_ranked": result.dense_ranked,
        "sparse_ranked": result.sparse_ranked,
        "rrf_ranked": [cid for cid, _ in ranked],
    }
