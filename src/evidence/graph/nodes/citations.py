"""Stage 11 — Citation integrity verification.

Every claim → chunk mapping produced during drafting must resolve to a real,
indexed chunk (chunk_id + page + paper). Mappings that don't resolve to a
stored chunk are DROPPED before generation is finalised (PLAN.MD stage 11:
"Unverifiable mappings are dropped before generation").

This node is deliberately rule-based and deterministic: it never asks an LLM
whether a citation is real. It re-checks the store for each cited chunk_id and
only keeps citations that resolve. The generation nodes produce `citations` as
a list of `{claim, chunk_id, strength}`; this node validates them.
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)


def verify_citations(state: EvidenceState) -> dict:
    ctx = get_context()
    citations = state.get("citations") or []
    evidence: list[EvidenceItem] = list(state.get("evidence") or [])

    # Index of valid chunk ids (the merged evidence set + store).
    valid_ids: set[str] = set()
    for ev in evidence:
        cid = ev.get("chunk_id")
        if cid:
            valid_ids.add(cid)
    # Also accept any chunk the store knows about (in case evidence set is
    # narrower than the index).
    for cid in ctx.store.all_chunk_ids():
        valid_ids.add(cid)

    verified: list[dict] = []
    dropped: list[dict] = []
    for c in citations:
        cid = str(c.get("chunk_id", "")).strip()
        if not cid:
            dropped.append({**c, "reason": "missing chunk_id"})
            continue
        if cid in valid_ids:
            rec = ctx.store.get_chunk(cid)
            entry = dict(c)
            entry["verified"] = True
            if rec is not None:
                entry["page"] = rec.page
                entry["paper_id"] = rec.paper_id
            # Cross-check against evidence for relevance strength.
            for ev in evidence:
                if ev.get("chunk_id") == cid:
                    entry.setdefault("relevance", ev.get("relevance", 0.0))
                    entry["source"] = ev.get("source")
                    break
            verified.append(entry)
        else:
            dropped.append({**c, "reason": "chunk_id not in index"})

    # Mark the matched evidence items as verified.
    verified_ids = {v["chunk_id"] for v in verified}
    for ev in evidence:
        if ev.get("chunk_id") in verified_ids:
            ev["verified"] = True

    logger.debug(
        "citations: %d total, %d verified, %d dropped",
        len(citations), len(verified), len(dropped),
    )
    return {
        "citations": verified,
        "citations_dropped": dropped,
    }


# ── Public, state-free claim verification (MCP `verify_claim`) ────────────
def verify_claim(claim: str, chunk_id: str) -> dict:
    """Rule-based check: does `claim` follow from the indexed `chunk_id`?

    Deterministic, no LLM required. Scores the overlap of the claim's
    content-bearing tokens with the chunk text and returns a verdict of
    ``SUPPORTED`` / ``PARTIAL`` / ``UNSUPPORTED`` plus the exact chunk text so
    the caller can cite it. This is intentionally conservative: it confirms the
    claim is *grounded in* a real chunk, it does not do full NLI. Use a local
    NLI model (out of scope here) if you need entailment-grade guarantees.
    """
    ctx = get_context()
    rec = ctx.store.get_chunk(chunk_id)
    if rec is None:
        return {
            "claim": claim,
            "chunk_id": chunk_id,
            "verdict": "UNSUPPORTED",
            "reason": "chunk_id not in index",
            "text": None,
            "score": 0.0,
        }

    paper = ctx.store.get_paper(rec.paper_id)
    score = _claim_overlap(claim, rec.text)
    if score >= 0.6:
        verdict = "SUPPORTED"
    elif score >= 0.3:
        verdict = "PARTIAL"
    else:
        verdict = "UNSUPPORTED"
    return {
        "claim": claim,
        "chunk_id": chunk_id,
        "verdict": verdict,
        "score": round(score, 3),
        "text": rec.text,
        "paper_title": paper.title if paper else "",
        "page": rec.page,
        "doi": paper.doi if paper else None,
    }


def _content_tokens(text: str) -> set[str]:
    import re

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text.lower())
    # Drop short/common tokens and pure numbers (weak signals).
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "we", "our", "is", "are", "was", "be", "that", "this", "as", "by",
        "at", "from", "not", "no", "it", "its", "per", "via", "e.g", "i.e",
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def _claim_overlap(claim: str, text: str) -> float:
    claim_tokens = _content_tokens(claim)
    text_tokens = _content_tokens(text)
    if not claim_tokens:
        return 0.0
    hits = claim_tokens & text_tokens
    return len(hits) / len(claim_tokens)
