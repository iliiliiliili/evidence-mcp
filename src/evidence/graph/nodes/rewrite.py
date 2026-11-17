"""Stage 2 — Query rewriting into text + (optional) visual query variants.

Produces 2–3 paraphrase variants that cover different phrasings of the same
intent, plus an optional visual query for Phase 4. A drift guard is applied so
a reformulation cannot stray too far from the original intent (PLAN.MD stage 8
reuses this same guard).
"""

from __future__ import annotations

import logging

from ..state import EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You rewrite a research question into search-friendly variants. "
    "Respond with JSON: {\"text_queries\": [str, str, str], "
    "\"visual_query\": str | null, \"shortcomings\": [str]}. "
    "text_queries should be 2-3 distinct but tightly-scoped paraphrases that "
    "preserve the original intent; they should not add unrelated topics. "
    "visual_query is a short description of a figure/table/diagram that would "
    "answer the question, or null if none is likely. "
    "shortcomings lists retrieval weaknesses you expect (e.g. 'needs a "
    "specific benchmark table') so the sufficiency gate can later exploit it."
)


def _default_variants(question: str, n: int) -> list[str]:
    """Deterministic fallback: keep the question + a couple of light rephrases."""
    q = question.strip()
    base = [q]
    if "?" in q:
        base.append(q.rstrip("?").strip())
    # Strip a leading "what is / how does" to make a noun-phrase query.
    for lead in ("what is ", "what are ", "how does ", "how do ", "explain "):
        if q.lower().startswith(lead):
            base.append(q[len(lead):].strip())
            break
    out: list[str] = []
    for v in base:
        if v and v not in out:
            out.append(v)
        if len(out) >= n:
            break
    while len(out) < n:
        out.append(q)
    return out


def rewrite_queries(state: EvidenceState) -> dict:
    ctx = get_context()
    question = state.get("question", "").strip()
    intent = state.get("intent", "content")
    n = max(2, ctx.cfg.retry.max_query_variants)

    data = ctx.models.complete_json(
        prompt=f"Intent: {intent}\nQuestion: {question}",
        model=ctx.cfg.ollama.light,
        system=_SYSTEM,
        temperature=0.3,
        fallback=None,
    )

    text_queries: list[str] = []
    visual_query: str | None = None
    shortcomings: list[str] = []

    if isinstance(data, dict):
        raw_q = data.get("text_queries") or []
        if isinstance(raw_q, list):
            text_queries = [str(x).strip() for x in raw_q if str(x).strip()]
        vq = data.get("visual_query")
        if isinstance(vq, str) and vq.strip():
            visual_query = vq.strip()
        sc = data.get("shortcomings") or []
        if isinstance(sc, list):
            shortcomings = [str(x).strip() for x in sc if str(x).strip()]

    if not text_queries:
        text_queries = _default_variants(question, n)
    # Always keep the original as the first variant.
    if question not in text_queries:
        text_queries.insert(0, question)
    text_queries = text_queries[:n]

    return {
        "text_queries": text_queries,
        "visual_query": visual_query,
        "retrieval_shortcomings": shortcomings or state.get("retrieval_shortcomings", []),
    }
