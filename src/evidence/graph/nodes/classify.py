"""Stage 1 — Query intent classification.

Routes into four intents (PLAN.MD): **content / bibliometric / trend /
current_world**. Uses the light model with JSON mode, and a rule-based keyword
fallback so the pipeline always produces a valid intent even if Ollama is down.
"""

from __future__ import annotations

import logging

from ..state import EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)

INTENTS = ("content", "bibliometric", "trend", "current_world")

_SYSTEM = (
    "You classify a user's research question into exactly one intent. "
    "Respond with JSON: {\"intent\": one of "
    "content|bibliometric|trend|current_world, \"reason\": str}. "
    "Definitions: "
    "content = asks about what a paper/method says (technical details, how, why). "
    "bibliometric = asks about authors, citations, who, counts, related works. "
    "trend = asks about how a field/method is evolving over time, state of the art. "
    "current_world = asks about very recent developments, 2024-2026, news, benchmarks."
)

# Deterministic fallback keywords (ordered by priority).
_KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "bibliometric",
        ("author", "authors", "cited", "citation", "who wrote", "co-citation", "references"),
    ),
    (
        "trend",
        ("evolving", "evolution", "trend", "history", "state of the art",
         "how has", "progression", "timeline"),
    ),
    (
        "current_world",
        ("recent", "latest", "2025", "2026", "2024", "current",
         "new ", "state-of-the-art", "sota", "news"),
    ),
]


def classify_intent(state: EvidenceState) -> dict:
    ctx = get_context()
    question = state.get("question", "").strip()
    if not question:
        return {"intent": "content"}

    data = ctx.models.complete_json(
        prompt=f"Question: {question}",
        model=ctx.cfg.ollama.light,
        system=_SYSTEM,
        temperature=0.0,
        fallback=None,
    )

    intent = None
    if isinstance(data, dict):
        raw = str(data.get("intent", "")).strip().lower()
        if raw in INTENTS:
            intent = raw
        else:
            # fuzzy match
            for cand in INTENTS:
                if cand in raw:
                    intent = cand
                    break

    if intent is None:
        intent = _keyword_fallback(question)
        logger.debug("intent fallback -> %s", intent)

    return {"intent": intent}


def _keyword_fallback(question: str) -> str:
    q = question.lower()
    for intent, kws in _KEYWORD_RULES:
        if any(kw in q for kw in kws):
            return intent
    return "content"
