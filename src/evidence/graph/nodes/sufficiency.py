"""Stage 7 — EvidenceItem sufficiency gate (100-point rubric).

Hybrid design (PLAN.MD stage 7 + "Key risks"):
  1. Rule-based scores for each of the 5 dimensions (coverage, relevance,
     depth, recency, consistency) — deterministic, cheap, and reliable.
  2. For *ambiguous* cases (total within ±15 of the threshold) an LLM review
     can nudge the verdict up or down, but never override a rule-based
     hard-fail / hard-pass.
  3. Relevance damping: weak chunks (below `relevance_floor`) contribute
     half-strength to relevance/depth, so a single good chunk can't carry a
     corpus of weak ones.

The node sets `sufficiency`, `sufficiency_decision`, and records
`retrieval_shortcomings` to drive the stage-8 reformulation loop.
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)


def _relevance_damp(ev: EvidenceItem, floor: float, factor: float) -> float:
    r = float(ev.get("relevance", 0.0))
    return r * factor if r < floor else r


def _score_coverage(evidence: list[EvidenceItem], question: str, cfg) -> tuple[float, str]:
    """How much of the question's sub-topics is represented."""
    if not evidence:
        return 0.0, "no evidence"
    # Distinct papers represented → breadth.
    papers = {ev.get("paper_id") for ev in evidence}
    n = len(evidence)
    score = min(20.0, 4.0 * len(papers) + 1.5 * n)
    return score, f"{n} chunks across {len(papers)} paper(s)"


def _score_relevance(evidence: list[EvidenceItem], cfg) -> tuple[float, str]:
    floor = cfg.sufficiency.relevance_floor
    damped = [_relevance_damp(ev, floor, cfg.sufficiency.relevance_damp_factor) for ev in evidence]
    if not damped:
        return 0.0, "no evidence"
    mean = sum(damped) / len(damped)
    score = min(20.0, mean * 20.0)
    return score, f"mean relevance {mean:.2f}"


def _score_depth(evidence: list[EvidenceItem], cfg) -> tuple[float, str]:
    """Depth = enough characters of on-topic text to ground a real answer."""
    floor = cfg.sufficiency.relevance_floor
    total_chars = sum(
        len(ev.get("text", "")) * (cfg.sufficiency.relevance_damp_factor
                                  if float(ev.get("relevance", 0)) < floor else 1.0)
        for ev in evidence
    )
    # ~6000 chars ≈ a solid answer; scale up to 20.
    score = min(20.0, 20.0 * (total_chars / 6000.0))
    return score, f"~{total_chars} grounded chars"


def _score_recency(evidence: list[EvidenceItem], intent: str) -> tuple[float, str]:
    """Recency. Full marks if intent doesn't care about recency."""
    if intent in ("content", "bibliometric"):
        return 15.0, "not gated on recency for this intent"
    years = [ev.get("year") for ev in evidence if ev.get("year")]
    if not years:
        return 8.0, "no year metadata (neutral)"
    newest = max(years)
    # 2026 → full, 2024 → most, older → decays.
    if newest >= 2025:
        return 20.0, f"newest {newest}"
    if newest >= 2023:
        return 15.0, f"newest {newest}"
    return 8.0, f"newest {newest}"


def _score_consistency(evidence: list[EvidenceItem], cfg) -> tuple[float, str]:
    """Consistency = internal coherence; approximated by relevance spread."""
    floor = cfg.sufficiency.relevance_floor
    rels = [
        _relevance_damp(ev, floor, cfg.sufficiency.relevance_damp_factor)
        for ev in evidence
    ]
    if len(rels) < 2:
        return 12.0, "single evidence item (cannot judge spread)"
    mean = sum(rels) / len(rels)
    var = sum((r - mean) ** 2 for r in rels) / len(rels)
    std = var ** 0.5
    # Tight, high cluster → high consistency.
    score = max(0.0, 20.0 - 25.0 * std) * min(1.0, mean + 0.3)
    score = min(20.0, score)
    return score, f"relevance spread σ={std:.2f}"


def _rule_scores(state: EvidenceState, cfg, evidence: list[EvidenceItem]) -> dict[str, float]:
    intent = state.get("intent", "content")
    question = state.get("question", "")
    dims = {
        "coverage": _score_coverage(evidence, question, cfg)[0],
        "relevance": _score_relevance(evidence, cfg)[0],
        "depth": _score_depth(evidence, cfg)[0],
        "recency": _score_recency(evidence, intent)[0],
        "consistency": _score_consistency(evidence, cfg)[0],
    }
    return dims


def score_evidence(
    question: str,
    evidence: list[EvidenceItem],
    *,
    cfg,
    intent: str = "content",
) -> dict:
    """Standalone (state-free) version of the 100-pt sufficiency rubric.

    This is the public surface the MCP `check_evidence` tool wraps: given a
    question + a set of already-retrieved `EvidenceItem` items, return per-dimension
    scores, a total, a verdict, and suggestions for missing evidence. No graph
    state is required, so it can be called on demand without running the
    pipeline. Reuses the same deterministic dimension scorers as the node.
    """
    threshold = cfg.sufficiency.threshold
    if len(evidence) < cfg.sufficiency.min_evidence_items:
        return {
            "total": 0.0,
            "dimensions": {},
            "verdict": "retry",
            "rationale": (
                f"only {len(evidence)} evidence item(s); "
                f"below min {cfg.sufficiency.min_evidence_items}"
            ),
            "suggestions": [
                "retrieve more evidence on the sub-topics in the question",
            ],
        }

    dims = {
        "coverage": _score_coverage(evidence, question, cfg),
        "relevance": _score_relevance(evidence, cfg),
        "depth": _score_depth(evidence, cfg),
        "recency": _score_recency(evidence, intent),
        "consistency": _score_consistency(evidence, cfg),
    }
    caps = cfg.sufficiency.dimensions
    scores = {k: min(v[0], caps.get(k, 20.0)) for k, v in dims.items()}
    total = sum(scores.values())
    verdict = "sufficient" if total >= threshold else "retry"

    # Suggest the weakest dimensions (what a reformulated query should target).
    weakest = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    suggestions = [
        f"strengthen {name} (scored {score:.1f}: {dims[name][1]})"
        for name, score in weakest
        if score < caps.get(name, 20.0)
    ]

    return {
        "total": round(total, 1),
        "dimensions": {k: round(v, 1) for k, v in scores.items()},
        "verdict": verdict,
        "threshold": threshold,
        "rationale": {k: v[1] for k, v in dims.items()},
        "suggestions": suggestions,
    }


_LLM_SYSTEM = (
    "You are reviewing whether a set of evidence is sufficient to answer a "
    "technical question. Respond with JSON: "
    "{\"decision\": \"sufficient\"|\"insufficient\", \"nudge\": float in [-10,10], "
    "\"shortcomings\": [str], \"reason\": str}. "
    "Only use 'nudge' to fine-tune a close call — do not overturn a clear rule "
    "verdict. List concrete shortcomings (missing tables, missing equations, "
    "missing author/baseline) that a reformulated query could target."
)


def score_sufficiency(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    evidence: list[EvidenceItem] = list(state.get("evidence") or [])
    threshold = cfg.sufficiency.threshold
    intent = state.get("intent", "content")
    question = state.get("question", "")
    retries = int(state.get("retries", 0))
    max_retries = cfg.retry.max_retries

    # Hard rule: too little evidence → immediate insufficient.
    if len(evidence) < cfg.sufficiency.min_evidence_items:
        decision = "retry" if retries < max_retries else _escalate(intent, cfg)
        out = {
            "sufficiency": {
                "total": 0.0,
                "dimensions": {},
                "rationale": (
                    f"only {len(evidence)} evidence item(s); "
                    f"below min {cfg.sufficiency.min_evidence_items}"
                ),
                "llm_reviewed": False,
            },
            "sufficiency_decision": decision,
            "retrieval_shortcomings": state.get("retrieval_shortcomings", [])
            + [f"insufficient evidence ({len(evidence)} items)"],
        }
        if decision == "retry":
            out["retries"] = retries + 1
        return out

    dims = _rule_scores(state, cfg, evidence)
    # Apply the configured per-dimension maxima (so config can rebalance).
    caps = cfg.sufficiency.dimensions
    total = sum(min(dims[k], caps.get(k, 20.0)) for k in caps)
    rationale = {k: v for k, v in dims.items()}

    llm_reviewed = False
    nudge = 0.0
    shortcomings = list(state.get("retrieval_shortcomings", []))
    # LLM review only for close calls (within ±15 of threshold).
    if abs(total - threshold) <= 15:
        review = ctx.models.complete_json(
            prompt=(
                f"Intent: {intent}\nQuestion: {question}\n"
                f"Rule-based total: {total:.1f}/100 (threshold {threshold})\n"
                f"EvidenceItem titles:\n"
                + "\n".join(f"- {ev.get('title','')[:80]} ({ev.get('source')})"
                            for ev in evidence[:8])
            ),
            model=cfg.ollama.reasoning,
            system=_LLM_SYSTEM,
            temperature=0.2,
            fallback=None,
        )
        if isinstance(review, dict):
            llm_reviewed = True
            try:
                nudge = float(review.get("nudge", 0.0))
            except (TypeError, ValueError):
                nudge = 0.0
            nudge = max(-10.0, min(10.0, nudge))
            sc = review.get("shortcomings") or []
            if isinstance(sc, list):
                for s in sc:
                    s = str(s).strip()
                    if s and s not in shortcomings:
                        shortcomings.append(s)
            rationale["llm_nudge"] = nudge
            rationale["llm_reason"] = str(review.get("reason", ""))

    final_total = max(0.0, min(100.0, total + nudge))

    # Decide: sufficient / retry / external.
    if final_total >= threshold:
        decision = "sufficient"
    elif retries < max_retries:
        decision = "retry"
    else:
        decision = _escalate(intent, cfg)

    # If we're escalating and external is off, still surface a decision.
    if decision == "external" and not cfg.external.enabled:
        # No external path available — fall through to best-effort generation
        # with what we have, flagging the shortfall.
        decision = "sufficient"
        rationale["external_unavailable"] = True

    # Guard: never loop external → merge → external again. If external was
    # already attempted this run, escalate to best-effort generation.
    if decision == "external" and state.get("external_used"):
        decision = "sufficient"
        rationale["external_already_used"] = True

    logger.debug(
        "sufficiency: total=%.1f decision=%s retries=%d/%d",
        final_total,
        decision,
        retries,
        max_retries,
    )
    out: dict = {
        "sufficiency": {
            "total": final_total,
            "dimensions": rationale,
            "llm_reviewed": llm_reviewed,
            "threshold": threshold,
        },
        "sufficiency_decision": decision,
        "retrieval_shortcomings": shortcomings,
    }
    # Advance the retry counter only when we actually loop back, so the gate
    # can never retry forever (PLAN.MD stage 8 budget).
    if decision == "retry":
        out["retries"] = retries + 1
    return out


def _escalate(intent: str, cfg) -> str:
    """When retries are exhausted, decide external vs. best-effort."""
    if cfg.external.enabled:
        return "external"
    # Without external, we still generate from what we have (best effort).
    return "sufficient"
