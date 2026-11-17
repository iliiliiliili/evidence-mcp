"""Stage 12 — Planner / Researcher / Writer / Critic sub-graph.

Four agents with distinct prompts + temperatures (PLAN.MD stage 12):

  Planner    → structured outline (sections + what each must cover)
  Researcher → maps outline sections to evidence items with strength ratings
  Writer     → cited draft (every load-bearing claim tagged [C<n>])
  Critic     → scores the draft against the rubric → revise / pass / regenerate

The Writer↔Critic loop is bounded by `generation.critic_max_cycles`; the whole
sub-graph can re-enter at `plan` once (max_regenerations) from stage-13 QC.
"""

from __future__ import annotations

import logging
from typing import Any

from ..state import EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)


# ── shared formatting helpers ─────────────────────────────────────────────
def _fmt_evidence_block(evidence: list[dict], max_items: int = 12) -> str:
    lines = []
    for i, ev in enumerate(evidence[:max_items], start=1):
        title = ev.get("title", "")[:90]
        src = ev.get("source", "?")
        rel = ev.get("relevance", 0.0)
        text = (ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 320:
            text = text[:320] + "…"
        lines.append(f"[E{i}] (src={src}, rel={rel:.2f}) {title}\n    {text}")
    return "\n".join(lines) or "(no evidence)"


# ── Planner ───────────────────────────────────────────────────────────────
_PLANNER_SYSTEM = (
    "You are a research planner. Given a question, intent, and a set of "
    "evidence items [E1..En], produce a concise outline for a cited answer. "
    "Respond with JSON: {\"outline\": [{\"section\": str, "
    "\"must_cover\": [str, ...], \"evidence_refs\": [int, ...]}]}. "
    "Each section's evidence_refs point to [E#] items that should back it. "
    "Keep to 3-6 sections. Do not invent facts; only plan what the evidence "
    "can support."
)


def _default_outline(state: EvidenceState) -> dict[str, Any]:
    return {
        "outline": [
            {
                "section": "Direct answer",
                "must_cover": ["answer the question"],
                "evidence_refs": [1],
            },
            {
                "section": "Supporting evidence",
                "must_cover": ["key claims with citations"],
                "evidence_refs": [],
            },
            {
                "section": "Limitations / caveats",
                "must_cover": ["what is not established"],
                "evidence_refs": [],
            },
        ]
    }


def plan(state: EvidenceState) -> dict:
    ctx = get_context()
    evidence: list[dict] = list(state.get("evidence") or [])
    question = state.get("question", "")
    intent = state.get("intent", "content")

    data = ctx.models.complete_json(
        prompt=(
            f"Intent: {intent}\nQuestion: {question}\n\n"
            f"Evidence:\n{_fmt_evidence_block(evidence)}\n\n"
            "Produce the outline."
        ),
        model=ctx.cfg.ollama.reasoning,
        system=_PLANNER_SYSTEM,
        temperature=ctx.cfg.generation.planner_temp,
        fallback=None,
    )

    if isinstance(data, dict) and isinstance(data.get("outline"), list) and data["outline"]:
        return {"outline": {"outline": data["outline"]}}
    return {"outline": _default_outline(state)}


# ── Researcher ────────────────────────────────────────────────────────────
_RESEARCHER_SYSTEM = (
    "You are an evidence mapper. For each outline section, choose which "
    "evidence items [E#] support it and rate each as strong/medium/weak. "
    "Respond with JSON: {\"map\": [{\"section\": str, "
    "\"evidence\": [{\"ref\": int, \"strength\": \"strong\"|\"medium\"|\"weak\", "
    "\"claim\": str}]}]}. The 'claim' is the specific statement the evidence "
    "will be used to make. Only use evidence that genuinely supports the claim."
)


def map_evidence(state: EvidenceState) -> dict:
    ctx = get_context()
    evidence: list[dict] = list(state.get("evidence") or [])
    outline = (state.get("outline") or {}).get("outline") or []
    question = state.get("question", "")

    data = ctx.models.complete_json(
        prompt=(
            f"Question: {question}\nOutline:\n{outline}\n\n"
            f"Evidence:\n{_fmt_evidence_block(evidence)}\n\nMap evidence to sections."
        ),
        model=ctx.cfg.ollama.light,
        system=_RESEARCHER_SYSTEM,
        temperature=ctx.cfg.generation.researcher_temp,
        fallback=None,
    )

    if isinstance(data, dict) and isinstance(data.get("map"), list) and data["map"]:
        return {"evidence_map": {"map": data["map"]}}

    # Fallback: attach all evidence to the first section as "medium".
    fallback_map = []
    for sec in outline:
        fallback_map.append(
            {
                "section": sec.get("section", ""),
                "evidence": [{"ref": i + 1, "strength": "medium", "claim": ""}
                             for i in range(min(3, len(evidence)))],
            }
        )
    return {"evidence_map": {"map": fallback_map}}
