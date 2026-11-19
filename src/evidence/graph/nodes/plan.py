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
from .evidence_util import group_from_path

logger = logging.getLogger(__name__)


# ── shared formatting helpers ─────────────────────────────────────────────
def _corpus_tree(ctx, max_papers_per_group: int = 12) -> str:
    """Render the FULL corpus as a tree of groups → papers, from the store.

    A paper's *group* is the immediate parent folder of its on-disk path
    (see `evidence_util.group_from_path`); papers in the same folder form a
    group. External (OpenAlex/S2/Crossref) works are tagged `external`.

    This is injected into the planner and writer prompts so the model has
    always-visible awareness of what's in the corpus, and can use that
    structure when a question calls for it (e.g. "compare group A with
    group B", "summarise the finance papers") — without the user having to
    pass `--group` explicitly. The structure is informational: it does NOT
    filter the evidence set. The user can still hard-restrict via the
    `allowed_groups` state field (`evidence ask --group …`).

    Cap the per-group listing to keep the prompt compact; large corpora
    just lose some tail papers from the listing (they're still in the
    evidence set if retrieved).
    """
    try:
        papers = ctx.store.list_papers()
    except Exception as exc:  # noqa: BLE001 - never let a bad store kill a query
        logger.warning("corpus_tree: store unavailable: %s", exc)
        return ""
    if not papers:
        return ""

    by_group: dict[str, list] = {}
    for p in papers:
        grp = group_from_path(p.path) or "(root)"
        by_group.setdefault(grp, []).append(p)

    lines: list[str] = ["corpus tree (groups → papers):"]
    for grp in sorted(by_group, key=lambda g: g.lower()):
        plist = by_group[grp]
        extra = len(plist) - max_papers_per_group
        suffix = f"  (+{extra} more)" if extra > 0 else ""
        lines.append(f"  group [{grp}]  ({len(plist)} paper(s)){suffix}")
        for p in plist[:max_papers_per_group]:
            title = (p.title or p.path)[:80]
            path = p.path if p.path else ""
            doi = f"  doi={p.doi}" if p.doi else ""
            lines.append(f"    - {title}  {path}{doi}".rstrip())
    return "\n".join(lines)


def _fmt_evidence_block(evidence: list[dict], max_items: int = 12) -> str:
    """Render the evidence list for the planner/researcher/writer prompts.

    Each item shows its [E#] index, source path, relevance, paper title, and
    — when a group is set — the group label. The group is the paper's
    immediate parent folder name (see `evidence_util.group_from_path`).
    Group-aware prompting lets the model *contrast* groups for comparative
    questions ("what does group A conclude, and how does group B change it?").
    """
    lines = []
    for i, ev in enumerate(evidence[:max_items], start=1):
        title = (ev.get("title") or "")[:90]
        src = ev.get("source", "?")
        rel = ev.get("relevance", 0.0)
        grp = ev.get("group") or ""
        text = (ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 320:
            text = text[:320] + "…"
        group_part = f" group={grp}" if grp else ""
        lines.append(
            f"[E{i}] (src={src}, rel={rel:.2f}{group_part}) {title}\n    {text}"
        )
    return "\n".join(lines) or "(no evidence)"


# ── Planner ───────────────────────────────────────────────────────────────
_PLANNER_SYSTEM = (
    "You are a research planner. Given a question, intent, and a set of "
    "evidence items [E1..En], produce a concise outline for a cited answer. "
    "Respond with JSON: {\"outline\": [{\"section\": str, "
    "\"must_cover\": [str, ...], \"evidence_refs\": [int, ...]}]}. "
    "Each section's evidence_refs point to [E#] items that should back it. "
    "Keep to 3-6 sections. Do not invent facts; only plan what the evidence "
    "can support.\n\n"
    "Evidence items may carry a `group` label (the folder the paper was "
    "ingested from). When the question is comparative — e.g. asks how a "
    "conclusion changes across groups, or asks to weigh one group against "
    "another — plan sections that cover EACH relevant group separately and "
    "include a synthesis section that explicitly contrasts them, attributing "
    "each claim to its group's evidence."
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

    # Corpus structure (groups → papers). Always shown so the planner can
    # reference it when the question calls for it; harmless for others.
    tree = _corpus_tree(ctx)

    data = ctx.models.complete_json(
        prompt=(
            f"Intent: {intent}\nQuestion: {question}\n\n"
            + (f"{tree}\n\n" if tree else "")
            + f"Evidence:\n{_fmt_evidence_block(evidence)}\n\n"
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
