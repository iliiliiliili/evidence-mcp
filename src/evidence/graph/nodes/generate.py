"""Stage 12 (cont.) — Writer (`draft`) and Critic (`critique`).

Writer: produces a cited draft. Every load-bearing claim is tagged with a
`[C<n>]` marker that maps (via the `citations` list it also returns) to an
evidence item [E#] and ultimately to a chunk_id. This marker system is what
lets `verify_citations` and the QC hallucination scan operate on a fully
rule-checkable basis (PLAN.MD "Key risks": "citation verification is fully
rule-checkable — prefer it over LLM judgment").

Critic: scores the draft against a rubric, returns pass / revise / fail with
specific feedback that feeds the next Writer pass.
"""

from __future__ import annotations

import logging
import re

from ..state import EvidenceState
from .context import get_context
from .plan import _corpus_tree, _fmt_evidence_block

logger = logging.getLogger(__name__)


# ── Writer ────────────────────────────────────────────────────────────────
_WRITER_SYSTEM = (
    "You are a careful technical writer. Using ONLY the provided evidence "
    "items [E1..En] and the outline, write a clear, well-structured answer to "
    "the question. Rules:\n"
    "1. Cite every load-bearing claim inline with a marker like [C1], [C2].\n"
    "2. The 'citations' array must map each [C<n>] to the evidence ref [E#] "
    "that supports it and the specific claim text.\n"
    "3. Do NOT state facts that no evidence item supports. If the evidence is "
    "insufficient for a sub-point, say so explicitly (e.g. 'the provided "
    "evidence does not establish…').\n"
    "4. Prefer the outline's sections as headings.\n"
    "5. Evidence items may carry a `group` label. When the question asks to "
    "compare or contrast groups, attribute each conclusion to its group "
    "(e.g. 'Group A concludes …', 'Group B finds …') and make the "
    "differences explicit. Never blend two groups' claims into one unattributed "
    "statement.\n\n"
    "Respond with JSON: {\"draft\": str, \"citations\": "
    "[{\"id\": int, \"claim\": str, \"evidence_ref\": int}]}. "
    "evidence_ref is the [E#] index (1-based) the claim draws on."
)


def _ref_to_chunk_id(evidence: list[dict], ref: int) -> str:
    """Map a 1-based [E#] ref to the underlying chunk_id (or '' if invalid)."""
    if isinstance(ref, int) and 1 <= ref <= len(evidence):
        return evidence[ref - 1].get("chunk_id", "")
    return ""


def draft(state: EvidenceState) -> dict:
    ctx = get_context()
    evidence: list[dict] = list(state.get("evidence") or [])
    question = state.get("question", "")
    outline = (state.get("outline") or {}).get("outline") or []
    evidence_map = state.get("evidence_map") or {}
    critic = state.get("critic_verdict") or {}
    cycles = int(state.get("critic_cycles", 0))

    prompt_parts = [
        f"Question: {question}",
        f"Outline:\n{outline}",
    ]
    # Corpus structure (groups → papers) so the writer can attribute claims
    # to groups when the question is comparative ("what does group A
    # conclude, and how does group B change it?"). Always shown — harmless
    # for non-comparative questions, but it lets the model name groups it
    # sees in the outline or evidence without the user having to pass --group.
    tree = _corpus_tree(ctx)
    if tree:
        prompt_parts.append(tree)
    if evidence_map.get("map"):
        prompt_parts.append(f"Evidence map:\n{evidence_map['map']}")
    prompt_parts.append(f"Evidence:\n{_fmt_evidence_block(evidence)}")
    if cycles > 0 and critic.get("feedback"):
        prompt_parts.append(
            f"Previous draft was critiqued (cycle {cycles}). Address this feedback:\n"
            f"{critic.get('feedback')}\nRevise accordingly."
        )
    prompt_parts.append("\nWrite the cited answer now.")

    data = ctx.models.complete_json(
        prompt="\n\n".join(prompt_parts),
        model=ctx.cfg.ollama.reasoning,
        system=_WRITER_SYSTEM,
        temperature=ctx.cfg.generation.writer_temp,
        fallback=None,
    )

    if not isinstance(data, dict) or not data.get("draft"):
        # Deterministic fallback: a minimal, honest, citation-tagged answer.
        data = _fallback_draft(state, evidence)

    draft_text = str(data.get("draft", "")).strip()

    # Resolve [C<n>] markers → chunk_id, storing raw citations for verification.
    citations: list[dict] = []
    raw_cits = data.get("citations") or []
    seen_ids: set[int] = set()
    for i, c in enumerate(raw_cits, start=1):
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            cid = i
        cid = int(cid) if str(cid).isdigit() else i
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        ref = c.get("evidence_ref")
        chunk_id = _ref_to_chunk_id(evidence, ref)
        citations.append(
            {
                "id": cid,
                "claim": str(c.get("claim", "")).strip(),
                "evidence_ref": ref,
                "chunk_id": chunk_id,
            }
        )
    # If the model produced no citations but referenced markers, back-fill a
    # citation per [C<n>] tag using the first evidence item as best-effort.
    tags = re.findall(r"\[C(\d+)\]", draft_text)
    if not citations and tags and evidence:
        for t in sorted(set(int(x) for x in tags)):
            citations.append(
                {
                    "id": t,
                    "claim": f"claim [C{t}] (auto-mapped)",
                    "evidence_ref": 1,
                    "chunk_id": evidence[0].get("chunk_id", ""),
                }
            )

    return {
        "draft": draft_text,
        "citations": citations,
    }


def _fallback_draft(state: EvidenceState, evidence: list[dict]) -> dict:
    """Build a minimal honest cited answer if the writer model is unavailable."""
    question = state.get("question", "")
    if not evidence:
        return {
            "draft": (
                f"Question: {question}\n\n"
                "No local evidence was retrieved for this question. "
                "No claims are made because none could be grounded."
            ),
            "citations": [],
        }
    lines = [f"Question: {question}\n"]
    cits = []
    for i, ev in enumerate(evidence[:3], start=1):
        marker = f"[C{i}]"
        snippet = (ev.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "…"
        title = ev.get("title", "untitled")
        lines.append(f"{marker} {title}: {snippet}")
        cits.append(
            {
                "id": i,
                "claim": f"excerpt from {title}",
                "evidence_ref": i,
                "chunk_id": ev.get("chunk_id", ""),
            }
        )
    lines.append(
        "\nNote: automated draft (writer model unavailable). Verify claims against sources."
    )
    return {"draft": "\n\n".join(lines), "citations": cits}


# ── Critic ────────────────────────────────────────────────────────────────
_CRITIC_SYSTEM = (
    "You are a rigorous reviewer of a cited technical answer. Score it 0-100 "
    "against: groundedness (claims supported by provided evidence), "
    "coverage (answers the question), coherence (clear structure), and "
    "citation integrity (markers map to real evidence). Respond with JSON: "
    "{\"score\": int, \"verdict\": \"pass\"|\"revise\"|\"fail\", "
    "\"feedback\": str, \"ungrounded_claims\": [str]}. "
    "Use 'fail' only if the draft is largely unsupported or off-topic. "
    "Be specific in feedback so the writer can fix it."
)


def critique(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    draft_text = state.get("draft", "")
    question = state.get("question", "")
    cycles = int(state.get("critic_cycles", 0))
    max_cycles = cfg.generation.critic_max_cycles
    threshold = cfg.generation.critic_pass_threshold

    # Quick rule check: any [C<n>] markers present at all?
    has_markers = bool(re.search(r"\[C\d+\]", draft_text))

    data = ctx.models.complete_json(
        prompt=(
            f"Question: {question}\n\nDraft:\n{draft_text}\n\n"
            "Score and review it."
        ),
        model=cfg.ollama.reasoning,
        system=_CRITIC_SYSTEM,
        temperature=cfg.generation.critic_temp,
        fallback=None,
    )

    score = 0
    verdict = "revise"
    feedback = ""
    if isinstance(data, dict):
        try:
            score = int(data.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        v = str(data.get("verdict", "revise")).lower()
        verdict = v if v in {"pass", "revise", "fail"} else "revise"
        feedback = str(data.get("feedback", "")).strip()

    # Rule-based override: no markers at all → cannot be a pass.
    if not has_markers:
        verdict = "fail" if cycles >= max_cycles else "revise"
        feedback = (feedback + " | No inline [C<n>] citation markers found.").strip(" |")

    # Enforce pass threshold.
    if verdict == "pass" and score < threshold:
        verdict = "revise"

    # If we've exhausted cycles and still failing, force a final pass-ish state
    # so the pipeline can finish (QC will still gate).
    if verdict == "fail" and cycles >= max_cycles:
        verdict = "revise"
        feedback = (
            feedback + " | Critic cycles exhausted; proceed to QC with current draft."
        ).strip(" |")

    return {
        "critic_verdict": {
            "score": score,
            "verdict": verdict,
            "feedback": feedback,
            "cycles": cycles,
        },
        "critic_cycles": cycles + 1,
    }
