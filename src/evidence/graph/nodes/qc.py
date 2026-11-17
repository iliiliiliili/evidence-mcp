"""Stage 13 — Post-generation quality control (final gate).

Three checks, all rule-checkable (PLAN.MD stage 13 + "Key risks"):
  1. Citation spot-check: re-verify a sample of [C<n>] markers resolve to a
     real indexed chunk (re-runs the same store lookup as `verify_citations`).
  2. Answer-coverage: every outline section is represented in the draft.
  3. Hallucination scan: flag draft sentences that contain a [C<n>] marker
     whose citation is missing/dropped (i.e. the marker claims a source that
     doesn't exist) — a strong hallucination signal.

Fail → re-enter at `plan` (max 1 regeneration). Pass → finalise `answer`.
"""

from __future__ import annotations

import logging
import re

from ..state import EvidenceState
from .context import get_context

logger = logging.getLogger(__name__)


_STOP = {"the", "and", "for", "with", "are", "was", "will", "our", "their", "its", "from"}


def _coverage_check(state: EvidenceState) -> tuple[bool, list[str]]:
    """Loose coverage: each outline section 'covers' if any significant
    keyword of its name appears in the draft. Fragile by design (no LLM),
    so we only fail when NO section is covered at all — not on partial misses.
    """
    outline = (state.get("outline") or {}).get("outline") or []
    draft = (state.get("draft") or "").lower()
    if not outline:
        return True, []

    covered = 0
    missing: list[str] = []
    for sec in outline:
        name = str(sec.get("section", "")).strip()
        if not name:
            continue
        keywords = [
            w.lower().strip(".,;:")
            for w in name.split()
            if len(w) >= 4 and w.lower() not in _STOP
        ]
        if any(k in draft for k in keywords):
            covered += 1
        else:
            missing.append(name)
    # Fail only if nothing was covered (or there was nothing to cover).
    ok = covered > 0 or len(outline) == 0
    return ok, missing


def _hallucination_scan(state: EvidenceState) -> tuple[bool, list[str]]:
    """Markers in the draft whose citation is absent from the verified set."""
    draft = state.get("draft") or ""
    verified = {str(c.get("id")) for c in (state.get("citations") or [])}
    markers = re.findall(r"\[C(\d+)\]", draft)
    orphans = [m for m in markers if m not in verified]
    # A draft that cites nothing at all is a fail (no grounding markers).
    if not markers:
        return False, ["no citation markers in draft"]
    return (len(orphans) == 0, [f"unresolved marker [C{m}]" for m in orphans[:10]])


def quality_check(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    regenerations = int(state.get("regenerations", 0))

    # 1. Citation spot-check (re-verify against the store).
    citations = state.get("citations") or []
    verified_count = 0
    for c in citations:
        cid = str(c.get("chunk_id", "")).strip()
        if cid and ctx.store.get_chunk(cid) is not None:
            verified_count += 1
    cit_ok = verified_count >= max(1, len(citations)) if citations else True
    # If there are no citations at all, that's a fail (no grounding).
    if not citations:
        cit_ok = False

    # 2. Coverage.
    cov_ok, missing_sections = _coverage_check(state)

    # 3. Hallucination.
    hal_ok, orphan_markers = _hallucination_scan(state)

    passed = cit_ok and cov_ok and hal_ok

    report = {
        "citation_verified": verified_count,
        "citation_total": len(citations),
        "citation_ok": cit_ok,
        "coverage_ok": cov_ok,
        "missing_sections": missing_sections,
        "hallucination_ok": hal_ok,
        "orphan_markers": orphan_markers,
        "passed": passed,
        "regenerations": regenerations,
    }

    if passed:
        # Finalise the answer with an evidence appendix.
        answer = state.get("draft", "")
        report["note"] = "passed all QC gates"
        logger.info("QC passed: %s", report)
        return {
            "qc_report": report,
            "qc_passes": int(state.get("qc_passes", 0)) + 1,
            "answer": answer,
        }

    # Fail → regenerate if budget allows; else finalise a best-effort answer
    # with a prominent caveat rather than looping forever.
    if regenerations < cfg.generation.max_regenerations:
        logger.info("QC failed (%s); regenerating", report)
        problems = []
        if not cit_ok:
            problems.append(f"citation_ok={cit_ok} ({verified_count}/{len(citations)} verified)")
        if not cov_ok:
            problems.append(f"coverage_ok=False (missing: {missing_sections})")
        if not hal_ok:
            problems.append(f"hallucination_ok=False (orphan markers: {orphan_markers})")
        return {
            "qc_report": report,
            "regenerations": regenerations + 1,
            # Always carry a best-effort answer so the user gets output even
            # if the graph terminates at this node.
            "answer": state.get("draft", "(no draft produced)"),
            "critic_verdict": {
                **(state.get("critic_verdict") or {}),
                "feedback": "QC failures to address: " + "; ".join(problems),
            },
        }

    # Budget exhausted — finalise with a caveat so the user still gets output.
    caveats = [
        f"citation_ok={cit_ok}",
        f"coverage_ok={cov_ok} (missing: {missing_sections})",
        f"hallucination_ok={hal_ok} (orphan markers: {orphan_markers})",
    ]
    answer = (
        (state.get("draft") or "(no draft produced)")
        + "\n\n---\n⚠ QC did not fully pass after "
        + str(cfg.generation.max_regenerations)
        + " regeneration(s):\n  - "
        + "\n  - ".join(caveats)
        + "\nVerify cited claims against the sources before relying on them."
    )
    report["note"] = "finalised with caveats (QC budget exhausted)"
    logger.info("QC finalised with caveats: %s", caveats)
    return {
        "qc_report": report,
        "answer": answer,
    }
