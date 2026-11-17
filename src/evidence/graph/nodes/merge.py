"""Stage 10 — Merge + dedupe multimodal evidence.

Combines the main text path (`evidence`) with the parallel-path accumulators
(`graph_evidence`, `visual_evidence`, `external_evidence`), dedupes by chunk
hash / chunk_id (keeping the highest relevance), and re-sorts by relevance so
the generation sub-graph always sees the strongest evidence first.

Provenance is carried on every item (PLAN.MD stage 10: "carry provenance on
every item").
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context  # noqa: F401  (imported for future store access)
from .evidence_util import dedupe_evidence

logger = logging.getLogger(__name__)


def merge_evidence(state: EvidenceState) -> dict:
    main: list[EvidenceItem] = list(state.get("main_evidence") or [])
    graph: list[EvidenceItem] = list(state.get("graph_evidence") or [])
    visual: list[EvidenceItem] = list(state.get("visual_evidence") or [])
    external: list[EvidenceItem] = list(state.get("external_evidence") or [])

    combined = main + graph + visual + external
    merged = dedupe_evidence(combined)

    # Provenance sanity: ensure every item has a provenance dict.
    for ev in merged:
        ev.setdefault("provenance", {})

    logger.debug(
        "merge: main=%d graph=%d visual=%d external=%d -> %d unique",
        len(main), len(graph), len(visual), len(external), len(merged),
    )
    return {"evidence": merged}
