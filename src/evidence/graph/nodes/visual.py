"""Stage 6 — Visual document retrieval (PLAN.MD, Phase 4).

ColSmol patch embeddings + MUVERA FDE index over page renders → ANN top-100 →
MaxSim rerank → top 5 pages. Disabled by default (`visual.enabled=False`) and
requires the `vision` extra (torch + transformers). When unavailable, the node
is a no-op.

If MUVERA is too heavy, fall back to plain MaxSim over a smaller page set, or
CLIP page embeddings as a lower-quality stopgap (PLAN.MD "Key risks").
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context
from .evidence_util import external_evidence  # noqa: F401  (visual uses its own path)

logger = logging.getLogger(__name__)


def retrieve_visual(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    if not cfg.visual.enabled:
        return {}

    try:
        import faiss  # type: ignore
        import numpy as np  # noqa: F401
    except ImportError:
        logger.info("faiss not available; skipping visual retrieval")
        return {}

    index_path = ctx.cfg.resolved_index_dir / "pages_muvera.faiss"
    if not index_path.exists():
        logger.info("no MUVERA page index at %s; skipping visual retrieval", index_path)
        return {}

    visual_query = state.get("visual_query") or state.get("question", "")
    try:
        # Embed the visual query (text → ColSmol/MUVERA query vector).
        q_vec = ctx.models.embed_one(visual_query)
        index = faiss.read_index(str(index_path))
        k = min(cfg.visual.candidates, index.ntotal)
        arr = np.asarray([q_vec], dtype="float32")
        faiss.normalize_L2(arr)
        scores, idxs = index.search(arr, k)
        top = list(zip(idxs[0], scores[0], strict=True))[: cfg.visual.top_pages]
    except Exception as exc:  # noqa: BLE001 - visual path is optional
        logger.info("visual retrieval failed: %s", exc)
        return {}

    # Map page ids back to (paper_id, page) via the store's kv table.
    evidence: list[EvidenceItem] = []
    for i, (pid, score) in enumerate(top):
        if pid < 0:
            continue
        meta_key = f"page::{int(pid)}"
        page_meta = ctx.store.kv_get(meta_key)
        if not page_meta:
            continue
        import json

        try:
            meta = json.loads(page_meta)
        except json.JSONDecodeError:
            continue
        ev = external_evidence(
            work_id=meta.get("page_id", str(pid)),
            title=f"{meta.get('paper_title','')}: page {meta.get('page')}",
            abstract=(meta.get("text") or "")[:400],
            source="visual",
            relevance=float(min(1.0, 0.4 + score * 0.5)),
            provenance={"visual_rank": i + 1, "query": visual_query},
        )
        ev["image"] = meta.get("image_path", "")
        ev["page"] = int(meta.get("page", 0))
        ev["paper_id"] = meta.get("paper_id", "")
        evidence.append(ev)

    if evidence:
        logger.info("retrieve_visual: %d page(s)", len(evidence))
    return {"visual_evidence": evidence, "visual_pages": [e["chunk_id"] for e in evidence]}
