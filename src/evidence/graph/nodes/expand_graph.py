"""Stage 5 — Graph-guided chunk expansion (PLAN.MD, Phase 2).

For each top chunk's paper, traverse Neo4j `SHARES_TOPIC` / `SHARES_METHOD`
edges to pull qualifying chunks from related papers. Budget-limited
(max +`graph_expansion.max_additional_chunks`).

Disabled by default (`graph_expansion.enabled=False`) and requires a Neo4j
instance. When unavailable, the node is a no-op so the rest of the pipeline
still runs (the `expand_via_graph` node simply returns `{}`).
"""

from __future__ import annotations

import logging

from ..state import EvidenceItem, EvidenceState
from .context import get_context
from .evidence_util import evidence_from_chunk

logger = logging.getLogger(__name__)

# Neo4j Cypher: for a paper, find related papers via shared topic/method edges.
_CYPHER = """
MATCH (p:Paper {paper_id: $paper_id})
  -[r:ON_TOPIC|USES_METHOD]-(shared:Topic|Method)
  -[r2:ON_TOPIC|USES_METHOD]-(rel:Paper)
WHERE rel.paper_id <> p.paper_id
RETURN DISTINCT rel.paper_id AS paper_id, shared.name AS shared
LIMIT 10
"""


def expand_via_graph(state: EvidenceState) -> dict:
    ctx = get_context()
    cfg = ctx.cfg
    if not cfg.graph_expansion.enabled:
        return {}

    try:
        import neo4j  # type: ignore
    except ImportError:
        logger.info("neo4j package not installed; skipping graph expansion")
        return {}

    evidence: list[EvidenceItem] = list(state.get("evidence") or [])
    if not evidence:
        return {}

    # Take the top-N papers' paper_ids and expand.
    top_paper_ids = list(dict.fromkeys(ev.get("paper_id", "") for ev in evidence[:5]))
    added: list[EvidenceItem] = []
    budget = cfg.graph_expansion.max_additional_chunks

    driver = None
    try:
        uri = "bolt://localhost:7687"  # override via env in production
        password = "evidence"
        driver = neo4j.GraphDatabase.driver(uri, auth=("neo4j", password))
        for pid in top_paper_ids:
            if len(added) >= budget:
                break
            with driver.session() as session:
                res = session.run(_CYPHER, paper_id=pid)
                for rec in res:
                    if len(added) >= budget:
                        break
                    related_paper = ctx.store.get_paper(rec["paper_id"])
                    if related_paper is None:
                        continue
                    for ch in ctx.store.chunks_for_paper(rec["paper_id"])[:3]:
                        if len(added) >= budget:
                            break
                        ev = evidence_from_chunk(
                            ctx.store,
                            ch.chunk_id,
                            source="graph",
                            relevance=0.5,
                            provenance={"via_paper": pid, "shared": rec["shared"]},
                        )
                        if ev is not None:
                            added.append(ev)
    except Exception as exc:  # noqa: BLE001 - graph expansion is optional
        logger.info("graph expansion unavailable: %s", exc)
        added = []
    finally:
        if driver:
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass

    if added:
        logger.info("expand_via_graph: added %d chunks", len(added))
    return {"graph_evidence": added}
