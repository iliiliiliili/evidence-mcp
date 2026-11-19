"""LangGraph StateGraph builder (PLAN.MD "Stage-by-stage → LangGraph node mapping").

Phase 1 topology (text-only):

    START → classify_intent → rewrite_queries → retrieve_hybrid → rerank
          → merge_evidence → score_sufficiency ─┬─ sufficient → plan
                                               ├─ retry → (loop back) rewrite_queries
                                               └─ external → search_external*
          plan → map_evidence → draft → critique ─┬─ pass → verify_citations
                                                 ├─ revise → draft
                                                 └─ (exhausted) → verify_citations
          verify_citations → quality_check ─┬─ pass → END
                                           └─ fail → plan (≤ max_regenerations)

*`search_external` is Phase 3; when external is disabled the gate routes to
best-effort generation instead (handled inside `score_sufficiency`).
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import EvidenceConfig
from ..indexes.retriever import Retriever
from ..indexes.store import Store
from ..models.ollama import Models
from .nodes import (
    classify_intent,
    critique,
    draft,
    map_evidence,
    merge_evidence,
    plan,
    quality_check,
    rerank,
    retrieve_hybrid,
    rewrite_queries,
    score_sufficiency,
    verify_citations,
)
from .nodes.context import NodeContext, set_context
from .nodes.expand_graph import expand_via_graph
from .nodes.external import search_external
from .nodes.visual import retrieve_visual
from .state import EvidenceState

logger = logging.getLogger(__name__)


class Evidence:
    """Facade that wires config → models → indexes → graph, and runs queries."""

    def __init__(self, cfg: EvidenceConfig | None = None):
        self.cfg = cfg or EvidenceConfig.load()
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        index_dir = self.cfg.resolved_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        self.models = Models(self.cfg)
        self.store = Store(cfg.resolved_store_path)
        self.retriever = Retriever(index_dir, self.store, self.models, self.cfg.retrieval)

        # Dim discovery: use the stored dim if a prior build exists.
        dim = self.store.kv_get("index_dim")
        self.retriever.dim = int(dim) if dim else 1024

        self.ctx = NodeContext(
            cfg=self.cfg,
            models=self.models,
            store=self.store,
            retriever=self.retriever,
        )
        set_context(self.ctx)
        self.graph = self._build_graph()

    # ── conditional-edge routers ─────────────────────────────────────────
    def _route_after_sufficiency(self, state: EvidenceState) -> str:
        decision = state.get("sufficiency_decision", "sufficient")
        if decision == "retry":
            return "rewrite_queries"  # loop back to reformulate (drift-guarded)
        if decision == "external" and self.cfg.external.enabled:
            return "search_external"
        # "sufficient" (or external disabled) → proceed to generation.
        return "plan"

    def _route_after_external(self, state: EvidenceState) -> str:
        # External works were appended to `external_evidence`; re-merge with the
        # local evidence before generation.
        return "merge_evidence"

    def _route_after_critique(self, state: EvidenceState) -> str:
        verdict = (state.get("critic_verdict") or {}).get("verdict", "pass")
        cycles = int(state.get("critic_cycles", 0))
        max_cycles = self.cfg.generation.critic_max_cycles
        if verdict == "pass":
            return "verify_citations"
        if verdict == "revise" and cycles < max_cycles:
            return "draft"  # revise loop
        # fail / exhausted → proceed to citation verification (QC will gate).
        return "verify_citations"

    def _route_after_qc(self, state: EvidenceState) -> str:
        report = state.get("qc_report") or {}
        if report.get("passed"):
            return END
        # Failed and budget remains → regenerate from plan.
        regenerations = int(state.get("regenerations", 0))
        if regenerations < self.cfg.generation.max_regenerations:
            return "plan"
        return END

    def _build_graph(self):
        builder = StateGraph(EvidenceState)

        builder.add_node("classify_intent", classify_intent)
        builder.add_node("rewrite_queries", rewrite_queries)
        builder.add_node("retrieve_hybrid", retrieve_hybrid)
        builder.add_node("rerank", rerank)
        builder.add_node("merge_evidence", merge_evidence)
        builder.add_node("expand_via_graph", expand_via_graph)
        builder.add_node("retrieve_visual", retrieve_visual)
        builder.add_node("score_sufficiency", score_sufficiency)
        builder.add_node("plan", plan)
        builder.add_node("map_evidence", map_evidence)
        builder.add_node("draft", draft)
        builder.add_node("critique", critique)
        builder.add_node("search_external", search_external)
        builder.add_node("verify_citations", verify_citations)
        builder.add_node("quality_check", quality_check)

        # Linear spine. `expand_via_graph` and `retrieve_visual` are optional
        # no-ops when disabled (Phases 2 & 4); they append to their own
        # accumulators, which `merge_evidence` combines at node 10.
        builder.add_edge(START, "classify_intent")
        builder.add_edge("classify_intent", "rewrite_queries")
        builder.add_edge("rewrite_queries", "retrieve_hybrid")
        builder.add_edge("retrieve_hybrid", "rerank")
        builder.add_edge("rerank", "expand_via_graph")
        builder.add_edge("expand_via_graph", "retrieve_visual")
        builder.add_edge("retrieve_visual", "merge_evidence")
        builder.add_edge("merge_evidence", "score_sufficiency")

        # Sufficiency gate.
        builder.add_conditional_edges(
            "score_sufficiency",
            self._route_after_sufficiency,
            {
                "rewrite_queries": "rewrite_queries",
                "search_external": "search_external",
                "plan": "plan",
            },
        )
        # External search → re-merge evidence (external_evidence was appended).
        builder.add_conditional_edges(
            "search_external",
            self._route_after_external,
            {"merge_evidence": "merge_evidence"},
        )

        # Generation sub-graph.
        builder.add_edge("plan", "map_evidence")
        builder.add_edge("map_evidence", "draft")
        builder.add_edge("draft", "critique")
        builder.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {
                "draft": "draft",
                "verify_citations": "verify_citations",
            },
        )
        builder.add_edge("verify_citations", "quality_check")
        builder.add_conditional_edges(
            "quality_check",
            self._route_after_qc,
            {
                END: END,
                "plan": "plan",
            },
        )

        # Checkpointer: in-memory for now; swap for SqliteSaver/PostgresSaver
        # (PLAN.MD "Key LangGraph mechanics").
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    # ── public API ────────────────────────────────────────────────────────
    def ask(
        self,
        question: str,
        *,
        thread_id: str | None = None,
        stream: bool = False,
        allowed_groups: list[str] | None = None,
    ):
        """Run the full pipeline on a question.

        Returns the final state dict (or a generator of state updates if
        `stream=True`).

        `allowed_groups` (optional) restricts the evidence to papers whose
        group (folder name) is in the set — see `merge_evidence` for the
        filter. Used for comparative analysis across paper groups.
        """
        config = {"configurable": {"thread_id": thread_id or "default"}}
        input_state: EvidenceState = {
            "question": question,
            "retries": 0,
            "regenerations": 0,
            "critic_cycles": 0,
            "qc_passes": 0,
            "citations": [],
            "citations_dropped": [],
            "external_used": [],
            "retrieval_shortcomings": [],
            "allowed_groups": list(allowed_groups or []),
        }
        if stream:
            return self.graph.stream(input_state, config=config)
        return self.graph.invoke(input_state, config=config)


def build_graph(cfg: EvidenceConfig | None = None) -> Evidence:
    """Convenience factory."""
    return Evidence(cfg)
