"""Typed, checkpointed LangGraph state for the EvidenceItem query pipeline.

The `EvidenceItem` record is the single currency that flows through every
retrieval path (dense, sparse, graph, visual, external). Every item carries
provenance so the citation-integrity stage and the QC stage can verify and
audit claim → chunk mappings (PLAN.MD stages 11 & 13).
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


def _replace(existing: Any, new: Any) -> Any:
    """Reducer that REPLACES instead of accumulating.

    LangGraph's default for list/dict fields is to merge (append/extend), which
    is what we want for the true accumulators (`evidence`'s sibling channels)
    but NOT for per-pass working state that must reset on each retry/regenerate.
    """
    return new

# Source labels (PLAN.MD `EvidenceItem.source`).
SourceKind = Literal[
    "dense",
    "sparse",
    "rrf",
    "graph",
    "visual",
    "openalex",
    "s2",
    "crossref",
]

SufficiencyDecision = Literal["sufficient", "retry", "external"]


class EvidenceItem(TypedDict, total=False):
    """A single unit of retrieved evidence.

    `text` and `image` are mutually exclusive in practice (text chunk vs. page
    render) but both are present on the type so downstream consumers can
    branch on availability.
    """

    chunk_id: str
    paper_id: str
    title: str
    page: int
    text: str
    image: str  # path to a page render (visual evidence)
    source: SourceKind
    relevance: float
    # How it was retrieved, for auditability (query variant, rank, provider…).
    provenance: dict[str, Any]
    # Filled in by the citation-integrity stage.
    doi: str | None
    url: str | None
    year: int | None
    authors: list[str]
    # Verified claim mapping (stage 11). Empty until verification succeeds.
    verified: bool
    # Paper's group, derived from the immediate parent folder name of the
    # paper's file path. Papers in the same folder form a group — useful for
    # comparative questions ("what does group A conclude vs group B?").
    # Convention, not config: to group papers, put them in subfolders of the
    # corpus. External (OpenAlex/S2/Crossref) items get the literal "external".
    group: str


class EvidenceState(TypedDict, total=False):
    """Full query-graph state.

    Reducer notes (critical for the retry/regenerate loops):
      - True accumulators (`retrieval_shortcomings`, `external_used`, and the
        parallel-path evidence channels) use `add` so they persist across passes.
      - Per-pass working state (`evidence`, `citations`, `outline`, `draft`, …)
        uses `_replace` so each retry/regenerate pass starts fresh instead of
        duplicating the previous pass's output.
    """

    # ── Inputs ────────────────────────────────────────────────────────────
    question: str
    intent: str  # "content" | "bibliometric" | "trend" | "current_world"
    text_queries: Annotated[list[str], _replace]
    visual_query: str | None
    # Optional paper-group constraint (see EvidenceItem.group). When non-empty,
    # `merge_evidence` filters to items whose group is in this set, so the
    # downstream generation sub-graph only sees evidence from the requested
    # group(s). Empty / absent = no filter.
    allowed_groups: Annotated[list[str], _replace]

    # ── Loop control ──────────────────────────────────────────────────────
    retries: int
    retrieval_shortcomings: Annotated[list[str], add]  # gate → stage-8 loop
    external_used: Annotated[list[dict[str, Any]], add]  # fetched works + audit

    # ── Retrieval outputs ─────────────────────────────────────────────────
    # `main_evidence` is the text path's working set (replaced each pass by
    # `retrieve_hybrid`); `evidence` is the FINAL merged set produced by
    # `merge_evidence` (main + graph + visual + external, deduped). The
    # parallel paths (graph / visual / external, Phases 2–4) each append to
    # their own accumulator so fan-out from node 3 merges cleanly at node 10.
    main_evidence: Annotated[list[EvidenceItem], _replace]
    evidence: Annotated[list[EvidenceItem], _replace]
    graph_evidence: Annotated[list[EvidenceItem], add]
    visual_evidence: Annotated[list[EvidenceItem], add]
    external_evidence: Annotated[list[EvidenceItem], add]
    dense_ranked: Annotated[list[str], _replace]  # chunk_ids in dense order
    sparse_ranked: Annotated[list[str], _replace]
    rrf_ranked: Annotated[list[str], _replace]
    reranked: Annotated[list[str], _replace]
    expanded_chunk_ids: Annotated[list[str], add]
    visual_pages: Annotated[list[str], add]  # page ids / paths

    # ── Gates ─────────────────────────────────────────────────────────────
    sufficiency: Annotated[dict[str, Any], _replace]  # rubric scores + rationale
    sufficiency_decision: SufficiencyDecision

    # ── Generation (stage 12 sub-graph) ───────────────────────────────────
    # All replaced each pass; counters (critic_cycles, qc_passes,
    # regenerations) and citations_dropped accumulate for loop-safety + audit.
    outline: Annotated[dict[str, Any], _replace]
    evidence_map: Annotated[dict[str, Any], _replace]
    draft: Annotated[str, _replace]
    critic_verdict: Annotated[dict[str, Any], _replace]
    critic_cycles: int  # accumulator (loop-safety)
    citations: Annotated[list[dict[str, Any]], _replace]  # claim → chunk map
    citations_dropped: Annotated[list[dict[str, Any]], add]  # audit trail

    # ── QC (stage 13) ─────────────────────────────────────────────────────
    qc_report: Annotated[dict[str, Any], _replace]
    qc_passes: int  # accumulator
    regenerations: int  # accumulator (loop-safety)

    # ── Final ─────────────────────────────────────────────────────────────
    answer: Annotated[str, _replace]
    error: str | None
    # Human-in-the-loop: set to block before external search (stage 9).
    await_approval: bool
