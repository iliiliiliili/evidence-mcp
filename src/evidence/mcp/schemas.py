"""MCP return-shape schemas (new plan: `mcp/schemas.py`).

These TypedDicts document the *contract* every tool returns so client agents
can rely on stable field names. They are converted to plain dicts at the
boundary (FastMCP serializes return values as JSON), so the schemas are
primarily documentation + a lightweight self-check.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Citation(TypedDict, total=False):
    """One cited chunk (the workhorse return shape for retrieval tools)."""

    chunk_id: str
    text: str
    paper: str
    year: int | None
    section: str | None
    page: int | None
    doi: str | None
    relevance: float
    source: str  # "local" | "graph" | "graph-fallback"


class Work(TypedDict, total=False):
    """One external literature work (OpenAlex / S2 / Crossref)."""

    paper: str
    year: int | None
    doi: str | None
    cited_by: int | None
    oa_pdf: str | None
    abstract: str
    source: str  # "openalex" | "s2" | "crossref"


class EvidenceScore(TypedDict, total=False):
    """`check_evidence` result: 100-pt rubric + verdict."""

    total: float
    dimensions: dict[str, float]
    verdict: str  # sufficient | retry
    threshold: float
    rationale: dict[str, str]
    suggestions: list[str]
    n_evidence: int


class VerifyResult(TypedDict, total=False):
    """`verify_claim` result: SUPPORTED | PARTIAL | UNSUPPORTED + chunk text."""

    claim: str
    chunk_id: str
    verdict: str
    score: float
    text: str | None
    paper_title: str
    page: int | None
    doi: str | None


class AnswerResult(TypedDict, total=False):
    """`answer_research` result: cited answer + evidence map + QC."""

    answer: str
    intent: str
    citations: list[dict[str, Any]]
    citations_dropped: list[dict[str, Any]]
    sufficiency: dict[str, Any]
    evidence_map: dict[str, Any]
    qc_report: dict[str, Any]
    answer_rate: float


__all__ = ["Citation", "Work", "EvidenceScore", "VerifyResult", "AnswerResult"]
