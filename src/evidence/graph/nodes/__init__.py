"""LangGraph node implementations.

Each node is a plain function `def node(state: EvidenceState) -> dict` (partial
state update). Nodes receive a dependency bundle via a module-level registry so
they stay pure and testable. See `context.py` for the registry.
"""

from .citations import verify_citations
from .classify import classify_intent
from .generate import critique, draft
from .merge import merge_evidence
from .plan import map_evidence, plan
from .qc import quality_check
from .rerank import rerank
from .retrieve import retrieve_hybrid
from .rewrite import rewrite_queries
from .sufficiency import score_sufficiency

__all__ = [
    "classify_intent",
    "rewrite_queries",
    "retrieve_hybrid",
    "rerank",
    "score_sufficiency",
    "merge_evidence",
    "verify_citations",
    "plan",
    "map_evidence",
    "draft",
    "critique",
    "quality_check",
]
