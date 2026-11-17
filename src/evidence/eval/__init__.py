"""Evaluation utilities (PLAN.MD Phase 5 + "Evaluation plan")."""

from .faithfulness import extract_citation_markers, spot_check_citations
from .golden import GoldenItem, load_golden_set
from .metrics import (
    aggregate_recall_at_k,
    citation_faithfulness,
    coverage_ratio,
    gate_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .runner import ItemReport, run_eval

__all__ = [
    "extract_citation_markers",
    "spot_check_citations",
    "GoldenItem",
    "load_golden_set",
    "aggregate_recall_at_k",
    "citation_faithfulness",
    "coverage_ratio",
    "gate_accuracy",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "ItemReport",
    "run_eval",
]
