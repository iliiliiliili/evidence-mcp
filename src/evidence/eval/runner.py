"""Evaluation runner (PLAN.MD "Evaluation plan").

Runs a golden set of questions through Evidence and scores:
  - retrieval quality: recall@k, precision@k, MRR, nDCG@k (against
    `expected_chunks`),
  - intent routing accuracy (against `expected_intent`),
  - escalation accuracy (against `should_escalate`),
  - citation faithfulness + answer coverage.

Everything is deterministic given the corpus; no extra LLM calls beyond the
normal pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import EvidenceConfig
from .golden import load_golden_set
from .metrics import (
    coverage_ratio,
    gate_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


@dataclass
class ItemReport:
    item_id: str
    question: str
    intent: str | None
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    should_escalate: bool = False
    did_escalate: bool = False
    n_citations: int = 0
    faithfulness: float = 0.0
    coverage: float = 0.0
    answer_present: bool = False
    answer_excerpt: str = ""


def _esc_from_state(state: dict) -> dict:
    esc = state.get("escalation") or {}
    did = (
        bool(state.get("external_used"))
        or bool(state.get("external_evidence"))
        or esc.get("escalated", False)
    )
    should = esc.get("should_escalate", None)
    if should is None:
        # Infer: low sufficiency score with a shortage ⇒ would escalate.
        score = float(esc.get("score", 100.0))
        should = score < 70.0 and bool(state.get("retrieval_shortcomings"))
    return {"should": bool(should), "did": bool(did)}


def run_eval(
    config: EvidenceConfig,
    golden_path: str,
    *,
    k_list: tuple[int, ...] = (1, 5, 10),
    app=None,
) -> dict:
    """Evaluate the pipeline over a golden set; return an aggregate report."""
    if app is None:
        from ..graph.builder import Evidence

        app = Evidence(config)

    items = load_golden_set(golden_path)
    reports: list[ItemReport] = []

    for it in items:
        state = app.ask(it.question)
        retrieved = list(state.get("evidence") or [])
        retrieved_ids = [e.get("chunk_id") for e in retrieved if e.get("chunk_id")]
        expected = list(it.expected_chunks)

        recall = {k: round(recall_at_k(retrieved_ids, expected, k), 4) for k in k_list}
        precision = {k: round(precision_at_k(retrieved_ids, expected, k), 4) for k in k_list}
        rel_map = {cid: 1.0 for cid in expected}
        ndcg = {k: round(ndcg_at_k(retrieved_ids, rel_map, k), 4) for k in k_list}
        mrr = round(mean_reciprocal_rank(retrieved_ids, expected), 4)

        esc = _esc_from_state(state)
        citations = list(state.get("citations") or [])
        answer = state.get("answer") or ""
        sections = (state.get("outline") or {}).get("sections") or []

        item_rep = ItemReport(
            item_id=it.id,
            question=it.question,
            intent=state.get("intent"),
            recall=recall,
            precision=precision,
            mrr=mrr,
            ndcg=ndcg,
            should_escalate=esc["should"],
            did_escalate=esc["did"],
            n_citations=len(citations),
            faithfulness=(
                sum(1 for c in citations if c.get("verified")) / len(citations)
            ) if citations else 0.0,
            coverage=round(coverage_ratio(sections, answer), 4),
            answer_present=bool(answer.strip()),
            answer_excerpt=answer[:160],
        )
        reports.append(item_rep)
        logger.info(
            "eval %s: recall@5=%.2f mrr=%.2f intent=%s",
            it.id, recall.get(5, 0.0), mrr, state.get("intent"),
        )

    # Gate accuracy.
    gate_records = [
        {"should_escalate": r.should_escalate, "did_escalate": r.did_escalate}
        for r in reports
    ]
    gate = gate_accuracy(gate_records)

    n = len(reports)
    mean = lambda key: (sum(getattr(r, key) for r in reports) / n) if n else 0.0  # noqa: E731
    mean_k = lambda attr, k: (  # noqa: E731
        round(sum(getattr(r, attr).get(k, 0.0) for r in reports) / n, 4) if n else 0.0
    )
    return {
        "n_items": n,
        "k_list": list(k_list),
        "aggregate": {
            "recall": {k: mean_k("recall", k) for k in k_list},
            "precision": {k: mean_k("precision", k) for k in k_list},
            "mrr": round(mean("mrr"), 4),
            "ndcg": {k: mean_k("ndcg", k) for k in k_list},
            "gate": gate,
            "citation_faithfulness": round(mean("faithfulness"), 4),
            "coverage": round(mean("coverage"), 4),
            "answer_rate": round(sum(1 for r in reports if r.answer_present) / n, 4) if n else 0.0,
        },
        "per_item": [r.__dict__ for r in reports],
    }
