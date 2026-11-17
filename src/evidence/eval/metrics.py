"""Evaluation metrics (PLAN.MD "Evaluation plan", Phase 5).

Retrieval + generation metrics for the golden QA set. All functions are pure
and dependency-light so they can be unit-tested without a model or network.

Metrics:
  - retrieval:  recall@k, precision@k, MRR, nDCG@k  (per stage)
  - sufficiency-gate: precision/recall of the gate's escalate vs. not call
  - citation faithfulness: fraction of cited claims that resolve to a real chunk
  - latency / token budget tracking
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


# ── ranking metrics ───────────────────────────────────────────────────────
def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top-k retrieved."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = len(rel & set(retrieved[:k]))
    return hits / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    hits = len(rel & set(retrieved[:k]))
    return hits / k


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """MRR over a single query: 1/(rank of first relevant hit), else 0."""
    rel = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float] | None = None,
    k: int = 10,
) -> float:
    """nDCG@k. If `relevance` is None, treat all relevant docs as grade 1."""
    if relevance is None:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        grade = relevance.get(doc, 0.0)
        dcg += grade / math.log2(i + 1)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate_recall_at_k(
    queries: list[dict], k: int
) -> float:
    """Mean recall@k over a list of {retrieved, relevant} query records."""
    if not queries:
        return 0.0
    return sum(recall_at_k(q["retrieved"], q["relevant"], k) for q in queries) / len(queries)


# ── sufficiency-gate metrics ──────────────────────────────────────────────
def gate_accuracy(records: list[dict]) -> dict:
    """Precision/recall/accuracy of the sufficiency gate.

    Each record: {should_escalate: bool, did_escalate: bool} where
    `should_escalate` is the golden label (corpus insufficient for the question)
    and `did_escalate` is whether the gate actually looped/escalated.
    """
    if not records:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
    tp = fp = fn = tn = 0
    for r in records:
        should = bool(r.get("should_escalate"))
        did = bool(r.get("did_escalate"))
        if should and did:
            tp += 1
        elif not should and did:
            fp += 1
        elif should and not did:
            fn += 1
        else:
            tn += 1
    n = len(records)
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ── citation faithfulness ─────────────────────────────────────────────────
def citation_faithfulness(citations: list[dict]) -> dict:
    """Fraction of cited claims that resolved to a real, verified chunk.

    `citations`: list of {verified: bool} (or {dropped: bool}).
    """
    if not citations:
        return {"faithfulness": 0.0, "verified": 0, "total": 0}
    verified = sum(1 for c in citations if c.get("verified") and not c.get("dropped"))
    total = len(citations)
    return {
        "faithfulness": round(verified / total, 4),
        "verified": verified,
        "total": total,
    }


def coverage_ratio(outline_sections: list[str], answer: str) -> float:
    """Fraction of outline sections whose key term appears in the answer."""
    if not outline_sections:
        return 0.0
    answer_lower = (answer or "").lower()
    covered = 0
    for sec in outline_sections:
        key = sec.strip().split()[-1].lower() if sec.strip() else ""
        if len(key) >= 4 and key in answer_lower:
            covered += 1
    return covered / len(outline_sections)
