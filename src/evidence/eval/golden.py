"""Golden QA set loader (PLAN.MD "Evaluation plan", Phase 5).

A golden set is a JSON/JSONL file of questions over *your* corpus with known
answers + expected source chunks, e.g.:

    [
      {
        "id": "q1",
        "question": "What BLEU score does the big Transformer achieve on WMT 2014 En-De?",
        "expected_chunks": ["transformer__abc#c2"],
        "expected_intent": "content",
        "should_escalate": false
      }
    ]

`expected_chunks` are chunk_ids (the unit of retrieval we evaluate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_chunks: list[str] = field(default_factory=list)
    expected_intent: str | None = None
    should_escalate: bool | None = None
    expected_answer: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> GoldenItem:
        known = {
            "id", "question", "expected_chunks", "expected_intent",
            "should_escalate", "expected_answer",
        }
        return cls(
            id=str(d.get("id", "")),
            question=str(d.get("question", "")),
            expected_chunks=[str(c) for c in d.get("expected_chunks", [])],
            expected_intent=d.get("expected_intent"),
            should_escalate=d.get("should_escalate"),
            expected_answer=d.get("expected_answer"),
            extra={k: v for k, v in d.items() if k not in known},
        )


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    """Load a golden set from a JSON array or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")
    raw = path.read_text(encoding="utf-8")

    items: list[dict] = []
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
    else:
        data = json.loads(raw)
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        if not isinstance(data, list):
            raise ValueError("golden set must be a JSON array (or {items: [...]})")
        items = data

    out = [GoldenItem.from_dict(d) for d in items if d.get("question")]
    # Ensure unique ids.
    seen: set[str] = set()
    for i, g in enumerate(out):
        if g.id in seen:
            g.id = f"{g.id or 'item'}_{i}"
        seen.add(g.id)
    return out
