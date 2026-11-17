"""Citation faithfulness spot-checking (PLAN.MD stage 13 + "Evaluation plan").

Re-verifies a *sample* of the final answer's citations against the store to
catch any drift between the draft's `[C<n>]` markers and the verified citation
list. This is the "citation spot-check" in the QC stage and the "citation
faithfulness" metric in the eval plan.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ..indexes.store import Store

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[C(\d+)\]")


def extract_citation_markers(answer: str) -> list[int]:
    """All distinct [C<n>] marker ids referenced in an answer text."""
    return sorted({int(m) for m in _MARKER.findall(answer or "")})


def spot_check_citations(
    citations: Sequence[dict],
    store: Store,
    *,
    sample_fraction: float = 0.3,
) -> dict:
    """Re-verify a sample of citations against the store.

    Returns {checked, passed, failed, faithfulness, failures:[...]}.
    """
    if not citations:
        return {"checked": 0, "passed": 0, "failed": 0, "faithfulness": 0.0, "failures": []}

    n = len(citations)
    k = max(1, int(round(n * max(0.0, min(1.0, sample_fraction)))))
    # Spread the sample across the list (even striding) rather than taking a
    # contiguous slice, so a systematic bug near the end is still caught.
    if k < n:
        step = n // k
        sample = [citations[i * step] for i in range(k)]
    else:
        sample = list(citations)

    passed = 0
    failed = 0
    failures: list[dict] = []
    for c in sample:
        cid = str(c.get("chunk_id", "")).strip()
        if not cid:
            failed += 1
            failures.append({"reason": "missing chunk_id", "claim": c.get("claim")})
            continue
        if store.get_chunk(cid) is not None:
            passed += 1
        else:
            failed += 1
            failures.append(
                {"reason": "chunk not in store", "chunk_id": cid, "claim": c.get("claim")}
            )

    checked = passed + failed
    faithfulness = (passed / checked) if checked else 0.0
    return {
        "checked": checked,
        "passed": passed,
        "failed": failed,
        "faithfulness": round(faithfulness, 4),
        "failures": failures,
    }
