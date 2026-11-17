"""Exponential backoff with jitter for external API calls (PLAN.MD stage 9).

OpenAlex's polite pool uses `mailto`; Semantic Scholar is hard-capped at
~1 RPS so we serialize + back off; Crossref is generous but we still cap.
"""

from __future__ import annotations

import random
import time


def backoff(attempt: int, *, base: float = 1.0, cap: float = 60.0, jitter: float = 0.5) -> float:
    """Sleep with exponential backoff + jitter, returning the delay used."""
    delay = min(cap, base * (2 ** attempt))
    delay *= 1.0 + random.uniform(-jitter, jitter)
    time.sleep(max(0.0, delay))
    return delay


def retry_on_error(
    fn,
    *,
    attempts: int = 3,
    base: float = 1.0,
    cap: float = 60.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """Call `fn()`, retrying on `httpx.HTTPStatusError` with a retryable status."""
    import httpx

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code not in retry_statuses or attempt == attempts - 1:
                raise
            backoff(attempt, base=base, cap=cap)
    raise last_exc  # pragma: no cover - unreachable
