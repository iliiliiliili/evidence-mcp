import sys

import pytest

# `evidence.external` re-exports the `backoff` *function*, which shadows the
# submodule name. Import the module object explicitly to patch its internals.
import evidence.external.backoff  # noqa: F401

bo = sys.modules["evidence.external.backoff"]


def _err_429():
    import httpx

    request = httpx.Request("GET", "https://x")
    response = httpx.Response(429, request=request)
    return httpx.HTTPStatusError("429", request=request, response=response)


def test_backoff_exponential_and_capped(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(bo.time, "sleep", lambda d: sleeps.append(d))
    bo.backoff(0, base=1.0, cap=100.0, jitter=0.0)
    bo.backoff(1, base=1.0, cap=100.0, jitter=0.0)
    bo.backoff(2, base=1.0, cap=100.0, jitter=0.0)
    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_respects_cap(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(bo.time, "sleep", lambda d: sleeps.append(d))
    bo.backoff(20, base=1.0, cap=5.0, jitter=0.0)
    assert sleeps == [5.0]


def test_retry_on_error_retries_then_succeeds(monkeypatch):
    import httpx  # noqa: F401

    monkeypatch.setattr(bo, "backoff", lambda *a, **k: None)  # no real sleep
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _err_429()
        return "ok"

    assert bo.retry_on_error(flaky, attempts=3) == "ok"
    assert attempts["n"] == 3


def test_retry_on_error_gives_up_after_attempts(monkeypatch):
    import httpx  # noqa: F401

    monkeypatch.setattr(bo, "backoff", lambda *a, **k: None)

    def always_429():
        raise _err_429()

    with pytest.raises(httpx.HTTPStatusError):
        bo.retry_on_error(always_429, attempts=2)
