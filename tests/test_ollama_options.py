"""Tests for the Ollama provider wrapper (option forwarding, timeouts)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from evidence.config import EvidenceConfig
from evidence.models.ollama import OllamaModel


class _FakeOllamaClient:
    """Records the last chat() kwargs so tests can assert option forwarding."""

    def __init__(self, *, host: str, timeout: float):
        self.host = host
        self.timeout = timeout
        self.last_chat: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> dict:
        self.last_chat = kwargs
        return {"message": {"content": '{"ok": true}'}}


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> _FakeOllamaClient:
    fake = _FakeOllamaClient(host="http://localhost:11434", timeout=300.0)
    fake_mod = SimpleNamespace(Client=lambda **_kw: fake)
    monkeypatch.setitem(sys.modules, "ollama", fake_mod)
    return fake


def test_complete_forwards_think_option(fake_ollama: _FakeOllamaClient) -> None:
    cfg = EvidenceConfig()
    cfg.ollama.think = False  # default
    model = OllamaModel(cfg)
    model.complete("hi", model="qwen3.5:9b", json=True)
    assert fake_ollama.last_chat["options"]["think"] is False
    assert fake_ollama.last_chat["options"]["temperature"] == 0.0
    assert fake_ollama.last_chat["format"] == "json"


def test_complete_respects_cfg_think_true(fake_ollama: _FakeOllamaClient) -> None:
    cfg = EvidenceConfig()
    cfg.ollama.think = True
    model = OllamaModel(cfg)
    model.complete("hi", model="qwen3.5:9b")
    assert fake_ollama.last_chat["options"]["think"] is True
    # No json format when json=False.
    assert fake_ollama.last_chat["format"] is None


def test_cfg_default_think_is_false() -> None:
    cfg = EvidenceConfig()
    assert cfg.ollama.think is False


class _FakeOllamaList:
    """Client that can list models and records pull() calls."""

    def __init__(self, models: list[str], *, pull_ok: bool = True, list_raises: bool = False):
        self.models = models
        self.pull_ok = pull_ok
        self.list_raises = list_raises
        self.last_chat: dict[str, Any] = {}
        self.pulled: list[str] = []

    def list(self) -> dict:
        # Real installed client exposes `list()`; the wrapper falls back to it
        # when `list_models()` is absent (which is the case here). Older clients
        # key each model as `model` (not `name`), so exercise that branch.
        if self.list_raises:
            raise RuntimeError("server down")
        return {"models": [{"model": m} for m in self.models]}

    def pull(self, model: str, stream: bool = True):
        if not self.pull_ok:
            raise RuntimeError("pull failed")
        self.pulled.append(model)
        if stream:
            return iter([{"status": "done"}])
        return {"status": "done"}

    def chat(self, **kwargs: Any) -> dict:
        self.last_chat = kwargs
        return {"message": {"content": '{"ok": true}'}}


def _cfg(auto_pull: bool, host: str = "http://localhost:11434") -> EvidenceConfig:
    cfg = EvidenceConfig()
    cfg.ollama.host = host
    cfg.ollama.auto_pull = auto_pull
    return cfg


def test_auto_pull_default_is_true() -> None:
    assert EvidenceConfig().ollama.auto_pull is True


def test_ensure_model_pulls_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaList(models=["other:7b"], pull_ok=True)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == ["qwen2.5:9b"]
    # A second call must not re-pull (memoized).
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == ["qwen2.5:9b"]


def test_ensure_model_skips_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaList(models=["qwen2.5:9b"])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == []


def test_ensure_model_no_pull_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaList(models=[])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=False))
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == []
    assert fake.models == []  # list was never even consulted for the pull path


def test_ensure_model_list_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A down/unlistable server must not block the call; no pull is attempted."""
    fake = _FakeOllamaList(models=[], list_raises=True)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    # Must not raise even though the model is absent.
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == []


def test_ensure_model_pull_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaList(models=[], pull_ok=False)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    # Pull fails but ensure_model swallows it so the real call can surface the error.
    model.ensure_model("qwen2.5:9b")
    assert fake.pulled == []


def test_complete_triggers_auto_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    """`complete()` pulls a missing model before issuing the chat request."""
    fake = _FakeOllamaList(models=[], pull_ok=True)
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    model.complete("hi", model="qwen2.5:9b", json=True)
    assert fake.pulled == ["qwen2.5:9b"]
    assert fake.last_chat["model"] == "qwen2.5:9b"


def test_complete_forwards_keep_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """`keep_alive` is forwarded to chat so Ollama keeps the model loaded."""
    fake = _FakeOllamaList(models=["qwen2.5:9b"])
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    cfg = _cfg(auto_pull=False)
    cfg.ollama.keep_alive = "168h"
    OllamaModel(cfg).complete("hi", model="qwen2.5:9b")
    assert fake.last_chat["keep_alive"] == "168h"


def test_embed_triggers_auto_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaList(models=[], pull_ok=True)
    seen: dict[str, Any] = {}

    def embed(model, input, keep_alive=None):
        seen["keep_alive"] = keep_alive
        return {"embeddings": [[0.0] for _ in input]}

    fake.embed = embed
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **_: fake))
    model = OllamaModel(_cfg(auto_pull=True))
    model.embed(["a", "b"], model="bge-m3")
    assert fake.pulled == ["bge-m3"]
    assert seen["keep_alive"] == "1h"  # default keep_alive forwarded to embed


def test_cfg_default_request_timeout_is_300s() -> None:
    cfg = EvidenceConfig()
    assert cfg.ollama.request_timeout == 300.0
