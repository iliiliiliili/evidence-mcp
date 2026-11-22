"""Tests for multiple Ollama hosts + host selection (config, CLI, model client).

Covers:
  - ``OllamaConfig.active_host_url()`` resolution (named hosts, active_host,
    legacy single-host fallback);
  - the container-field guard in ``EvidenceConfig.load`` (a flat env string for
    the ``hosts`` map must NOT crash loading);
  - ``OllamaModel`` building its client against the resolved URL;
  - CLI: global ``--host`` selection, the ``evidence hosts`` command, and the
    "which host" banner when >1 host is defined.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from evidence import cli
from evidence.cli import app
from evidence.config import EvidenceConfig
from evidence.models import Models, OllamaModel, OpenAIModel

runner = CliRunner()


def _cfg_with_hosts(hosts: dict[str, str], active: str | None = None) -> EvidenceConfig:
    cfg = EvidenceConfig()
    cfg.ollama.hosts = dict(hosts)
    cfg.ollama.active_host = active
    return cfg


# ── config resolution ───────────────────────────────────────────────────────
def test_active_host_url_falls_back_to_legacy_single_host():
    cfg = EvidenceConfig()
    cfg.ollama.host = "http://single:11434"
    assert cfg.ollama.hosts == {}
    assert cfg.ollama.active_host_url() == "http://single:11434"


def test_active_host_url_uses_first_named_host_by_default():
    cfg = _cfg_with_hosts({"a": "http://a:1", "b": "http://b:2"})
    assert cfg.ollama.active_host_url() == "http://a:1"


def test_active_host_url_respects_active_host():
    cfg = _cfg_with_hosts({"a": "http://a:1", "b": "http://b:2"}, active="b")
    assert cfg.ollama.active_host_url() == "http://b:2"


def test_active_host_url_ignores_unknown_active_name():
    cfg = _cfg_with_hosts({"a": "http://a:1"}, active="does-not-exist")
    # Unknown active name → fall back to the first defined host.
    assert cfg.ollama.active_host_url() == "http://a:1"


def test_host_names_preserves_definition_order():
    cfg = _cfg_with_hosts({"b": "u", "a": "v", "c": "w"})
    assert cfg.ollama.host_names() == ["b", "a", "c"]


def test_named_hosts_take_precedence_over_legacy_host():
    cfg = EvidenceConfig()
    cfg.ollama.host = "http://legacy:9999"
    cfg.ollama.hosts = {"gpu-a": "http://10.0.0.5:6006"}
    assert cfg.ollama.active_host_url() == "http://10.0.0.5:6006"


# ── container-field env guard ───────────────────────────────────────────────
def test_env_override_for_hosts_map_does_not_crash(monkeypatch):
    # A flat env string can't be a name→URL map; it must be ignored, not crash.
    monkeypatch.setenv("EVIDENCE_OLLAMA_HOSTS", "gpu-a")
    # active_host (a scalar) should still work.
    monkeypatch.setenv("EVIDENCE_OLLAMA_ACTIVE_HOST", "gpu-a")
    cfg = EvidenceConfig.load(None)
    assert cfg.ollama.hosts == {}  # not coerced from the string
    assert cfg.ollama.active_host == "gpu-a"


# ── model client uses the resolved URL ─────────────────────────────────────
class _RecordingClient:
    def __init__(self, *, host: str, timeout: float):
        self.host = host
        self.timeout = timeout

    def chat(self, **kwargs: Any) -> dict:
        return {"message": {"content": '{"ok": true}'}}


def test_ollama_model_client_uses_resolved_host(monkeypatch):
    captured: dict[str, str] = {}

    def factory(**kw):
        captured["host"] = kw["host"]
        return _RecordingClient(**kw)

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=factory))

    cfg = _cfg_with_hosts({"a": "http://a:1", "b": "http://b:2"}, active="b")
    OllamaModel(cfg)._client_inner()
    assert captured["host"] == "http://b:2"


def test_ollama_model_client_uses_legacy_host_when_no_named(monkeypatch):
    captured: dict[str, str] = {}

    def factory(**kw):
        captured["host"] = kw["host"]
        return _RecordingClient(**kw)

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=factory))
    cfg = EvidenceConfig()
    cfg.ollama.host = "http://legacy:11434"
    OllamaModel(cfg)._client_inner()
    assert captured["host"] == "http://legacy:11434"


# ── CLI: host selection + banner + hosts command ────────────────────────────
def _write_config(tmp_path, monkeypatch, hosts: dict[str, str], active: str | None = None) -> None:
    import yaml

    cfg_file = tmp_path / "config.yaml"
    payload = {
        "data_dir": str(tmp_path / ".evidence"),
        "ollama": {
            "host": "http://default:11434",
            "reasoning": "m",
            "light": "m",
            "embed": "m",
            "hosts": hosts,
            **({"active_host": active} if active else {}),
        },
    }
    cfg_file.write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(cli, "_ACTIVE_HOST_OVERRIDE", None)
    return str(cfg_file)


def test_cli_hosts_command_lists_hosts(tmp_path, monkeypatch):
    cfg_file = _write_config(
        tmp_path, monkeypatch,
        {"gpu-a": "http://10.0.0.5:6006", "gpu-b": "http://10.0.0.6:6006"},
        active="gpu-a",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts", "--config", cfg_file])
    assert result.exit_code == 0, result.output
    assert "gpu-a" in result.output
    assert "gpu-b" in result.output
    assert "http://10.0.0.5:6006" in result.output


def test_cli_hosts_command_no_hosts_uses_single(tmp_path, monkeypatch):
    cfg_file = _write_config(tmp_path, monkeypatch, {})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts", "--config", cfg_file])
    assert result.exit_code == 0, result.output
    assert "http://default:11434" in result.output


def test_cli_global_host_selects_named_host(tmp_path, monkeypatch):
    cfg_file = _write_config(
        tmp_path, monkeypatch,
        {"gpu-a": "http://10.0.0.5:6006", "gpu-b": "http://10.0.0.6:6006"},
    )
    monkeypatch.chdir(tmp_path)
    # Select gpu-b via the global flag; the banner must reflect it.
    result = runner.invoke(
        app, ["--host", "gpu-b", "hosts", "--config", cfg_file]
    )
    assert result.exit_code == 0, result.output
    assert "gpu-b" in result.output
    # Banner lists the host in use (stderr is captured into output by runner).
    assert "10.0.0.6:6006" in result.output


def test_cli_global_host_adhoc_url(tmp_path, monkeypatch):
    cfg_file = _write_config(
        tmp_path, monkeypatch, {"gpu-a": "http://10.0.0.5:6006"}
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["--host", "http://192.168.1.20:11434", "hosts", "--config", cfg_file],
    )
    assert result.exit_code == 0, result.output
    assert "192.168.1.20:11434" in result.output


def test_cli_banner_shown_only_when_multiple_hosts(tmp_path, monkeypatch, capsys):
    # Single host → no banner line about "of N defined".
    cfg_file = _write_config(tmp_path, monkeypatch, {"only": "http://x:1"})
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts", "--config", cfg_file])
    assert "of 1 defined" not in result.output

    # Multiple hosts → banner present.
    cfg_file2 = _write_config(
        tmp_path, monkeypatch, {"a": "http://a:1", "b": "http://b:2"}
    )
    monkeypatch.setattr(cli, "_ACTIVE_HOST_OVERRIDE", None)
    result2 = runner.invoke(app, ["hosts", "--config", cfg_file2])
    assert "of 2 defined" in result2.output


def test_apply_host_override_name_and_url():
    cfg = _cfg_with_hosts({"a": "http://a:1", "b": "http://b:2"})
    cli._ACTIVE_HOST_OVERRIDE = "b"
    try:
        cli._apply_host_override(cfg)
    finally:
        cli._ACTIVE_HOST_OVERRIDE = None
    assert cfg.ollama.active_host == "b"
    assert cfg.ollama.active_host_url() == "http://b:2"

    # Ad-hoc URL becomes a transient host.
    cfg2 = _cfg_with_hosts({"a": "http://a:1"})
    cli._ACTIVE_HOST_OVERRIDE = "http://9.9.9.9:6006"
    try:
        cli._apply_host_override(cfg2)
    finally:
        cli._ACTIVE_HOST_OVERRIDE = None
    assert cfg2.ollama.active_host_url() == "http://9.9.9.9:6006"


# ── CLI: hosts-add (persistent write) ────────────────────────────────────────
def test_hosts_add_creates_and_persists(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"data_dir": str(tmp_path / ".evidence")}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["hosts-add", "gpu-c", "http://10.0.0.7:6006", "--config", str(cfg_file)]
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["hosts"]["gpu-c"] == "http://10.0.0.7:6006"
    assert data["ollama"].get("active_host") is None  # not activated by default


def test_hosts_add_replaces_existing_name(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"ollama": {"hosts": {"gpu-c": "http://old:11434"}}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["hosts-add", "gpu-c", "http://10.0.0.9:6006", "--config", str(cfg_file)],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["hosts"] == {"gpu-c": "http://10.0.0.9:6006"}


def test_hosts_add_activate_sets_default(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {"ollama": {"hosts": {"a": "http://a:1", "b": "http://b:2"}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["hosts-add", "c", "http://c:3", "--activate", "--config", str(cfg_file)],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["hosts"]["c"] == "http://c:3"
    assert data["ollama"]["active_host"] == "c"
    # The new active host now resolves to the new URL.
    assert EvidenceConfig.load(str(cfg_file)).ollama.active_host_url() == "http://c:3"


def test_hosts_add_without_config_creates_under_evidence(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts-add", "solo", "http://only:11434"])
    assert result.exit_code == 0, result.output
    created = tmp_path / ".evidence" / "config.yaml"
    assert created.exists()
    data = yaml.safe_load(created.read_text(encoding="utf-8"))
    assert data["ollama"]["hosts"]["solo"] == "http://only:11434"


# ── host TYPES (openai) + per-ROLE routing ──────────────────────────────────
def _cfg_rich() -> EvidenceConfig:
    cfg = EvidenceConfig()
    o = cfg.ollama
    o.hosts = {"chat": "http://chat:6006", "embeds": "http://embeds:8080"}
    o.host_types = {"embeds": "openai"}
    o.reasoning = "llama3"
    o.light = "llama3"
    o.embed = "bge-m3"
    return cfg


def test_host_type_defaults_to_ollama():
    cfg = _cfg_rich()
    assert cfg.ollama.host_type("chat") == "ollama"
    assert cfg.ollama.host_type("embeds") == "openai"
    assert cfg.ollama.host_type(None) == "ollama"
    assert cfg.ollama.host_type("does-not-exist") == "ollama"


def test_role_for_model_maps_to_roles():
    cfg = _cfg_rich()
    assert cfg.ollama.role_for_model("llama3") == "reasoning"  # == reasoning name
    assert cfg.ollama.role_for_model("bge-m3") == "embed"
    assert cfg.ollama.role_for_model(None) == "reasoning"


def test_host_for_role_falls_back_to_active_host():
    cfg = _cfg_rich()
    # No host_by_role → every role uses the active/default host.
    assert cfg.ollama.host_for_role("reasoning") == "chat"
    assert cfg.ollama.host_for_role("embed") == "chat"
    # Route embed elsewhere; reasoning unchanged.
    cfg.ollama.host_by_role = {"embed": "embeds"}
    assert cfg.ollama.host_for_role("embed") == "embeds"
    assert cfg.ollama.host_for_role("reasoning") == "chat"
    # A route to an unknown host falls back to the default (never breaks).
    cfg.ollama.host_by_role = {"embed": "missing"}
    assert cfg.ollama.host_for_role("embed") == "chat"


def test_facade_routes_roles_to_different_providers():
    cfg = _cfg_rich()
    cfg.ollama.host_by_role = {"embed": "embeds"}
    models = Models(cfg)
    # embed role → the openai host → OpenAIModel
    assert type(models._provider(role="embed")) is OpenAIModel
    # reasoning role → the ollama host → OllamaModel
    assert type(models._provider(role="reasoning")) is OllamaModel
    # model-name routing: the embed model name resolves to the openai host.
    assert type(models._provider(model="bge-m3")) is OpenAIModel


def test_openai_model_base_url_and_key():
    cfg = _cfg_rich()
    cfg.ollama.api_keys = {"embeds": "sk-test"}
    pm = OpenAIModel(cfg, "embeds")
    assert pm.base_url == "http://embeds:8080"
    assert pm.api_key == "sk-test"
    assert pm._headers()["Authorization"] == "Bearer sk-test"
    # A key supplied via env also works.
    cfg2 = _cfg_rich()
    import os

    os.environ["EVIDENCE_APIKEY_EMBEDS"] = "env-key"
    try:
        assert cfg2.ollama.api_key_for("embeds") == "env-key"
    finally:
        del os.environ["EVIDENCE_APIKEY_EMBEDS"]


def test_hosts_add_openai_type_persists_type(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"data_dir": str(tmp_path / ".evidence")}),
                        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["hosts-add", "llama", "http://10.0.0.9:8080", "--type", "openai",
         "--config", str(cfg_file)],
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["hosts"]["llama"] == "http://10.0.0.9:8080"
    assert data["ollama"]["host_types"]["llama"] == "openai"


def test_hosts_add_rejects_bad_type(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    import yaml

    cfg_file.write_text(yaml.safe_dump({"ollama": {"hosts": {"a": "http://a:1"}}}),
                        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["hosts-add", "x", "http://x:1", "--type", "bogus", "--config", str(cfg_file)],
    )
    assert result.exit_code == 1


def test_hosts_use_sets_default_for_all_roles(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {"ollama": {"hosts": {"a": "http://a:1", "b": "http://b:2"},
                         "active_host": "a"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts-use", "b", "--config", str(cfg_file)])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["active_host"] == "b"
    cfg = EvidenceConfig.load(str(cfg_file))
    # No role routed → all roles now resolve to the new default host.
    assert cfg.ollama.host_for_role("reasoning") == "b"
    assert cfg.ollama.host_for_role("embed") == "b"


def test_hosts_use_rejects_undefined_host(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"ollama": {"hosts": {"a": "http://a:1"}}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts-use", "nope", "--config", str(cfg_file)])
    assert result.exit_code == 1


def test_hosts_route_persists_role_routing(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {"ollama": {"hosts": {"chat": "http://chat:1", "embeds": "http://e:1"},
                         "active_host": "chat"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["hosts-route", "embed", "embeds", "--config", str(cfg_file)]
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["ollama"]["host_by_role"]["embed"] == "embeds"
    cfg = EvidenceConfig.load(str(cfg_file))
    assert cfg.ollama.host_for_role("embed") == "embeds"
    assert cfg.ollama.host_for_role("reasoning") == "chat"  # unchanged default


def test_hosts_route_rejects_unknown_role(tmp_path, monkeypatch):
    import yaml

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump({"ollama": {"hosts": {"a": "http://a:1"}}}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hosts-route", "bogus", "a", "--config", str(cfg_file)])
    assert result.exit_code == 1
