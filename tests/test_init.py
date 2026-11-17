"""Tests for `evidence init` (one-command project setup) + config auto-discovery."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from evidence.cli import app
from evidence.config import EvidenceConfig, _find_project_config

runner = CliRunner()


def test_init_scaffolds_project(tmp_path, monkeypatch):
    """`evidence init` creates config, corpus dir, .mcp.json, and updates .gitignore."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "note.md").write_text("hello")

    result = runner.invoke(app, ["init", "--yes", "--skip-ingest"])
    assert result.exit_code == 0, result.output

    cfg = tmp_path / ".evidence" / "config.yaml"
    assert cfg.is_file()
    text = cfg.read_text(encoding="utf-8")
    assert "data_dir: ./.evidence" in text
    assert "corpus_dir: ./corpus" in text

    mcp_json = tmp_path / ".mcp.json"
    assert mcp_json.is_file()
    payload = json.loads(mcp_json.read_text(encoding="utf-8"))
    # Server name is unique-per-project (evidence-<dir>) so a shared MCP
    # gateway can't merge two projects into one process.
    names = list(payload["servers"])
    assert len(names) == 1 and names[0].startswith("evidence-")
    server = payload["servers"][names[0]]
    assert server["command"].endswith("evidence-mcp")
    assert server["args"][:1] == ["--config"]

    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    # We ignore the specific state files (not the whole `.evidence/` dir) so the
    # generated `ledger.md` review artifact stays committable.
    assert ".evidence/ledger.sqlite" in gitignore
    assert ".evidence/index/" in gitignore
    assert "corpus/" in gitignore
    # The whole-directory rule must NOT be present (it would hide ledger.md).
    assert "\n.evidence/\n" not in gitignore


def test_init_external_defaults_to_true(tmp_path, monkeypatch):
    """`evidence init --yes` enables external literature search by default."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--yes", "--skip-ingest"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    assert "enabled: true" in text.split("external:")[1].split("graph_expansion:")[0]


def test_init_auto_pull_defaults_to_true(tmp_path, monkeypatch):
    """`evidence init --yes` enables Ollama auto-pull by default."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--yes", "--skip-ingest"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    ollama_block = text.split("ollama:")[1].split("retrieval:")[0]
    assert "auto_pull: true" in ollama_block


def test_init_can_disable_auto_pull(tmp_path, monkeypatch):
    """`--no-auto-pull` writes `auto_pull: false` into the ollama section."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["init", "--yes", "--skip-ingest", "--no-auto-pull"]
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    ollama_block = text.split("ollama:")[1].split("retrieval:")[0]
    assert "auto_pull: false" in ollama_block


def test_init_can_disable_external(tmp_path, monkeypatch):
    """`--no-external` writes `enabled: false` for the external section."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "--yes", "--skip-ingest", "--no-external"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    ext_block = text.split("external:")[1].split("graph_expansion:")[0]
    assert "enabled: false" in ext_block


def test_init_ollama_happy_path_bakes_models(tmp_path, monkeypatch):
    """Interactive Ollama success: reasoning + embedding models are baked in."""
    import evidence.cli as cli

    monkeypatch.setattr(
        cli, "_ollama_list_models", lambda host, timeout=5.0: ["alpha", "beta"]
    )
    monkeypatch.chdir(tmp_path)
    # inputs: confirm=y, host=<enter>, reasoning# 2, embedding# 1
    result = runner.invoke(cli.app, ["init", "--skip-ingest"], input="y\n\n2\n1\n")
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    # reasoning + light use the chosen reasoning model (beta); embed uses alpha.
    assert "reasoning: beta" in text
    assert "light: beta" in text
    assert "embed: alpha" in text
    assert "host: http://localhost:11434" in text


def test_init_ollama_embed_defaults_to_bge_m3(tmp_path, monkeypatch):
    """When bge-m3 is available, it is the default embedding choice."""
    import evidence.cli as cli

    monkeypatch.setattr(
        cli,
        "_ollama_list_models",
        lambda host, timeout=5.0: ["qwen2.5:7b", "bge-m3"],
    )
    monkeypatch.chdir(tmp_path)
    # sorted models: 1=bge-m3, 2=qwen2.5:7b
    # inputs: confirm=y, host=<enter>, reasoning# 2, embedding=<enter>=default(bge-m3)
    result = runner.invoke(cli.app, ["init", "--skip-ingest"], input="y\n\n2\n\n")
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    assert "reasoning: qwen2.5:7b" in text
    assert "embed: bge-m3" in text


def test_init_ollama_skip_keeps_defaults(tmp_path, monkeypatch):
    """User declines Ollama → template defaults are untouched."""
    import evidence.cli as cli

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init", "--skip-ingest"], input="n\n")
    assert result.exit_code == 0, result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    assert "embed: bge-m3" in text
    assert "reasoning: qwen2.5:9b" in text


def test_init_ollama_unreachable_skips(tmp_path, monkeypatch):
    """Ollama unreachable → offered reconfigure, user declines → defaults kept."""
    import evidence.cli as cli

    monkeypatch.setattr(cli, "_ollama_list_models", lambda host, timeout=5.0: None)
    monkeypatch.chdir(tmp_path)
    # confirm=y, host=<enter default (unreachable)>, offer-different-host? n
    result = runner.invoke(cli.app, ["init", "--skip-ingest"], input="y\n\nn\n")
    assert result.exit_code == 0, result.output
    assert "Could not reach Ollama" in result.output
    text = (tmp_path / ".evidence" / "config.yaml").read_text(encoding="utf-8")
    assert "embed: bge-m3" in text


def test_init_refuses_overwrite_without_yes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".evidence").mkdir()
    (tmp_path / ".evidence" / "config.yaml").write_text("data_dir: ~/.custom\n")
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert "already exists" in result.output
    # Existing config is preserved.
    assert (tmp_path / ".evidence" / "config.yaml").read_text() == "data_dir: ~/.custom\n"


def test_auto_discovery_finds_config(tmp_path, monkeypatch):
    (tmp_path / ".evidence").mkdir()
    (tmp_path / ".evidence" / "config.yaml").write_text("data_dir: ./found\n")
    monkeypatch.chdir(tmp_path)
    assert _find_project_config() == tmp_path / ".evidence" / "config.yaml"

    cfg = EvidenceConfig.load()  # no explicit path → auto-discovery
    # `./found` in the YAML is rebased against the project root (the config
    # lives at `<root>/.evidence/config.yaml`, so the root is one level up).
    assert Path(cfg.data_dir) == tmp_path / "found"


def test_auto_discovery_walks_up(tmp_path, monkeypatch):
    (tmp_path / ".evidence").mkdir()
    (tmp_path / ".evidence" / "config.yaml").write_text("data_dir: ./root\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert _find_project_config() == tmp_path / ".evidence" / "config.yaml"


def test_no_config_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _find_project_config() is None
