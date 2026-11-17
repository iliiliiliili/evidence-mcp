"""Tests for the `evidence ledger impact` CLI subcommand."""

from __future__ import annotations

from typer.testing import CliRunner

from evidence.cli import app
from evidence.ledger.store import Ledger

runner = CliRunner()


def test_ledger_impact_by_shared_evidence(tmp_path, monkeypatch):
    """Two records share a chunk ⇒ impact on one surfaces the other."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    ev = [{"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"}]
    lg.record("mod.a", "decision A", evidence=ev)
    lg.record("mod.b", "decision B", evidence=ev)
    lg.close()

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ledger", "impact", "mod.a"])
    assert result.exit_code == 0, result.output
    assert "mod.b" in result.output
    assert "p1::c1" in result.output


def test_ledger_impact_by_dependency(tmp_path, monkeypatch):
    """A record depending on a root ⇒ impact on the root lists the dependent."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.root", "root decision")
    lg.record("mod.child", "child decision", depends_on=["mod.root"])
    lg.close()

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ledger", "impact", "mod.root"])
    assert result.exit_code == 0, result.output
    assert "mod.child" in result.output


def test_ledger_impact_unknown_symbol(tmp_path, monkeypatch):
    """A bare chunk_id with no citing records is a clean no-op."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.close()

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ledger", "impact", "no::chunk"])
    assert result.exit_code == 0, result.output
    assert "No other decisions" in result.output


def test_ledger_symbols_lists_recorded(tmp_path, monkeypatch):
    """`ledger symbols` lists symbols that have a decision record."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.with_ev", "decision A",
              evidence=[{"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"}])
    lg.record("mod.without_ev", "decision B")
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "symbols"])
    assert r.exit_code == 0, r.output
    assert "mod.with_ev" in r.output
    assert "mod.without_ev" in r.output

    # --ungrounded keeps only the record with no evidence.
    r = runner.invoke(app, ["ledger", "symbols", "--ungrounded"])
    assert r.exit_code == 0, r.output
    assert "mod.with_ev" not in r.output
    assert "mod.without_ev" in r.output


def test_ledger_auto_update_refreshes_and_writes_md(tmp_path, monkeypatch):
    """`ledger auto-update` refreshes fingerprints and rewrites ledger.md."""
    from evidence.ledger.fingerprint import symbol_fingerprint

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def f(a):\n    return a + 1\n")

    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    # Record `f` with a WRONG fingerprint → would be stale_code.
    bad_fp = symbol_fingerprint("def f(a):\n    return 999\n", "mod.f")
    assert bad_fp is not None
    lg.record("mod.f", "decision", code_fingerprint=bad_fp)
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "auto-update", str(repo)])
    assert r.exit_code == 0, r.output

    # After refresh, the record is re-stamped to fresh.
    lg2 = Ledger(ledger_path)
    assert lg2.lookup("mod.f")["status"] == "fresh"
    lg2.close()

    # ledger.md was written next to the ledger.
    md = tmp_path / "ledger.md"
    assert md.is_file()
    assert "mod.f" in md.read_text(encoding="utf-8")


def test_ledger_hook_install_and_uninstall(tmp_path, monkeypatch):
    """`ledger hook install` writes an executable post-checkout hook."""
    import stat

    # init a git repo
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)

    r = runner.invoke(app, ["ledger", "hook", "install"])
    assert r.exit_code == 0, r.output

    hook = tmp_path / ".git" / "hooks" / "post-checkout"
    assert hook.is_file()
    text = hook.read_text(encoding="utf-8")
    assert "evidence-auto-update" in text
    assert "ledger auto-update" in text
    assert hook.stat().st_mode & stat.S_IXUSR  # executable

    # idempotent — running again keeps a single block.
    runner.invoke(app, ["ledger", "hook", "install"])
    assert hook.read_text(encoding="utf-8").count("evidence-auto-update") == 1

    # uninstall removes the evidence block.
    r = runner.invoke(app, ["ledger", "hook", "uninstall"])
    assert r.exit_code == 0, r.output
    assert "evidence-auto-update" not in hook.read_text(encoding="utf-8")


def test_ledger_hook_requires_git(tmp_path, monkeypatch):
    """`ledger hook install` fails cleanly outside a git repo."""
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "hook", "install"])
    assert r.exit_code == 1
    assert "not a git repo" in r.output


def test_ledger_symbols_json(tmp_path, monkeypatch):
    import json

    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.x", "decision A",
              evidence=[{"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"}])
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "symbols", "--json"])
    assert r.exit_code == 0, r.output
    rows = [json.loads(line) for line in r.output.splitlines() if line.strip()]
    assert rows[0]["symbol"] == "mod.x"
    assert rows[0]["evidence"] == 1


def test_ledger_export_default_is_terminal_view(tmp_path, monkeypatch):
    """`ledger export` (no flags) prints a color view; writes no file."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.a", "decision A",
              evidence=[{"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"}])
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "export"])
    assert r.exit_code == 0, r.output
    # The record is shown, and its evidence is human-readable (not the raw
    # machine key, which belongs only to JSON / the DB).
    assert "mod.a" in r.output
    assert "decision A" in r.output
    assert "evidence" in r.output
    assert "p1::c1" not in r.output
    # No file is written by the default view.
    assert not (tmp_path / "ledger.md").exists()


def test_ledger_export_format_markdown_writes_file(tmp_path, monkeypatch):
    """`ledger export -f markdown` writes `<ledger-dir>/ledger.md`."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.a", "decision A")
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "export", "-f", "markdown"])
    assert r.exit_code == 0, r.output

    out_file = tmp_path / "ledger.md"
    assert out_file.is_file(), f"expected {out_file} to exist; output={r.output!r}"
    text = out_file.read_text(encoding="utf-8")
    assert "mod.a" in text
    assert "# Evidence Ledger" in text


def test_ledger_export_stdout_flag(tmp_path, monkeypatch):
    """`--stdout` prints the raw markdown to stdout and writes no file."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.b", "decision B")
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "export", "--stdout"])
    assert r.exit_code == 0, r.output
    # Raw markdown (the file-style header), not the color view.
    assert "# Evidence Ledger" in r.output
    assert "mod.b" in r.output
    # And no file was created.
    assert not (tmp_path / "ledger.md").exists()


def test_ledger_export_json_default(tmp_path, monkeypatch):
    """JSON default writes `<ledger-dir>/ledger.json`."""
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger_path))
    lg = Ledger(ledger_path)
    lg.record("mod.c", "decision C")
    lg.close()

    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["ledger", "export", "--format", "json"])
    assert r.exit_code == 0, r.output

    out_file = tmp_path / "ledger.json"
    assert out_file.is_file(), f"expected {out_file} to exist; output={r.output!r}"
    import json

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    assert payload["records"][0]["symbol"] == "mod.c"
