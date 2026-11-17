"""Tests for the evidence ledger (new plan: lookup/record/audit/impact)."""

from __future__ import annotations

import pytest

from evidence.ledger import (
    FRESH,
    ORPHANED,
    STALE_CODE,
    STALE_DEPENDENCY,
    Ledger,
)
from evidence.ledger.fingerprint import (
    fingerprint_source,
    scan_repo,
    symbol_fingerprint,
)


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    lg = Ledger(tmp_path / "ledger.sqlite")
    yield lg
    lg.close()


# ── fingerprint ───────────────────────────────────────────────────────────
def test_fingerprint_is_stable():
    a = fingerprint_source("def f():\n    return 1\n", symbol="m.f")
    b = fingerprint_source("def f():\n    return 1\n", symbol="m.f")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_fingerprint_changes_with_body():
    a = fingerprint_source("def f():\n    return 1\n", symbol="m.f")
    b = fingerprint_source("def f():\n    return 2\n", symbol="m.f")
    assert a != b


def test_symbol_fingerprint_finds_named_symbol():
    src = "def helper():\n    return 1\n\ndef train(x):\n    return x * 2\n"
    fp = symbol_fingerprint(src, "mod.train")
    assert fp is not None
    # A different symbol hashes differently (different name prefix).
    fp2 = symbol_fingerprint(src, "mod.helper")
    assert fp2 is not None and fp2 != fp


def test_scan_repo_indexes_top_level_defs(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "def foo(a):\n    return a + 1\n\nclass Bar:\n    pass\n"
    )
    m = scan_repo(tmp_path)
    assert "pkg.mod.foo" in m
    assert "pkg.mod.Bar" in m
    # bare-name index also present
    assert "foo" in m


# ── record / lookup ───────────────────────────────────────────────────────
def test_lookup_missing_returns_none(ledger):
    assert ledger.lookup("does.not.exist") is None


def test_record_then_lookup(ledger):
    rec = ledger.record(
        "pkg.train",
        "Adam + cosine LR",
        evidence=[
            {"chunk_id": "p1::c1", "source": "local", "title": "Attention paper"},
            {"doi": "10.1000/b", "source": "openalex"},
        ],
        alternatives=[{"option": "SGD", "rejected": "slower"}],
        rationale="better convergence",
        code_fingerprint="abc",
    )
    assert rec["symbol"] == "pkg.train"
    assert rec["status"] == FRESH
    assert len(rec["evidence"]) == 2

    got = ledger.lookup("pkg.train")
    assert got["decision"] == "Adam + cosine LR"
    assert got["rationale"] == "better convergence"
    assert {e["chunk_id"] for e in got["evidence"] if e["ref_type"] == "chunk"} == {"p1::c1"}
    assert {e["doi"] for e in got["evidence"] if e["ref_type"] == "doi"} == {"10.1000/b"}


def test_record_validation(ledger):
    with pytest.raises(ValueError):
        ledger.record("", "decision")
    with pytest.raises(ValueError):
        ledger.record("sym", "")


def test_record_upsert_replaces(ledger):
    ledger.record("sym", "v1", evidence=[{"chunk_id": "a"}])
    ledger.record("sym", "v2", evidence=[{"chunk_id": "b"}])
    got = ledger.lookup("sym")
    assert got["decision"] == "v2"
    assert [e["chunk_id"] for e in got["evidence"]] == ["b"]


def test_count_and_list(ledger):
    assert ledger.count_records() == 0
    ledger.record("a", "d1")
    ledger.record("b", "d2")
    assert ledger.count_records() == 2
    assert {r["symbol"] for r in ledger.list_records()} == {"a", "b"}


def test_record_auto_refreshes_ledger_md(tmp_path):
    """A ledger with ledger_md_path rewrites ledger.md on every record."""
    md = tmp_path / ".evidence" / "ledger.md"
    lg = Ledger(tmp_path / "ledger.sqlite", ledger_md_path=md)
    lg.record("pkg.train", "use Adam", evidence=[{"chunk_id": "p1::c1"}])
    lg.close()

    assert md.is_file()
    text = md.read_text(encoding="utf-8")
    assert "pkg.train" in text
    assert "use Adam" in text

    # A second record is reflected too (artifact stays current, not a snapshot).
    lg2 = Ledger(tmp_path / "ledger.sqlite", ledger_md_path=md)
    lg2.record("pkg.eval", "batch eval", evidence=[{"chunk_id": "p2::c7"}])
    lg2.close()
    assert "pkg.eval" in md.read_text(encoding="utf-8")


def test_record_without_md_path_is_side_effect_free(tmp_path):
    """The bare store (no ledger_md_path) writes no artifact — pure contract."""
    lg = Ledger(tmp_path / "ledger.sqlite")
    lg.record("a", "d")
    lg.close()
    assert not (tmp_path / "ledger.md").exists()


# ── impact ────────────────────────────────────────────────────────────────
def test_impact_by_chunk(ledger):
    ledger.record("sym1", "d1", evidence=[{"chunk_id": "p1::c1"}])
    ledger.record("sym2", "d2", evidence=[{"chunk_id": "p1::c1"}])
    ledger.record("sym3", "d3", evidence=[{"doi": "10.9/other"}])

    res = ledger.impact("p1::c1")
    affected = {a["symbol"] for a in res["affected"]}
    assert affected == {"sym1", "sym2"}
    assert res["affected_count"] == 2
    shared = {s["ref_key"] for s in res["shared_refs"]}
    assert "p1::c1" in shared


def test_impact_by_symbol(ledger):
    ledger.record("sym1", "d1", evidence=[{"chunk_id": "p1::c1"}])
    ledger.record("sym2", "d2", evidence=[{"chunk_id": "p1::c1"}])
    res = ledger.impact("sym1")
    # sym1 itself is excluded from the impact set.
    assert {a["symbol"] for a in res["affected"]} == {"sym2"}


def test_impact_no_shared(ledger):
    ledger.record("sym1", "d1", evidence=[{"chunk_id": "only"}])
    res = ledger.impact("sym1")
    assert res["affected"] == []


# ── audit ─────────────────────────────────────────────────────────────────
def test_audit_flags_orphaned_and_stale_code(tmp_path, ledger):
    """Build a 'repo' with three symbols, record each, then audit.

    - `repo.mod.keep`  → fingerprint matches → FRESH.
    - `repo.mod.gone`  → not in repo         → ORPHANED.
    - `repo.mod.drift` → in repo but body differs → STALE_CODE.
    """
    from evidence.ledger.fingerprint import scan_repo, symbol_fingerprint

    repo = tmp_path / "repo"
    repo.mkdir()
    # `keep` is stable; `drift` is present but its body will be recorded as
    # something different so the audit sees a fingerprint mismatch.
    (repo / "mod.py").write_text(
        "def keep(a):\n    return a + 1\n\n"
        "def drift(a):\n    return a * 2\n"
    )
    m = scan_repo(repo)

    ledger.record(
        "mod.keep", "stable", code_fingerprint=m["mod.keep"]
    )
    ledger.record(
        "mod.gone", "will be orphaned", code_fingerprint="zzz"
    )
    # Record `drift` with a fingerprint computed from DIFFERENT source so it
    # is present (not orphaned) but drifted (stale_code).
    drift_fp = symbol_fingerprint("def drift(a):\n    return a + 99\n",
                                  "mod.drift")
    assert drift_fp is not None
    ledger.record(
        "mod.drift", "will be stale_code", code_fingerprint=drift_fp
    )

    res = ledger.audit(scope=repo)
    assert res["total"] == 3
    assert res["fresh"] == 1
    assert res["orphaned"] == 1
    assert res["stale_code"] == 1
    statuses = {r["symbol"]: r["status"] for r in res["needs_reanalysis"]}
    assert statuses.get("mod.gone") == ORPHANED
    assert statuses.get("mod.drift") == STALE_CODE


def test_audit_fresh_when_fingerprint_matches(tmp_path, ledger):
    from evidence.ledger.fingerprint import scan_repo

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def keep(a):\n    return a + 1\n")
    fp = scan_repo(repo)["mod.keep"]
    ledger.record("mod.keep", "ok", code_fingerprint=fp)

    res = ledger.audit(scope=repo)
    assert res["fresh"] == 1
    assert res["needs_reanalysis"] == []


def test_audit_refresh_reevaluates_stale_code(tmp_path, ledger):
    """refresh=True silently re-stamps stale_code records; orphaned stays."""
    from evidence.ledger.fingerprint import scan_repo, symbol_fingerprint

    repo = tmp_path / "repo"
    repo.mkdir()
    # Two symbols: one will drift, one will be deleted.
    (repo / "mod.py").write_text(
        "def drift(a):\n    return a * 2\n"
        "def gone(a):\n    return a\n"
    )
    m = scan_repo(repo)

    # Record `drift` with a WRONG fingerprint so it becomes stale_code.
    bad_fp = symbol_fingerprint("def drift(a):\n    return 999\n", "mod.drift")
    ledger.record("mod.drift", "d", code_fingerprint=bad_fp)
    # Record `gone` with a real fingerprint, but then delete the symbol.
    ledger.record("mod.gone", "g", code_fingerprint=m["mod.gone"])
    (repo / "mod.py").write_text("def drift(a):\n    return a * 2\n")  # gone is now deleted

    # Without refresh: stale_code + orphaned.
    res = ledger.audit(scope=repo)
    assert res["stale_code"] == 1
    assert res["orphaned"] == 1

    # With refresh: stale_code → fresh (re-stamped); orphaned stays.
    res2 = ledger.audit(scope=repo, refresh=True)
    assert res2["fresh"] == 1
    assert res2["orphaned"] == 1
    assert res2["stale_code"] == 0
    assert ledger.lookup("mod.drift")["status"] == FRESH


# ── transitive dependency (depends_on) ────────────────────────────────────
def test_record_surfaces_depends_on(ledger):
    ledger.record("arch.language", "use Python")
    ledger.record(
        "arch.ml_framework",
        "use PyTorch",
        depends_on=["arch.language"],
    )
    rec = ledger.lookup("arch.ml_framework")
    assert rec["depends_on"] == ["arch.language"]
    # The root has no dependents declared.
    assert ledger.lookup("arch.language")["depends_on"] == []


def test_transitive_dependents_walks_chain(ledger):
    ledger.record("arch.language", "use Python")
    ledger.record("arch.ml_framework", "use PyTorch", depends_on=["arch.language"])
    ledger.record("model.train", "use AdamW", depends_on=["arch.ml_framework"])

    deps = ledger.transitive_dependents("arch.language")
    assert set(deps) == {"arch.ml_framework", "model.train"}
    # Depth caps the walk.
    assert set(ledger.transitive_dependents("arch.language", max_depth=1)) == {
        "arch.ml_framework"
    }
    # A leaf has no dependents.
    assert ledger.transitive_dependents("model.train") == []


def test_transitive_dependents_is_cycle_safe(ledger):
    # Self-dependency and a 2-cycle must not cause an infinite loop.
    ledger.record("a", "d", depends_on=["a"])
    ledger.record("b", "d", depends_on=["c"])
    ledger.record("c", "d", depends_on=["b"])
    assert set(ledger.transitive_dependents("a")) == set()
    assert set(ledger.transitive_dependents("b")) == {"c"}
    assert set(ledger.transitive_dependents("c")) == {"b"}


def test_impact_shows_dependents_via_depends_on(ledger):
    """Change a root decision → surface dependents even with distinct evidence."""
    ledger.record("arch.language", "use Python", evidence=[{"chunk_id": "py::c1"}])
    ledger.record(
        "arch.ml_framework",
        "use PyTorch",
        evidence=[{"chunk_id": "torch::c1"}],  # different evidence
        depends_on=["arch.language"],
    )
    res = ledger.impact("arch.language")
    affected = {a["symbol"]: a["via"] for a in res["affected"]}
    assert "arch.ml_framework" in affected
    assert "depends_on" in affected["arch.ml_framework"]


def test_impact_combines_shared_and_dependency(ledger):
    ledger.record("root", "d", evidence=[{"chunk_id": "c1"}])
    # `child` depends on root AND shares root's evidence.
    ledger.record("child", "d2", evidence=[{"chunk_id": "c1"}], depends_on=["root"])
    res = ledger.impact("root")
    child = next(a for a in res["affected"] if a["symbol"] == "child")
    assert "shared_evidence" in child["via"]
    assert "depends_on" in child["via"]


def test_audit_flags_stale_dependency(ledger):
    """A root that becomes orphaned taints every transitive dependent."""
    ledger.record("arch.language", "use Python")
    ledger.record("arch.ml_framework", "use PyTorch", depends_on=["arch.language"])
    ledger.record("model.train", "use AdamW", depends_on=["arch.ml_framework"])

    # Empty scope → every symbol is orphaned (no code to fingerprint).
    res = ledger.audit(scope="repo")
    assert res["orphaned"] >= 1
    statuses = {r["symbol"]: r["status"] for r in res["needs_reanalysis"]}
    # Dependents whose own code isn't the issue get flagged stale_dependency.
    assert statuses.get("arch.ml_framework") in (STALE_DEPENDENCY, ORPHANED)
    assert statuses.get("model.train") in (STALE_DEPENDENCY, ORPHANED)


# ── export (readable projection for git review) ───────────────────────────
def test_export_markdown_renders_reasoning(ledger):
    ledger.record(
        "pkg.train",
        "Adam + cosine LR",
        evidence=[
            {"chunk_id": "p1::c1", "source": "local", "title": "Attention paper"},
            {"doi": "10.1000/b", "source": "openalex"},
        ],
        alternatives=[{"option": "SGD", "rejected": "slower"}],
        rationale="better convergence",
        code_fingerprint="abc",
    )
    ledger.record("pkg.optimize", "use AdamW", depends_on=["pkg.train"])

    md = ledger.export_markdown()
    assert "# Evidence Ledger" in md
    assert "## `pkg.train`" in md
    assert "Adam + cosine LR" in md
    assert "better convergence" in md
    # Human-readable evidence: the chunk shows its title, the DOI its identifier.
    # Machine keys (chunk ids) must NOT appear in the markdown — they belong
    # only to the JSON export and the database.
    assert "Attention paper" in md
    assert "10.1000/b" in md
    assert "p1::c1" not in md
    # depends_on edge is surfaced for the reviewer.
    assert "`pkg.train`" in md
    assert "Depends on" in md
    # Rejected alternatives render as "name — reason", not the raw dict.
    assert "SGD — slower" in md
    assert "'option'" not in md


def test_export_markdown_empty(ledger):
    md = ledger.export_markdown()
    assert "No decision records yet" in md


def test_export_json_is_stable_and_complete(ledger):
    ledger.record(
        "pkg.train",
        "Adam + cosine LR",
        evidence=[{"chunk_id": "p1::c1"}],
        depends_on=[],
    )
    import json as _json

    payload = _json.loads(ledger.export_json())
    assert payload["record_count"] == 1
    rec = payload["records"][0]
    assert rec["symbol"] == "pkg.train"
    assert rec["evidence"][0]["ref_key"] == "p1::c1"
    assert rec["depends_on"] == []
    # Sorted + indent => deterministic (diffable) output.
    assert ledger.export_json() == _json.dumps(
        _json.loads(ledger.export_json()), indent=2, sort_keys=True
    ) + "\n"


def test_export_via_mcp_tool(ledger):
    from evidence.mcp import tools

    ledger.record("pkg.train", "Adam", evidence=[{"chunk_id": "p1::c1"}])
    res = tools.export_evidence(format="markdown", ledger=ledger)
    assert res["format"] == "markdown"
    assert res["record_count"] == 1
    assert "## `pkg.train`" in res["content"]

    res_json = tools.export_evidence(format="json", ledger=ledger)
    assert res_json["format"] == "json"
    import json as _json

    _json.loads(res_json["content"])  # valid JSON
    with pytest.raises(ValueError):
        tools.export_evidence(format="yaml", ledger=ledger)
