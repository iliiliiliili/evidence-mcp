"""Tests for autonomous `evidence populate` (scan, draft, ground, record).

Grounding is stubbed to a no-op so tests stay hermetic (no index, no network);
the Ollama client is faked so the reasoning-model call returns a canned decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence.config import EvidenceConfig
from evidence.ledger.store import Ledger
from evidence.populate import PopulateService, scan_symbols

_SRC = '''
"""Demo module."""
HIDDEN_LIMIT = 12


def compute_total(items):
    """Sum the given items."""
    return sum(items)


class Cart:
    """A small cart."""
    def __init__(self):
        self.items = []
'''


class _FakeOllamaClient:
    def __init__(self, **_):
        pass

    def chat(self, **kwargs) -> dict:
        decision = {
            "decision": "Use a plain built-in sum() over a generator for the total.",
            "rationale": "Stdlib-only, O(n), no extra dependency for a simple total.",
            "alternatives": [
                {"name": "numpy.sum", "why_rejected": "heavy dependency for scalar math"},
            ],
            "depends_on": ["sum"],
        }
        return {"message": {"content": json.dumps(decision)}}


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> _FakeOllamaClient:
    fake = _FakeOllamaClient()
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: fake))
    return fake


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(_SRC, encoding="utf-8")
    return tmp_path


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.sqlite")


@pytest.fixture
def cfg(tmp_path: Path) -> EvidenceConfig:
    c = EvidenceConfig()
    c.external.enabled = False  # disable network grounding in tests
    return c


def test_scan_symbols_finds_functions_classes_and_consts(src_dir: Path):
    syms = scan_symbols(src_dir, max_symbols=20)
    names = {s.name for s in syms}
    assert "mod.compute_total" in names
    assert "mod.Cart" in names
    assert "mod.HIDDEN_LIMIT" in names
    by_kind = {s.name: s.kind for s in syms}
    assert by_kind["mod.compute_total"] == "function"
    assert by_kind["mod.Cart"] == "class"
    assert by_kind["mod.HIDDEN_LIMIT"] == "const"
    # Fingerprint is non-empty and stable.
    fp1 = {s.name: s.fingerprint for s in scan_symbols(src_dir)}
    fp2 = {s.name: s.fingerprint for s in scan_symbols(src_dir)}
    assert fp1["mod.compute_total"] == fp2["mod.compute_total"]
    assert fp1["mod.compute_total"]


def test_scan_symbols_limits_count(src_dir: Path):
    assert len(scan_symbols(src_dir, max_symbols=1)) == 1


def _svc(cfg, ledger, tmp_path: Path) -> PopulateService:
    svc = PopulateService(cfg, ledger=ledger)
    svc._app_or_none = lambda: None  # grounding stub: no local corpus
    return svc


def test_populate_records_into_ledger(
    cfg, ledger, src_dir, tmp_path, fake_ollama
):
    svc = _svc(cfg, ledger, tmp_path)
    report = svc.run(src_dir, max_symbols=10)
    assert report.scanned == 3
    assert report.dry_run is False
    rec = ledger.lookup("mod.compute_total")
    assert rec is not None
    assert "sum()" in rec["decision"]
    assert rec["alternatives"]
    # No corpus + external disabled ⇒ un-grounded ⇒ a no_evidence conflict.
    kinds = {c.kind for c in report.conflicts}
    assert "no_evidence" in kinds


def test_populate_dry_run_writes_nothing(cfg, ledger, src_dir, tmp_path, fake_ollama):
    svc = _svc(cfg, ledger, tmp_path)
    svc.run(src_dir, max_symbols=10, dry_run=True)
    assert ledger.lookup("mod.compute_total") is None
    assert ledger.count_records() == 0


def test_populate_skips_existing_without_force(cfg, ledger, src_dir, fake_ollama):
    ledger.record("mod.compute_total", "pre-existing decision")
    svc = _svc(cfg, ledger, None)
    report = svc.run(src_dir, max_symbols=10)
    assert "mod.compute_total" in report.skipped
    assert ledger.lookup("mod.compute_total")["decision"] == "pre-existing decision"


def test_populate_force_overwrites(cfg, ledger, src_dir, fake_ollama):
    ledger.record("mod.compute_total", "old decision")
    svc = PopulateService(cfg, ledger=ledger)
    svc._app_or_none = lambda: None
    svc.run(src_dir, max_symbols=10, force=True)
    assert "sum()" in ledger.lookup("mod.compute_total")["decision"]


def test_report_as_text_includes_summary(cfg, ledger, src_dir, fake_ollama):
    svc = PopulateService(cfg, ledger=ledger)
    svc._app_or_none = lambda: None
    report = svc.run(src_dir, max_symbols=10)
    text = report.as_text()
    assert "evidence populate" in text
    assert "recorded:" in text
    assert "conflicts:" in text


def test_scan_symbols_zero_limit_is_unlimited(src_dir: Path):
    """max_symbols=0 (or negative) means 'no cap' — used by the `symbols` CLI."""
    assert len(scan_symbols(src_dir, max_symbols=0)) == 3
    assert len(scan_symbols(src_dir, max_symbols=-1)) == 3


def test_symbols_cli_lists_names(tmp_path, monkeypatch, src_dir):
    from typer.testing import CliRunner

    from evidence.cli import app

    result = CliRunner().invoke(app, ["symbols", str(src_dir)])
    assert result.exit_code == 0, result.output
    assert "mod.compute_total" in result.output
    assert "mod.Cart" in result.output
    assert "3 symbol(s)" in result.output


def test_symbols_cli_filters(tmp_path, src_dir):
    import json as _json

    from typer.testing import CliRunner

    from evidence.cli import app

    # kind filter
    r = CliRunner().invoke(app, ["symbols", str(src_dir), "--kind", "class", "--json"])
    assert r.exit_code == 0, r.output
    rows = [_json.loads(line) for line in r.output.splitlines() if line.strip()]
    assert [x["kind"] for x in rows] == ["class"]

    # prefix filter on the bare name
    r = CliRunner().invoke(app, ["symbols", str(src_dir), "--prefix", "compute"])
    assert r.exit_code == 0
    assert "mod.compute_total" in r.output
    assert "mod.Cart" not in r.output


# ── cited / external-merge grounding ─────────────────────────────────────────


class _FakeOllamaWithCited:
    """Like _FakeOllamaClient but the decision carries a `cited` list."""

    def __init__(self, **_):
        pass

    def chat(self, **kwargs) -> dict:
        decision = {
            "decision": "Use MCD (Bayesian dropout) as the baseline.",
            "rationale": "Paper-standard comparison for epistemic uncertainty.",
            "alternatives": [{"name": "BBB", "why_rejected": "heavier memory"}],
            "depends_on": [],
            "cited": [
                {
                    "name": "Dropout as a Bayesian Approximation",
                    "authors": "Gal & Ghahramani",
                    "doi": "10.48550/arXiv.1505.05416",
                }
            ],
        }
        return {"message": {"content": json.dumps(decision)}}


def test_record_persists_cited_and_lookup_returns_it(tmp_path):
    from evidence.ledger.store import Ledger

    lg = Ledger(tmp_path / "ledger.sqlite")
    lg.record(
        "mod.compute_total",
        "decision",
        evidence=[{"ref_type": "doi", "ref_key": "10.1/x", "source": "openalex"}],
        cited=[{"name": "Some Paper", "authors": "A & B", "doi": "10.1/x"}],
    )
    rec = lg.lookup("mod.compute_total")
    assert rec is not None
    assert "cited" in rec
    assert rec["cited"][0]["name"] == "Some Paper"
    assert rec["cited"][0]["doi"] == "10.1/x"
    lg.close()


def test_ground_with_cited_doi_attaches_doi_directly(cfg, ledger, src_dir, tmp_path):
    """When a `cited` entry already carries a DOI, it is attached verbatim
    (no network needed)."""
    svc = PopulateService(cfg, ledger=ledger)
    svc._app_or_none = lambda: None  # no local corpus
    # Simulate the draft returning a `cited` entry with a known DOI.
    cited = [{"name": "Dropout as a Bayesian Approximation",
              "authors": "Gal & Ghahramani",
              "doi": "10.48550/arXiv.1505.05416"}]
    evidence, notes = svc._ground("mod.MCDDropout", "Use MCD",
                                  external=False, cited=cited)
    keys = {e["ref_key"] for e in evidence}
    assert "10.48550/arXiv.1505.05416" in keys
    # The DOI ref should be typed as "doi".
    doi_refs = [e for e in evidence if e["ref_key"] == "10.48550/arXiv.1505.05416"]
    assert doi_refs[0]["ref_type"] == "doi"


def test_ground_saves_online_refs_when_app_available(
    cfg, ledger, src_dir, tmp_path
):
    """When the local corpus is available, `_ground` saves every external DOI
    it attached into the corpus (flag-aware, best-effort)."""
    from evidence.external import online_corpus as oc
    from evidence.indexes.store import Store

    cfg.external.save_online_to_corpus = False  # metadata-only (no download)
    svc = PopulateService(cfg, ledger=ledger)
    cited = [{"name": "Dropout", "authors": "Gal & Ghahramani",
              "doi": "10.48550/arXiv.1505.05416"}]

    # Build a fake app context that exposes a real Store on a temp path.
    store = Store(tmp_path / "corpus.sqlite")
    try:
        class _FakeApp:
            pass
        fake = _FakeApp()
        fake.store = store
        svc._app = fake  # skip _app_or_none's lazy construction

        calls: list[dict] = []
        real_save = oc.save_online_ref
        def spy(work, **kwargs):
            calls.append(dict(work))
            return real_save(work, **kwargs)
        import evidence.external.online_corpus as ocmod
        ocmod.save_online_ref = spy  # type: ignore[assignment]
        try:
            svc._ground("mod.MCDDropout", "Use MCD",
                        external=False, cited=cited)
        finally:
            ocmod.save_online_ref = real_save  # type: ignore[assignment]

        # The DOI was attached to evidence AND saved into the local corpus.
        # The paper_id is keyed on the *normalized* (lowercase) DOI, while the
        # PaperRecord.doi keeps the original casing (DOIs are case-sensitive
        # for display).
        assert any(c.get("doi") == "10.48550/arXiv.1505.05416" for c in calls)
        paper = store.get_paper("online__doi__10.48550/arxiv.1505.05416")
        assert paper is not None
        assert paper.doi == "10.48550/arXiv.1505.05416"
    finally:
        store.close()


def test_ground_with_cited_no_doi_external_off_falls_back_to_citation(
    cfg, ledger, src_dir, tmp_path
):
    """When a `cited` entry has no DOI and external is off, it is still
    attached as a `citation` ref (so the reviewer sees what the model wanted)."""
    svc = PopulateService(cfg, ledger=ledger)
    svc._app_or_none = lambda: None
    cited = [{"name": "Weight Uncertainty in Neural Networks",
              "authors": "Blundell et al."}]
    evidence, notes = svc._ground("mod.BBB", "Bayesian by Backprop",
                                  external=False, cited=cited)
    keys = {e["ref_key"] for e in evidence}
    # The citation ref carries the name as its key.
    assert any("Blundell" in k or "Weight Uncertainty" in k for k in keys)
    citation_refs = [e for e in evidence if e["ref_type"] == "citation"]
    assert len(citation_refs) >= 1


def test_ground_merges_local_and_external_when_external_on(
    cfg, ledger, src_dir, tmp_path
):
    """When external is enabled, local corpus AND external literature results
    are merged (not gated on 'local empty')."""
    cfg.external.enabled = True
    svc = PopulateService(cfg, ledger=ledger)
    # Stub the local corpus to return one chunk.
    svc._run_local_only = lambda q: [
        {"chunk_id": "paper#c1", "paper": "VNNs", "doi": "10.1/local"}
    ]
    # Stub the external literature to return one DOI.
    svc._run_external_only = lambda q: [
        {"doi": "10.48550/arXiv.1505.05416",
         "title": "Dropout as a Bayesian Approximation",
         "source": "openalex"}
    ]

    evidence, notes = svc._ground("mod.MCDDropout", "Use MCD",
                                  external=True, cited=[])
    keys = {e["ref_key"] for e in evidence}
    # Both local chunk AND external DOI should be present.
    assert "paper#c1" in keys, f"local chunk missing from {keys}"
    assert "10.48550/arXiv.1505.05416" in keys, f"external DOI missing from {keys}"


def test_draft_is_given_the_local_paper_to_cite_as_primary(
    cfg, ledger, src_dir, fake_ollama
):
    """Regression test for the VAE-vs-VNNs mixup: `populate_symbol` must
    surface the SPECIFIC local paper the code implements to the drafter, with
    an explicit rule against substituting a famous same-name method."""
    from evidence.populate import scan_symbols

    # A local paper the code supposedly implements.
    local_paper = [
        {"chunk_id": "vnn#c0", "paper": "Variational Neural Networks for "
         "Uncertainty Estimation", "year": 2022, "doi": "10.48550/arXiv.2207.01524"}
    ]
    svc = PopulateService(cfg, ledger=ledger)
    # Stub the local search the drafter uses for context.
    svc._local_papers = lambda q, limit=3: local_paper

    # Capture the prompt actually sent to the model.
    captured = {}
    def _capture(prompt, **kwargs):
        captured["prompt"] = prompt
        return {
            "decision": "Compose VariationalLinear layers into an MLP with a "
                        "Gaussian head.",
            "rationale": "Faithful to the paper's Epistemic-NN formulation.",
            "alternatives": [], "depends_on": ["VariationalLinear"],
            "cited": [{"name": "Variational Neural Networks for "
                          "Uncertainty Estimation"}],
        }
    svc.models.complete_json = _capture

    syms = scan_symbols(src_dir, max_symbols=1)
    assert syms, "expected a symbol"
    pr = svc.populate_symbol(syms[0])

    prompt = captured["prompt"]
    # The local paper is surfaced to the drafter by name...
    assert "Variational Neural Networks" in prompt
    # ...and the prompt explicitly warns against substituting a famous
    # same-name method (the exact VAE failure mode).
    assert "do NOT substitute a more famous paper" in prompt
    # The draft's own cited list is preserved (local-first).
    assert pr.cited[0]["name"].startswith("Variational Neural Networks")


def test_populate_local_only_warning_when_external_off(
    cfg, ledger, src_dir, tmp_path, fake_ollama
):
    """When a record is local-only and external was off, a note is added
    suggesting `--external`."""
    cfg.external.enabled = False
    svc = PopulateService(cfg, ledger=ledger)
    # Make _run_local_only return a local chunk (bypasses the app check).
    svc._run_local_only = lambda q: [
        {"chunk_id": "paper#c1", "paper": "VNNs", "doi": ""}
    ]

    from evidence.populate import scan_symbols

    syms = scan_symbols(src_dir, max_symbols=1)
    assert syms, "expected at least one symbol"
    pr = svc.populate_symbol(syms[0], external=False)
    # Should have local evidence and a warning note.
    assert pr.evidence, "expected local evidence"
    assert any("--external" in n for n in pr.notes), (
        f"expected a --external hint in notes: {pr.notes}"
    )
