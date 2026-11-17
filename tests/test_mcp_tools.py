"""Tests for the MCP tool layer (new plan) — offline / deterministic.

We build a local `AppContext` over the seeded store (see conftest) and pass it
explicitly to each tool, so no global singleton, no Ollama, no network. The
network-bound tools (search_literature / expand_citations) are exercised with
their providers disabled so the code path is covered without hitting the net.
"""

from __future__ import annotations

import pytest

from evidence.config import EvidenceConfig
from evidence.facade import AppContext
from evidence.mcp import tools as T


@pytest.fixture
def app(seeded_store, tmp_path, models) -> AppContext:
    """AppContext over the seeded store, with FAISS + BM25 indexes built so
    hybrid retrieval actually returns results."""
    # Use the SAME dir the seeded store lives in, so the retriever finds the
    # FAISS/BM25 artefacts it writes there.
    index_dir = seeded_store.path.parent
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence", index_dir=index_dir)
    app = AppContext(cfg)
    # Point at the seeded store instead of the default on-disk one.
    app.store.close()
    app.store = seeded_store
    app.ctx.store = seeded_store
    app.retriever.store = seeded_store

    # Build FAISS + BM25 over the seeded chunks (deterministic embedder from
    # the `models` fixture — no Ollama needed).
    from evidence.ingestion.pipeline import _rebuild_indexes

    ids = seeded_store.all_chunk_ids()
    texts = [seeded_store.get_chunk(c).text for c in ids]
    _rebuild_indexes(cfg, models, seeded_store, ids, texts)
    app.retriever.faiss = None  # force reload of the just-built index
    app.retriever.bm25 = None
    return app


def test_search_corpus_returns_citation_shapes(app):
    out = T.search_corpus("attention mechanism query key", limit=3, app=app)
    assert isinstance(out, list)
    assert len(out) >= 1
    first = out[0]
    for key in ("chunk_id", "text", "paper", "year", "section", "relevance", "source"):
        assert key in first
    assert first["source"] == "local"
    assert len(first["text"]) <= app.cfg.mcp.max_chunk_chars + 1  # truncated
    # retrieval_mode is surfaced so a client knows how results were produced.
    assert first["retrieval_mode"] in {"hybrid", "bm25_only", "dense_only", "unavailable"}


def test_retrieval_mode_reports_hybrid(app):
    from evidence.facade import retrieval_mode

    # Both FAISS and BM25 were built in the fixture → hybrid.
    assert retrieval_mode(app) == "hybrid"


def test_retrieval_mode_bm25_only(app):
    """Disable the dense backend → mode reports bm25_only."""
    from evidence.facade import retrieval_mode

    app.cfg.retrieval.use_dense = False
    assert retrieval_mode(app) == "bm25_only"
    app.cfg.retrieval.use_dense = True


def test_search_corpus_limit_respected(app):
    out = T.search_corpus("graph neural networks", limit=1, app=app)
    assert len(out) <= 1


def test_expand_via_graph_fallback(app):
    # Seed has p1::c1 (attention) and p2::c1 (GNN). Expanding p1 should surface
    # a chunk from a DIFFERENT paper (p2) via the text-nearest fallback.
    out = T.expand_via_graph("p1::c1", relation="SHARES_TOPIC", limit=5, app=app)
    assert isinstance(out, list)
    for row in out:
        assert row["source"] in {"graph", "graph-fallback"}
        assert row["paper"] != ""  # resolved to a paper title


def test_check_evidence_verdict(app):
    out = T.check_evidence(
        "How does attention combine tokens?",
        ["p1::c1", "p1::c2", "p2::c1"],
        app=app,
    )
    assert out["verdict"] in {"sufficient", "retry"}
    assert isinstance(out["total"], (int, float))
    assert set(out["dimensions"]) == {
        "coverage", "relevance", "depth", "recency", "consistency",
    }
    assert out["n_evidence"] == 3


def test_check_evidence_below_min_shortcircuits(app):
    out = T.check_evidence("q", ["p1::c1"], app=app)  # only 1 < min 3
    assert out["verdict"] == "retry"
    assert out["total"] == 0.0


def test_verify_claim_supported_and_unsupported(app):
    ok = T.verify_claim(
        "attention aligns query and key vectors over tokens",
        "p1::c1",
        app=app,
    )
    assert ok["verdict"] == "SUPPORTED"
    assert ok["score"] > 0.6
    assert "attention" in ok["text"].lower()

    bad = T.verify_claim(
        "quantum entanglement of photons in a vacuum chamber",
        "p1::c1",
        app=app,
    )
    assert bad["verdict"] == "UNSUPPORTED"


def test_verify_claim_missing_chunk(app):
    out = T.verify_claim("anything", "no::such", app=app)
    assert out["verdict"] == "UNSUPPORTED"
    assert out["reason"] == "chunk_id not in index"


def test_get_paper(app):
    rec = T.get_paper("p1", app=app)
    assert rec is not None
    assert rec["title"] == "Attention-based Sequence Models"
    assert rec["doi"] == "10.1000/a"
    assert rec["n_chunks"] == 2
    assert len(rec["outline"]) == 2

    assert T.get_paper("nope", app=app) is None


def test_corpus_stats(app):
    st = T.corpus_stats(app=app)
    assert st["papers"] == 2
    assert st["chunks"] == 3
    assert "index_dir" in st


def test_ingest_gated_by_flag(app):
    # allow_ingest is False by default → tool refuses.
    out = T.ingest_paper("whatever.pdf", app=app)
    assert "error" in out
    assert "allow_ingest" in out["error"]


def test_ingest_paper_enabled_local_txt(app, tmp_path):
    # Enable ingestion and add a local .txt file → it must be indexed.
    app.cfg.mcp.allow_ingest = True
    p = tmp_path / "note.txt"
    p.write_text("Slip angle estimation improves under low-traction conditions. " * 4)
    out = T.ingest_paper(str(p), app=app)
    assert "error" not in out
    assert out["chunks"] >= 1
    # The new chunk is now retrievable / present in the store.
    assert any(app.store.get_chunk(c) is not None for c in [out.get("paper_id", "") + "#c0"])


def test_search_figures_unavailable_envelope(app):
    out = T.search_figures("some figure", app=app)
    assert out["available"] is False
    assert "reason" in out
    assert out["results"] == []


def test_network_tools_noop_when_providers_off(app):
    # Disable all external providers → no network, empty (or fallback) results.
    app.cfg.external.providers = []
    out = T.search_literature("any query", cfg=app.cfg)
    assert out == []
    out2 = T.expand_citations("10.1000/a", cfg=app.cfg)
    assert out2 == []


def test_answer_research_bundle_mode(app):
    """With prefer_client_generation on, answer_research returns a bundle for
    the calling model — no local generation, and it carries instruction +
    sufficiency + retrieved chunks."""
    app.cfg.mcp.prefer_client_generation = True
    out = T.answer_research("How does attention work?", app=app)
    assert out["mode"] == "bundle"
    assert isinstance(out["instruction"], str) and out["instruction"]
    assert "cite" in out["instruction"].lower()
    assert isinstance(out["chunks"], list)
    assert len(out["chunks"]) >= 1
    assert "sufficiency" in out
    assert "next_steps" in out
    # No finished `answer` is produced in bundle mode (client writes it).
    assert "answer" not in out
    app.cfg.mcp.prefer_client_generation = False


def test_research_bundle_is_thin_and_offline(app):
    """Bundle mode must not require the local generation models — it only does
    retrieval + a deterministic sufficiency score."""
    from evidence.mcp.tools import _research_bundle

    bundle = _research_bundle("graph neural networks", app)
    assert bundle["mode"] == "bundle"
    # Every chunk comes from local retrieval and carries the mode.
    assert all(c["source"] == "local" for c in bundle["chunks"])
    assert all(c["retrieval_mode"] for c in bundle["chunks"])


def test_server_builds_and_registers_tools():
    """The FastMCP server builds and registers every tool + the resource."""
    from evidence.mcp import create_server

    server = create_server()
    names = {t.name for t in server._tool_manager.list_tools()}
    expected = {
        "search_corpus", "expand_via_graph", "search_figures", "check_evidence",
        "verify_claim", "get_paper", "search_literature", "expand_citations",
        "ingest_paper", "answer_research", "lookup_evidence", "record_evidence",
        "populate_scope", "populate_symbol",
        "audit_evidence", "impact_analysis", "export_evidence",
    }
    assert expected <= names
    resources = list(server._resource_manager._resources.keys())
    assert "papers://corpus/stats" in resources


def test_populate_symbol_client_authored_grounds_and_records(tmp_path, monkeypatch):
    """populate_symbol: the CALLER writes the decision; the tool fingerprints,
    grounds (via stubbed local/external), and records — no local LLM needed."""
    from evidence.ledger.store import Ledger

    # A real scope so the fingerprint succeeds.
    src = tmp_path / "code"
    src.mkdir()
    (src / "nn.py").write_text(
        "class MCDDropout:\n    def forward(self, x, n_mc=20):\n        return x\n",
        encoding="utf-8",
    )

    # Stub the grounding sources (no network / no real index): one external DOI.
    from evidence.populate import PopulateService

    real_ground = PopulateService._ground

    def _stub_ground(self, symbol, decision, *, external=None, cited=None):
        self._run_local_only = lambda q: [
            {"chunk_id": "vnn#c1", "paper": "VNNs", "doi": ""}
        ]
        self._run_external_only = lambda q: [
            {"doi": "10.48550/arXiv.1505.05416",
             "title": "Dropout as a Bayesian Approximation",
             "source": "openalex"}
        ]
        return real_ground(self, symbol, decision,
                           external=external, cited=cited)

    monkeypatch.setattr(PopulateService, "_ground", _stub_ground)

    # Hermetic ledger.
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(tmp_path / "ledger.sqlite"))
    from evidence.config import EvidenceConfig

    cfg = EvidenceConfig.load()
    cfg.external.enabled = True  # so the external branch of _ground runs
    import evidence.ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "_LEDGER", None, raising=False)

    out = T.populate_symbol(
        "nn.MCDDropout",
        "Implement MCD (Bayesian dropout) as an epistemic baseline.",
        rationale="Standard BMA-family comparison.",
        alternatives=[{"name": "BBB", "why_rejected": "heavier memory"}],
        cited=[{"name": "Dropout as a Bayesian Approximation",
                "authors": "Gal & Ghahramani"}],
        scope=str(src),
        external=True,
        cfg=cfg,
    )

    # Grounded with BOTH local and external refs (merged).
    keys = {e["ref_key"] for e in out["evidence"]}
    assert "vnn#c1" in keys, f"local ref missing from {keys}"
    assert "10.48550/arXiv.1505.05416" in keys, f"external DOI missing from {keys}"
    assert out["grounded"] is True
    assert out["cited"][0]["name"] == "Dropout as a Bayesian Approximation"

    # A fingerprint was computed from the scope.
    lg = Ledger(tmp_path / "ledger.sqlite")
    rec = lg.lookup("nn.MCDDropout")
    assert rec["code_fingerprint"], "expected a non-empty code fingerprint"
    assert rec["decision"].startswith("Implement MCD")
    lg.close()


def test_record_evidence_persists_cited(tmp_path):
    """record_evidence accepts a `cited` list and the ledger round-trips it."""
    from evidence.ledger.store import Ledger

    lg = Ledger(tmp_path / "ledger.sqlite")
    out = T.record_evidence(
        "nn.MCDDropout",
        "Implement MCD (Bayesian dropout).",
        evidence=[{"ref_type": "doi", "ref_key": "10.48550/arXiv.1505.05416",
                    "source": "cited"}],
        cited=[{"name": "Dropout as a Bayesian Approximation",
                "authors": "Gal & Ghahramani",
                "doi": "10.48550/arXiv.1505.05416"}],
        ledger=lg,
    )
    assert out["symbol"] == "nn.MCDDropout"
    # `cited` is surfaced on the returned record.
    assert out["cited"][0]["name"] == "Dropout as a Bayesian Approximation"
    # ...and is persisted so a subsequent lookup returns it.
    assert lg.lookup("nn.MCDDropout")["cited"][0]["doi"] == "10.48550/arXiv.1505.05416"
    lg.close()


def test_populate_scope_runs_scan_draft_ground_record(
    tmp_path, monkeypatch
):
    """populate_scope is a thin wrapper over PopulateService.run — it scans,
    drafts (via the local model), grounds, and records into the ledger."""
    import json
    import sys
    from types import SimpleNamespace

    from evidence.ledger.store import Ledger

    # A tiny scope with one symbol.
    src = tmp_path / "code"
    src.mkdir()
    (src / "mod.py").write_text(
        '"""Demo."""\n\n\ndef compute(items):\n    """Sum."""\n    return sum(items)\n',
        encoding="utf-8",
    )

    # A canned reasoning-model response (with a `cited` entry).
    class _FakeOllama:
        def __init__(self, **_):
            pass

        def chat(self, **kwargs):
            return {"message": {"content": json.dumps({
                "decision": "Use a plain built-in sum() for the total.",
                "rationale": "Stdlib-only, O(n).",
                "alternatives": [{"name": "numpy.sum", "why_rejected": "heavy dep"}],
                "depends_on": [],
                "cited": [{"name": "PEP 287", "authors": "C. Linsley"}],
            })}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: _FakeOllama()))

    # Point the global ledger at a temp file so the tool's get_ledger() hits it.
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(tmp_path / "ledger.sqlite"))
    from evidence.config import EvidenceConfig

    cfg = EvidenceConfig.load()
    cfg.external.enabled = False  # keep hermetic
    # Prevent the global cache from returning a stale ledger from another test.
    import evidence.ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "_LEDGER", None, raising=False)

    out = T.populate_scope(str(src), limit=5, force=False, external=False, cfg=cfg)

    assert out["scanned"] >= 1
    assert len(out["recorded"]) >= 1
    first = out["recorded"][0]
    for key in ("symbol", "decision", "evidence", "cited", "grounded", "notes"):
        assert key in first
    # The model's `cited` entry is carried through to the record.
    assert first["cited"], "expected a cited entry to be captured"

    # The record was actually written into the ledger.
    lg = Ledger(tmp_path / "ledger.sqlite")
    assert lg.count_records() >= 1
    lg.close()
