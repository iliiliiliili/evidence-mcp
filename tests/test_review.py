"""Tests for `evidence review` — structured scientific review of one paper.

Hermetic: a temp ``Store`` (no network), a faked Ollama client that returns
canned JSON for the two review phases, and external reference lookup disabled
(or stubbed) so nothing touches the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence.config import EvidenceConfig
from evidence.indexes.store import Store
from evidence.review import (
    ReviewService,
    find_undefined_acronyms,
    references_text,
    resolve_paper,
    review_paper,
    split_references,
)


# ── faked Ollama client (two phases: analyze, then references) ──────────────
class _FakeOllamaClient:
    def __init__(self, **_):
        self.calls = 0

    def chat(self, **kwargs) -> dict:
        self.calls += 1
        if self.calls == 1:
            payload = {
                "summary": "Proposes a mechanism and reports an improvement on a benchmark.",
                "strengths": ["clear method description", "reproducible setup"],
                "weaknesses": ["small dataset", "no ablation"],
                "incorrect_statements": [
                    {
                        "statement": "This is the first method to use the FMH module.",
                        "problem": "The FMH module predates this paper.",
                        "chunk_id": "p1::c2",
                        "confidence": "medium",
                    }
                ],
                "undefined_terms": [
                    {
                        "term": "FMH",
                        "context": "Methods",
                        "suggestion": "Define FMH on first use.",
                    }
                ],
            }
        else:
            payload = {
                "references": [
                    {
                        "index": 1,
                        "title": "Foundations of Sequence Modeling",
                        "authors": "Smith et al.",
                        "year": 2019,
                        "doi": "10.1000/f",
                        "relevance": "high",
                        "role": "method",
                        "notes": "Core prior method this builds on.",
                    },
                    {
                        "index": 2,
                        "title": "A Survey of Benchmarks",
                        "authors": "Lee",
                        "year": 2020,
                        "relevance": "low",
                        "role": "comparison",
                    },
                ]
            }
        return {"message": {"content": json.dumps(payload)}}


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> _FakeOllamaClient:
    fake = _FakeOllamaClient()
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: fake))
    return fake


@pytest.fixture
def cfg(tmp_path: Path) -> EvidenceConfig:
    c = EvidenceConfig()
    c.data_dir = tmp_path
    c.external.enabled = False  # no network
    return c


@pytest.fixture
def store(cfg: EvidenceConfig) -> Store:
    s = Store(cfg.resolved_store_path)
    return s


def _seed(store: Store) -> None:
    from evidence.indexes.store import ChunkRecord, PaperRecord

    p = PaperRecord(
        paper_id="p1", title="Attention-based Sequence Models", path="a.pdf",
        source_format="pdf", n_pages=2, year=2021, doi="10.1000/a",
        authors=["Alice Smith", "Bob Jones"],
        abstract="A mechanism over tokens.",
        meta={},
    )
    store.upsert_paper(p)
    store.upsert_chunk(ChunkRecord(
        chunk_id="p1::c1", paper_id="p1", ordinal=0, heading="Intro", page=1,
        text="We propose a mechanism that uses the FMH module over the sequence.",
        tokens=12, meta={},
    ))
    store.upsert_chunk(ChunkRecord(
        chunk_id="p1::c2", paper_id="p1", ordinal=1, heading="Methods", page=2,
        text="We train with the Adam optimizer.",
        tokens=8, meta={},
    ))
    store.upsert_chunk(ChunkRecord(
        chunk_id="p1::c3", paper_id="p1", ordinal=2, heading="References", page=2,
        text="[1] Smith et al. Foundations of Sequence Modeling. 2019. 10.1000/f\n"
             "[2] Lee. A Survey of Benchmarks. 2020.",
        tokens=10, meta={},
    ))


# ── paper resolution ────────────────────────────────────────────────────────
def test_resolve_paper_by_id_and_title(store):
    _seed(store)
    assert resolve_paper(store, "p1") == ("p1", store.get_paper("p1"))
    by_title = resolve_paper(store, "attention-based")
    assert by_title is not None and by_title[0] == "p1"
    assert resolve_paper(store, "does not exist") is None


# ── deterministic acronym scan ──────────────────────────────────────────────
def test_find_undefined_acronyms_flags_used_but_undefined():
    text = "The model uses the FMH module. Later we reuse FMH. See the BGE-M3 encoder."
    terms = {t["term"] for t in find_undefined_acronyms(text)}
    assert "FMH" in terms
    # A common/known token is suppressed.
    assert "GPU" not in terms


def test_find_undefined_acronyms_ignores_defined():
    text = "We introduce the Flow-Mixing Heuristic (FMH) module and use FMH."
    terms = {t["term"] for t in find_undefined_acronyms(text)}
    assert "FMH" not in terms


# ── full review (offline, two model phases) ────────────────────────────────
def test_review_produces_five_part_report(store, cfg, fake_ollama):
    _seed(store)
    report = ReviewService(cfg, store=store).review("p1", external=False)

    assert report.paper_id == "p1"
    assert report.n_chunks == 3
    assert report.summary
    assert report.strengths and report.weaknesses
    assert report.incorrect_statements
    assert any(t["term"] == "FMH" for t in report.undefined_terms)
    assert len(report.references) == 2
    ref0 = report.references[0]
    assert ref0["relevance"] == "high" and ref0["role"] == "method"
    assert ref0["resolved"]["found"] is None  # external disabled
    # Both model phases ran.
    assert fake_ollama.calls == 2


def test_review_reference_roles_and_doIs(store, cfg, fake_ollama):
    _seed(store)
    report = ReviewService(cfg, store=store).review("p1", external=False)
    ref0 = report.references[0]
    assert ref0["doi"] == "10.1000/f"
    assert ref0["year"] == 2019
    assert report.references[1]["relevance"] == "low"


# ── graceful degradation with no model ──────────────────────────────────────
def test_review_without_model_falls_back_to_abstract(store, cfg, monkeypatch):
    _seed(store)
    # Point the model at a model name that will fail (no server) → fallback.
    cfg.ollama.reasoning = "definitely-not-installed:9b"
    cfg.ollama.auto_pull = False
    # Simulate the ollama client raising on every call.
    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=lambda **kw: _BrokenClient()),
    )
    report = ReviewService(cfg, store=store).review("p1", external=False)
    # Falls back to the abstract for the summary.
    assert "A mechanism over tokens." in report.summary
    assert report.strengths == [] and report.weaknesses == []
    # Deterministic acronym scan still surfaces an undefined term.
    assert any(t["term"] == "FMH" for t in report.undefined_terms)


class _BrokenClient:
    def chat(self, **_):
        raise RuntimeError("no server")


# ── external reference resolution (stubbed, no network) ─────────────────────
def test_review_resolves_references_when_external(store, cfg, fake_ollama, monkeypatch):
    import evidence.mcp.tools as tools

    _seed(store)
    calls = []

    def _fake_search(query, limit=1, cfg=None):
        calls.append(query)
        return [
            {"paper": "Foundations of Sequence Modeling", "year": 2019,
             "doi": "10.1000/f", "cited_by": 42, "source": "openalex"}
        ]

    monkeypatch.setattr(tools, "search_literature", _fake_search)
    report = ReviewService(cfg, store=store).review("p1", external=True, resolve_limit=5)
    ref0 = report.references[0]
    assert ref0["resolved"]["found"] is True
    assert ref0["resolved"]["citations"] == 42
    assert calls  # at least one lookup happened


# ── public entry point ──────────────────────────────────────────────────────
def test_review_paper_entry_point(store, cfg, fake_ollama):
    _seed(store)
    result = review_paper("p1", external=False, cfg=cfg)
    assert result["paper_id"] == "p1"
    assert "summary" in result and "references" in result
    # Unknown paper returns a structured error, not an exception.
    missing = review_paper("nope", cfg=cfg)
    assert "error" in missing


def test_review_paper_markdown_has_sections(store, cfg, fake_ollama):
    _seed(store)
    md = ReviewService(cfg, store=store).review("p1", external=False).to_markdown()
    assert "# Scientific Review" in md
    for heading in ("## 1. Summary", "## 4. Possibly incorrect statements",
                    "## 5. Undefined terms", "## 6. References"):
        assert heading in md


# ── reference-list splitting (the "check the references" fix) ───────────────
def test_split_references_numbered():
    text = ("[1] Smith et al. Foundations of Sequence Modeling. 2019.\n"
            "[2] Lee. A Survey of Benchmarks. 2020.\n"
            "[3] A very long reference that wraps across\n    multiple lines here.")
    entries = split_references(text)
    assert len(entries) == 3
    assert "Foundations of Sequence Modeling" in entries[0]
    assert "A Survey of Benchmarks" in entries[1]
    # wrapped continuation is kept with entry [3]
    assert "multiple lines here" in entries[2]


def test_split_references_dot_style_and_unnumbered():
    assert len(split_references("1. First work.\n2. Second work.")) == 2
    # No markers at all → the whole thing is one entry (better than empty).
    assert split_references("Just a paragraph of references") == [
        "Just a paragraph of references"
    ]
    assert split_references("") == []


def test_references_text_and_split_from_store(store, cfg):
    _seed(store)  # c3 is heading="References" with two [n] entries
    entries = split_references(references_text(store.chunks_for_paper("p1")))
    assert len(entries) == 2
    assert any("Foundations of Sequence Modeling" in e for e in entries)


# ── facade primitives: get_paper / get_chunk / get_references ───────────────
def _fake_app(cfg, store):
    return SimpleNamespace(store=store, cfg=cfg)


def test_get_paper_derives_title_and_references(store, cfg):
    from evidence.facade import get_paper

    _seed(store)
    # Stored title is a real title here; use a weak one to exercise derivation.
    p = store.get_paper("p1")
    p.title = "11299"  # simulate a filename-stem stored title
    store.upsert_paper(p)
    app = _fake_app(cfg, store)
    rec = get_paper("11299", app=app)  # title-substring / id lookup
    assert rec is not None
    # Derived from the first chunk's text (starts "We propose a mechanism…")
    assert rec["title"] != "11299"
    # References are now inline + split into entries.
    assert isinstance(rec["references"], list) and len(rec["references"]) == 2
    # stored_title preserves the original for provenance.
    assert rec["stored_title"] == "11299"


def test_get_paper_accepts_title_substring(store, cfg):
    from evidence.facade import get_paper

    _seed(store)
    app = _fake_app(cfg, store)
    assert get_paper("attention-based", app=app) is not None
    assert get_paper("does not exist", app=app) is None


def test_get_chunk_returns_full_text_and_context(store, cfg):
    from evidence.facade import get_chunk

    _seed(store)
    app = _fake_app(cfg, store)
    rec = get_chunk("p1::c2", app=app)
    assert rec is not None
    # Full, untruncated text (not clipped to max_chunk_chars).
    assert rec["text"] == "We train with the Adam optimizer."
    assert rec["section"] == "Methods"
    assert any(c["chunk_id"] == "p1::c1" for c in rec["context"])
    assert get_chunk("nope::c0", app=app) is None


def test_get_references_structured_and_untruncated(store, cfg):
    from evidence.facade import get_references

    _seed(store)
    app = _fake_app(cfg, store)
    refs = get_references("p1", app=app)
    assert len(refs) == 2
    assert any("10.1000/f" in r for r in refs)


# ── search_literature identifier fast-path ──────────────────────────────────
def test_extract_identifier_dois_and_arxiv():
    from evidence.mcp.tools import _extract_identifier

    assert _extract_identifier("see 10.1000/f for details") == "10.1000/f"
    assert _extract_identifier("doi.org/10.1145/123.456,") == "10.1145/123.456"
    assert _extract_identifier("arXiv:2510.05710v2") == "2510.05710"
    assert _extract_identifier("plain title search") is None
    # DOI wins when both are present.
    assert _extract_identifier("arXiv:2510.05710 (DOI 10.48550/arXiv.2510.05710)") == (
        "10.48550/arXiv.2510.05710"
    )


def test_search_literature_fast_path_resolves_doi_directly(cfg, monkeypatch):
    import evidence.mcp.tools as T

    # Stub the provider so the fast path resolves without the network.
    monkeypatch.setattr(
        T.crossref, "get_work",
        lambda doi, mailto=None: {"doi": doi, "title": "The Right Paper", "year": 2019},
    )
    out = T.search_literature("check 10.1000/f", cfg=cfg)
    assert len(out) == 1
    assert out[0]["paper"] == "The Right Paper"
    assert out[0]["match"] == "identifier"
    assert out[0]["resolved_via"] == "10.1000/f"
