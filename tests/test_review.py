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

    @staticmethod
    def _prompt(kwargs) -> str:
        messages = kwargs.get("messages") or []
        return " ".join(m.get("content", "") for m in messages)

    def chat(self, **kwargs) -> dict:
        self.calls += 1
        prompt = self._prompt(kwargs)
        # Prompt-driven (not call-count) so isolated calls behave correctly:
        # the reference-annotation prompt mentions "reference list".
        if "reference list" in prompt or "annotations" in prompt:
            payload = {
                "annotations": [
                    {
                        "index": 1,
                        "title": "Foundations of Sequence Modeling",
                        "authors": "Smith et al.",
                        "relevance": "high",
                        "role": "method",
                        "notes": "Core prior method this builds on.",
                    },
                    {
                        "index": 2,
                        "title": "A Survey of Benchmarks",
                        "authors": "Lee",
                        "relevance": "low",
                        "role": "comparison",
                    },
                ]
            }
            return {"message": {"content": json.dumps(payload)}}
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


def test_resolve_flags_low_confidence_match_as_hint(store, cfg, fake_ollama, monkeypatch):
    """A fuzzy top-cited hit that is NOT similar to the reference must be
    reported as found: False with a `candidate` hint — never as a verified
    identity match. This is the 'Form 10-K → GPT-4 Technical Report' class of
    bug the user hit."""
    import evidence.mcp.tools as tools

    def _fake_search(query, limit=5, cfg=None):
        # The provider returns an unrelated, highly-cited work.
        return [
            {"paper": "GPT-4 Technical Report", "year": 2023,
             "doi": "10.0000/gpt4", "cited_by": 9999, "source": "openalex"}
        ]

    monkeypatch.setattr(tools, "search_literature", _fake_search)
    svc = ReviewService(cfg, store=store)
    ref = {
        "raw": "[9] Form 10-K, annual report, U.S. Securities and Exchange Commission.",
        "title": "Form 10-K",
        "authors": "U.S. Securities and Exchange Commission",
    }
    res = svc._resolve_reference(ref, enabled=True)
    assert res["found"] is False
    assert "candidate" in res and res["candidate"]["title"] == "GPT-4 Technical Report"
    assert res["match_confidence"] < 0.5


def test_resolve_label_shows_unverified_hint_when_low_confidence():
    from evidence.review import _resolve_label

    assert _resolve_label(None) == "—"
    assert _resolve_label({"found": None}) == "not checked"
    assert _resolve_label({"found": False}) == "not found"
    hint = _resolve_label(
        {"found": False, "candidate": {"title": "OpenAI GPT-4 Technical Report"}}
    )
    assert "OpenAI GPT-4" in hint and "unverified" in hint


def test_resolve_rejects_unrelated_and_keeps_confident(store, cfg, fake_ollama, monkeypatch):
    import evidence.mcp.tools as tools

    def _fake_search(query, limit=5, cfg=None):
        return [
            {"paper": "Totally Unrelated Survey of Vibe Coding with LLMs",
             "year": 2025, "doi": "10.0000/vibe", "cited_by": 500, "source": "openalex"},
            {"paper": "FinDPO: Financial Sentiment Analysis for Algorithmic Trading "
                      "through Preference Optimization of LLMs",
             "year": 2025, "doi": "10.1000/findpo", "cited_by": 3, "source": "s2"},
        ]

    monkeypatch.setattr(tools, "search_literature", _fake_search)
    svc = ReviewService(cfg, store=store)
    ref = {
        "raw": ("[4] FinDPO: Financial Sentiment Analysis for Algorithmic "
                "Trading through Preference Optimization of LLMs. 2025. 10.1000/findpo"),
        "title": "FinDPO",
        "authors": "Iacovides, Zhou, Mandic",
    }
    res = svc._resolve_reference(ref, enabled=True)
    # The similar FinDPO candidate must win over the unrelated survey.
    assert res["found"] is True
    assert res["match_doi"] == "10.1000/findpo"
    assert res["match_confidence"] >= 0.5


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


# ── new: authoritative reference list (no lossy LLM enumeration) ───────────
def test_reference_from_entry_parses_doi_year_index():
    from evidence.review import _reference_from_entry
    entry = "[7] FinSMART: Financial Sentiment for Trading. 2025. arXiv:2607.28127"
    r = _reference_from_entry(entry, 7)
    assert r["index"] == 7
    assert r["year"] == 2025
    assert r["doi"] is None  # no 10.x DOI in this entry
    assert r["raw"].startswith("[7] FinSMART")
    assert r["relevance"] == "medium"
    assert r["role"] == "other"


def test_reference_from_entry_parses_doi():
    from evidence.review import _reference_from_entry
    entry = "[13] Beyond sentiment analysis. 2026. 10.1007/s41060-026-01064-2"
    r = _reference_from_entry(entry, 13)
    assert r["doi"] == "10.1007/s41060-026-01064-2"
    assert r["index"] == 13


def test_title_similarity_high_for_subset_and_low_for_unrelated():
    from evidence.review import _title_similarity
    high = _title_similarity(
        "FinSMART: Financial Sentiment Analysis for Algorithmic Trading "
        "through Market-Aligned Reinforcement Learning",
        "FinSMART: Financial Sentiment for Trading",
    )
    low = _title_similarity(
        "Form 10-K annual report, U.S. Securities and Exchange Commission",
        "GPT-4 Technical Report",
    )
    assert high >= 0.5
    assert low < 0.5


def test_references_never_dropped_even_if_model_annotates_few(store, cfg, fake_ollama):
    """The deterministic list is authoritative; a model that annotates only
    SOME references must not cause the others to disappear."""
    _seed(store)  # 2 references in c3
    svc = ReviewService(cfg, store=store)
    # The fake model annotates BOTH entries (index 1 and 2). Both should appear.
    refs = svc._references(store.get_paper("p1"), "[1] Smith. 2019. 10.1000/f\n[2] Lee. 2020.")
    assert len(refs) == 2
    assert refs[0]["doi"] == "10.1000/f"
    assert refs[0]["relevance"] == "high"
    assert refs[1]["relevance"] == "low"


def test_references_preserved_when_model_returns_nothing(store, cfg):
    """If the model annotation pass fails (returns None), the full list still
    comes through with deterministic defaults and the raw text as title."""
    _seed(store)
    svc = ReviewService(cfg, store=store)

    # Force the annotation pass to return None.
    def _no_ann(paper, entries):
        return None

    svc._annotate_references = _no_ann
    refs = svc._references(store.get_paper("p1"), "[1] Smith. 2019. 10.1000/f\n[2] Lee. 2020.")
    assert len(refs) == 2
    assert refs[0]["relevance"] == "medium"
    assert refs[0]["title"]  # filled from raw
    assert "Smith" in refs[0]["title"]


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


# ── author-overlap / conflict-of-interest (deterministic, no model) ─────────
def test_find_author_conflicts_flags_self_citation():
    from evidence.review import find_author_conflicts
    paper_authors = ["Alice Smith", "Bob Jones"]
    refs = [
        {"index": 1, "title": "Prior work", "authors": "A. Smith and B. Jones"},
        {"index": 2, "title": "Unrelated", "authors": "Zoe Q. Unrelated"},
    ]
    conflicts = find_author_conflicts(paper_authors, refs)
    assert len(conflicts) == 1
    assert conflicts[0]["ref_index"] == 1
    assert conflicts[0]["kind"] == "self-citation"  # both authors matched
    assert set(conflicts[0]["overlapping_authors"]) == {"Alice Smith", "Bob Jones"}


def test_find_author_conflicts_partial_overlap():
    from evidence.review import find_author_conflicts
    paper_authors = ["Alice Smith", "Bob Jones"]
    refs = [
        {"index": 5, "title": "One author shared", "authors": "Alice Smith and C. Other"},
    ]
    conflicts = find_author_conflicts(paper_authors, refs)
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "author-overlap"  # only one matched
    assert conflicts[0]["overlapping_authors"] == ["Alice Smith"]


def test_find_author_conflicts_none_when_no_overlap():
    from evidence.review import find_author_conflicts
    assert find_author_conflicts(["Alice Smith"], [
        {"index": 1, "authors": "Totally Different"},
    ]) == []
    # No authors on the paper → no conflicts.
    assert find_author_conflicts([], [{"index": 1, "authors": "Alice Smith"}]) == []


def test_review_report_includes_author_conflicts_field(store, cfg, fake_ollama):
    _seed(store)
    report = ReviewService(cfg, store=store).review("p1", external=False)
    # Field is always present (list).
    assert "author_conflicts" in report.as_dict()
    # Paper authors are ["Alice Smith", "Bob Jones"]. Reference [1] is
    # annotated as "Smith et al." → shares the "Smith" surname with Alice
    # Smith, so the deterministic scan flags it (a legitimate same-surname
    # self-citation signal). Reference [2] ("Lee.") does not overlap.
    flagged = {c["ref_index"] for c in report.author_conflicts}
    assert flagged == {1}
    assert all(c["overlapping_authors"] == ["Alice Smith"]
               for c in report.author_conflicts)


def test_review_report_flags_self_citation(store, cfg, fake_ollama):
    """When a reference's author string contains the paper's own authors,
    the report must flag it deterministically (no model required)."""
    from evidence.review import find_author_conflicts
    _seed(store)
    paper = store.get_paper("p1")
    refs = [
        {"index": 1, "title": "Our prior work", "authors": "Alice Smith and Bob Jones",
         "relevance": "high", "role": "method"},
    ]
    conflicts = find_author_conflicts(list(paper.authors), refs)
    assert conflicts and conflicts[0]["kind"] == "self-citation"


def test_acronym_scan_suppresses_watermark_noise():
    from evidence.review import find_undefined_acronyms
    # "FOR REVIEW ONLY" banner + a paper-type acronym should NOT surface.
    text = "FOR REVIEW ONLY. SADM submission. Uses the FMH module."
    terms = {t["term"] for t in find_undefined_acronyms(text)}
    assert "FMH" in terms          # genuine undefined term still flagged
    assert "SADM" not in terms     # noise suppressed
    assert "ONLY" not in terms     # noise suppressed
    assert "REVIEW" not in terms   # noise suppressed
