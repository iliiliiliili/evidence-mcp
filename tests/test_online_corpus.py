"""Tests for external/online_corpus.py — save + lazy DOI resolution.

All tests run offline: the full-doc path and the network paths are stubbed;
the metadata-only path and the cache / local-corpus short-circuit are
exercised against the real SQLite Store.
"""

from __future__ import annotations

import json

from evidence.config import EvidenceConfig
from evidence.external import online_corpus as oc
from evidence.indexes.store import PaperRecord


# ── _normalize_doi ─────────────────────────────────────────────────────────
def test_normalize_doi_bare():
    assert oc._normalize_doi("10.1000/abc") == "10.1000/abc"


def test_normalize_doi_prefix():
    assert oc._normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"


def test_normalize_doi_empty():
    assert oc._normalize_doi("") is None
    assert oc._normalize_doi(None) is None


def test_paper_id_for_doi_stable():
    assert oc._paper_id_for_doi("10.1000/abc") == "online__doi__10.1000/abc"


# ── _work_to_meta ──────────────────────────────────────────────────────────
def test_work_to_meta_basic():
    w = {
        "doi": "10.1000/abc",
        "title": "A paper",
        "year": 2020,
        "authors": ["Alice", "Bob"],
        "abstract": "The abstract.",
        "pdf_url": "https://example.com/a.pdf",
        "source": "openalex",
    }
    meta = oc._work_to_meta(w)
    assert meta["doi"] == "10.1000/abc"
    assert meta["title"] == "A paper"
    assert meta["year"] == 2020
    assert meta["authors"] == ["Alice", "Bob"]
    assert meta["abstract"] == "The abstract."
    assert meta["pdf_url"] == "https://example.com/a.pdf"


def test_work_to_meta_accepts_citation_shape():
    # search_literature returns `paper` rather than `title` for some providers.
    w = {"doi": "10.1000/x", "paper": "P", "year": 2021, "source": "s2"}
    meta = oc._work_to_meta(w)
    assert meta["title"] == "P"
    assert meta["pdf_url"] is None


# ── save_online_ref — flag OFF (metadata-only) ─────────────────────────────
def test_save_flag_off_writes_meta_only(tmp_path, models):
    cfg = EvidenceConfig(
        data_dir=tmp_path / "evidence",
        index_dir=tmp_path / "idx",
    )
    cfg.external.save_online_to_corpus = False
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        report = oc.save_online_ref(
            {"doi": "10.1000/abc", "title": "Paper A", "year": 2020,
             "abstract": "An abstract.", "authors": ["A"],
             "pdf_url": "https://example.com/a.pdf"},
            cfg=cfg, models=models, store=store,
        )
        assert report["saved"] is True
        assert report["mode"] == "meta"
        assert report["chunks"] == 1
        paper = store.get_paper("online__doi__10.1000/abc")
        assert paper is not None
        assert paper.title == "Paper A"
        assert paper.doi == "10.1000/abc"
        assert "An abstract." in (paper.abstract or "")
        chunk = store.get_chunk("online__doi__10.1000/abc#c0")
        assert chunk is not None
        assert chunk.heading == "Abstract"
    finally:
        store.close()


def test_save_idempotent(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    cfg.external.save_online_to_corpus = False
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        work = {"doi": "10.1000/abc", "title": "P", "abstract": "A."}
        first = oc.save_online_ref(work, cfg=cfg, models=models, store=store)
        second = oc.save_online_ref(work, cfg=cfg, models=models, store=store)
        assert first["saved"] and first["mode"] == "meta"
        assert second["saved"] and second["mode"] == "already"
    finally:
        store.close()


def test_save_no_doi_no_id_is_noop(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        report = oc.save_online_ref({"title": "no doi"}, cfg=cfg,
                                    models=models, store=store)
        assert report["saved"] is False
        assert "no DOI or id" in (report["error"] or "")
        assert store.count_papers() == 0
    finally:
        store.close()


def test_save_flag_on_no_pdf_url_writes_meta_only(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    # Flag ON but no PDF URL → metadata-only path.
    cfg.external.save_online_to_corpus = True
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        report = oc.save_online_ref(
            {"doi": "10.1000/x", "title": "P", "abstract": "A."},
            cfg=cfg, models=models, store=store,
        )
        assert report["saved"] is True
        assert report["mode"] == "meta"
    finally:
        store.close()


# ── resolve_doi_meta — cache + local-corpus short-circuit ─────────────────
def test_resolve_cache_hit(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        store.kv_set(oc._kv_key("10.1000/abc"),
                     json.dumps({"title": "Cached title", "year": 2019,
                                "doi": "10.1000/abc"}))
        # allow_network=False proves we don't even hit the net.
        out = oc.resolve_doi_meta("10.1000/abc", cfg=cfg, store=store,
                                  allow_network=False)
        assert out["title"] == "Cached title"
        assert out["year"] == 2019
    finally:
        store.close()


def test_resolve_local_corpus_short_circuits(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        store.upsert_paper(PaperRecord(
            paper_id=oc._paper_id_for_doi("10.1000/xyz"),
            title="Saved locally",
            path="", source_format="online", n_pages=0,
            year=2022, doi="10.1000/xyz", authors=[],
            abstract=None, meta={},
        ))
        out = oc.resolve_doi_meta("10.1000/xyz", cfg=cfg, store=store,
                                  allow_network=False)
        assert out["title"] == "Saved locally"
        assert out["year"] == 2022
    finally:
        store.close()


def test_resolve_normalizes_prefix_and_case(tmp_path, models):
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        # Cache is keyed on the normalised (lower, prefix-stripped) DOI.
        store.kv_set(oc._kv_key("10.1000/abc"),
                     json.dumps({"title": "T", "year": 2000,
                                "doi": "10.1000/abc"}))
        out = oc.resolve_doi_meta("https://doi.org/10.1000/ABC", cfg=cfg,
                                  store=store, allow_network=False)
        assert out["title"] == "T"
    finally:
        store.close()


def test_resolve_no_doi_returns_none_title():
    cfg = EvidenceConfig()
    out = oc.resolve_doi_meta("", cfg=cfg)
    assert out["title"] is None
    assert out["doi"] is None


# ── save_online_ref → full-doc path (stubbed download + extract) ──────────
def test_save_flag_on_full_doc(tmp_path, models, monkeypatch):
    """When the flag is on and a PDF URL is available, the full-doc path runs
    (stubbed download + extract) and yields mode='full' with the extracted
    chunks indexed under the DOI paper id."""
    cfg = EvidenceConfig(data_dir=tmp_path / "evidence",
                         index_dir=tmp_path / "idx")
    cfg.external.save_online_to_corpus = True
    from evidence.indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        # Stub the two I/O dependencies so no network / no PDF toolchain is
        # needed: _download returns a marker path, extract_pdf returns a fake
        # document with known text.
        monkeypatch.setattr(
            oc, "_download", lambda url: tmp_path / "fake.pdf",
        )

        class _FakeDoc:
            text = "Introduction\n\nThis is the body of the paper.\n\n" \
                   "Methods\n\nWe use a deterministic approach."
            n_pages = 2
            title = "Full-doc title"

        import evidence.ingestion.pdf as pdf_mod

        monkeypatch.setattr(pdf_mod, "extract_pdf", lambda p: _FakeDoc())

        report = oc.save_online_ref(
            {"doi": "10.1000/full", "title": "Full-doc title", "year": 2021,
             "abstract": "Abstract text.", "authors": ["A"],
             "pdf_url": "https://example.com/full.pdf"},
            cfg=cfg, models=models, store=store,
        )
        assert report["saved"] is True
        assert report["mode"] == "full"
        assert report["chunks"] >= 1
        paper = store.get_paper("online__doi__10.1000/full")
        assert paper is not None
        assert paper.source_format == "online"
        # At least one chunk should contain the body text we stubbed in.
        chunks = store.chunks_for_paper("online__doi__10.1000/full")
        joined = "\n".join(c.text for c in chunks)
        assert "deterministic approach" in joined
    finally:
        store.close()
