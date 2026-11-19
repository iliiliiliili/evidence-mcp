"""Tests for hierarchical chunking (sections) and the chunk-graph (links).

Hermetic: a temp ``Store`` (no network), a faked Ollama client for the LLM
link pass, and no external services. Verifies:
  - the plain-text section detector (numbered, known labels, all-caps);
  - ``chunk_document`` assigns ``section_id`` + first/last flags;
  - ``facade.get_sections`` / ``get_section`` stitch a span back together;
  - ``links.build_chunk_links`` emits deterministic links (always) and, when a
    model is available, LLM links on top — and NEVER hard-fails;
  - ``links.linked_chunks`` / ``links_for_paper`` expose the graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence.config import EvidenceConfig
from evidence.indexes.store import ChunkRecord, PaperRecord, Store


# ── faked Ollama client for the LLM link pass ───────────────────────────────
class _FakeOllamaClient:
    def __init__(self, **_):
        self.calls = 0

    def list_models(self):
        return {"models": [{"name": "reasoning"}]}

    def chat(self, **kwargs) -> dict:
        self.calls += 1
        payload = {
            "links": [
                {
                    "src_chunk_id": "p1::c0",
                    "dst_chunk_id": "p1::c2",
                    "relation": "depends_on",
                    "confidence": 0.8,
                    "reason": "Results rely on the Method setup.",
                }
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
    c.external.enabled = False
    c.ollama.reasoning = "reasoning"
    return c


@pytest.fixture
def store(cfg: EvidenceConfig) -> Store:
    return Store(cfg.resolved_store_path)


def _seed_paper_and_chunks(store: Store, n: int = 4, section: str = "Method") -> None:
    p = PaperRecord(
        paper_id="p1",
        title="A Method Paper",
        path="a.pdf",
        source_format="pdf",
        n_pages=1,
        year=2024,
        doi="10.1000/a",
        authors=["A. Author"],
        abstract="A paper with several chunks in one section.",
        meta={},
    )
    store.upsert_paper(p)
    texts = [
        "We introduce the core mechanism of the method.",
        "The mechanism is trained with a specific loss.",
        "Experiments show the method improves over baselines.",
        "Ablations confirm the loss is the key factor.",
    ]
    for i in range(n):
        store.upsert_chunk(
            ChunkRecord(
                chunk_id=f"p1::c{i}",
                paper_id="p1",
                ordinal=i,
                heading=section,
                page=1,
                text=texts[i % len(texts)],
                tokens=10,
                meta={
                    "heading": section,
                    "section_id": "sec0",
                    "section_heading": section,
                    "is_first_in_section": i == 0,
                    "is_last_in_section": i == n - 1,
                },
            )
        )


# ── section detector ────────────────────────────────────────────────────────
def test_split_sections_detects_numbered_and_labels():
    from evidence.ingestion.chunker import split_sections

    text = (
        "Paper Title Line\n"
        "\n"
        "Introduction\n"
        "We introduce a novel approach.\n"
        "\n"
        "2. Method\n"
        "The method has three stages.\n"
        "\n"
        "3.1 Training\n"
        "Training uses Adam.\n"
        "\n"
        "RESULTS\n"
        "The results improve baselines.\n"
        "\n"
        "Conclusion\n"
        "We conclude the work is useful.\n"
    )
    secs = split_sections(text)
    headings = [s.heading for s in secs]
    assert "Introduction" in headings
    assert "Method" in headings
    assert "Training" in headings
    assert "RESULTS" in headings
    assert "Conclusion" in headings
    # No duplicate "Introduction" from preamble.
    assert headings.count("Introduction") == 1


def test_split_sections_fallback_no_markers():
    from evidence.ingestion.chunker import split_sections

    secs = split_sections("just words with no structure here at all")
    assert len(secs) == 1 and secs[0].heading == ""


def test_chunk_document_assigns_section_ids_and_flags():
    from evidence.config import ChunkingConfig
    from evidence.ingestion.chunker import chunk_document

    cfg = ChunkingConfig(max_tokens=40, overlap_tokens=2, min_chunk_tokens=4)
    text = (
        "Introduction\n"
        "Some intro text that is long enough to be one chunk of content.\n"
        "Method\n"
        "Some method text that is long enough to be a distinct chunk.\n"
        "Results\n"
        "Some results text that is long enough to be a distinct chunk too.\n"
    )
    chunks = chunk_document(text, cfg)
    assert chunks, "expected at least one chunk"
    # Every chunk carries a section_id and the first chunk of each section
    # is flagged.
    firsts = [c for c in chunks if c.is_first_in_section]
    assert firsts, "expected at least one first-in-section chunk"
    for c in chunks:
        assert c.section_id, f"chunk {c.chunk_id} missing section_id"
    # Distinct sections produce distinct ids.
    assert len({c.section_id for c in chunks}) >= 2


def _ctx(store):
    """Minimal AppContext-shaped object for facade functions (store only)."""
    return SimpleNamespace(store=store)


# ── facade: get_sections / get_section ──────────────────────────────────────
def test_get_sections_lists_sections(store):
    _seed_paper_and_chunks(store, n=4, section="Method")
    from evidence.facade import get_sections

    secs = get_sections("p1", app=_ctx(store))
    assert len(secs) == 1
    s = secs[0]
    assert s["heading"] == "Method"
    assert s["n_chunks"] == 4
    assert s["chunk_ids"] == ["p1::c0", "p1::c1", "p1::c2", "p1::c3"]
    assert s["first_chunk"] == "p1::c0"
    assert s["last_chunk"] == "p1::c3"


def test_get_section_stitches_untruncated(store):
    _seed_paper_and_chunks(store, n=4, section="Method")
    from evidence.facade import get_section

    sec = get_section("p1", "sec0", app=_ctx(store))
    assert sec is not None
    assert sec["heading"] == "Method"
    assert sec["n_chunks"] == 4
    # Text should be the 4 chunks joined with blank lines.
    assert sec["text"].count("\n\n") == 3
    # Resolving by heading substring also works.
    by_heading = get_section("p1", "Method", app=_ctx(store))
    assert by_heading is not None and by_heading["section_id"] == "sec0"


def test_get_section_missing_returns_none(store):
    _seed_paper_and_chunks(store, n=2)
    from evidence.facade import get_section

    assert get_section("p1", "does-not-exist", app=_ctx(store)) is None
    assert get_section("no-such-paper", "sec0", app=_ctx(store)) is None


# ── links: deterministic + LLM ──────────────────────────────────────────────
def test_build_chunk_links_deterministic_no_model(store, cfg):
    _seed_paper_and_chunks(store, n=4)
    from evidence.links import build_chunk_links

    # No model → only deterministic links should be emitted (continuation +
    # same-section see_also).
    report = build_chunk_links(store, models=None, cfg=cfg, paper_id="p1", reset=True)
    assert report.papers_processed == 1
    assert report.deterministic_links > 0
    assert report.llm_links == 0

    links = store.links_for_paper("p1")
    assert len(links) == report.links_emitted
    relations = {ln["relation"] for ln in links}
    assert "continuation" in relations
    assert "see_also" in relations


def test_build_chunk_links_with_llm(store, cfg, fake_ollama):
    _seed_paper_and_chunks(store, n=4)
    from evidence.links import build_chunk_links
    from evidence.models.ollama import Models

    models = Models(cfg)
    report = build_chunk_links(store, models=models, cfg=cfg, paper_id="p1", reset=True)
    assert report.papers_processed == 1
    assert report.llm_links >= 1
    assert fake_ollama.calls >= 1

    links = store.links_for_paper("p1")
    rels = {ln["relation"] for ln in links}
    assert "depends_on" in rels  # the LLM link we canned


def test_build_chunk_links_never_hard_fails_on_model_error(store, cfg, monkeypatch):
    _seed_paper_and_chunks(store, n=4)
    from evidence.links import build_chunk_links
    from evidence.models.ollama import Models

    class _BoomClient:
        def __init__(self, **_):
            pass

        def chat(self, **_):
            raise RuntimeError("model exploded")

    monkeypatch.setitem(
        sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: _BoomClient())
    )
    models = Models(cfg)
    # Should NOT raise; deterministic links still present.
    report = build_chunk_links(store, models=models, cfg=cfg, paper_id="p1", reset=True)
    assert report.deterministic_links > 0
    assert report.llm_links == 0


def test_links_persist_and_replay(store, cfg):
    _seed_paper_and_chunks(store, n=3)
    from evidence.links import build_chunk_links

    build_chunk_links(store, models=None, cfg=cfg, paper_id="p1", reset=True)
    first = store.links_for_paper("p1")
    assert first

    # A second build (reset=False) should not duplicate existing (src,dst,rel).
    build_chunk_links(store, models=None, cfg=cfg, paper_id="p1", reset=False)
    second = store.links_for_paper("p1")
    # (src, dst, relation) tuples are unique.
    keyset = {(ln["src_chunk_id"], ln["dst_chunk_id"], ln["relation"]) for ln in second}
    assert len(keyset) == len(second)


# ── linked_chunks / links_for_paper exposure ────────────────────────────────
def test_linked_chunks_out_in(store, cfg):
    _seed_paper_and_chunks(store, n=4)
    from evidence.links import build_chunk_links, linked_chunks

    build_chunk_links(store, models=None, cfg=cfg, paper_id="p1", reset=True)

    out = linked_chunks("p1::c1", store=store, direction="out")
    assert out["n_neighbours"] >= 1
    # At least one neighbour should be c2 (the continuation).
    ids = {n["chunk_id"] for n in out["neighbours"]}
    assert "p1::c2" in ids
    # Every neighbour carries a preview.
    for n in out["neighbours"]:
        assert "preview" in n


def test_linked_chunks_unknown_chunk(store, cfg):
    from evidence.links import linked_chunks

    res = linked_chunks("does-not-exist", store=store)
    assert res["paper_id"] is None
    assert res["neighbours"] == []


def test_links_for_paper_shape(store, cfg):
    _seed_paper_and_chunks(store, n=4)
    from evidence.links import build_chunk_links, links_for_paper

    build_chunk_links(store, models=None, cfg=cfg, paper_id="p1", reset=True)
    res = links_for_paper("p1", store=store)
    assert res["n_links"] >= 1
    assert "links" in res
    for ln in res["links"]:
        assert "src_preview" in ln
        assert "dst_preview" in ln
        assert ln["dst_paper_id"] == "p1"


def test_links_for_paper_missing(store):
    from evidence.links import links_for_paper

    res = links_for_paper("no-such-paper", store=store)
    assert res["n_links"] == 0
    assert res["links"] == []


# ── store-level sanity ──────────────────────────────────────────────────────
def test_store_chunk_links_roundtrip(store):
    store.upsert_paper(
        PaperRecord(
            paper_id="p1", title="x", path="x", source_format="pdf", meta={}
        )
    )
    store.upsert_chunk(
        ChunkRecord(
            chunk_id="p1::a", paper_id="p1", ordinal=0, heading="h", page=1,
            text="a", tokens=1, meta={},
        )
    )
    store.upsert_chunk(
        ChunkRecord(
            chunk_id="p1::b", paper_id="p1", ordinal=1, heading="h", page=1,
            text="b", tokens=1, meta={},
        )
    )
    store.upsert_chunk_link("p1::a", "p1::b", "continuation", 1.0, "adjacent")
    # Upserting the same (src,dst,relation) updates confidence/reason.
    store.upsert_chunk_link("p1::a", "p1::b", "continuation", 0.5, "updated")
    rows = store.links_for_chunk("p1::a", direction="out")
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.5
    assert rows[0]["reason"] == "updated"
    assert store.count_chunk_links() == 1
    store.clear_chunk_links("p1")
    assert store.count_chunk_links() == 0
