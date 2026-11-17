"""Tests for evidence grouping by paper (ledger render + markdown export).

Covers:
  - `_group_evidence` buckets refs by their parent paper (stable order).
  - `_evidence_paper_key` returns the paper title (chunks) or the DOI label.
  - `_evidence_section_label` returns just the section (no paper name).
  - `_markdown_evidence_group` renders a single-ref paper as one bullet and
    a multi-ref paper as a bold header + per-section bullets.
  - `Ledger.export_markdown(group_fn=...)` uses the custom renderer when
    given, and the flat fallback otherwise.
"""

from __future__ import annotations

import pytest

from evidence.cli import (
    _evidence_paper_key,
    _evidence_section_label,
    _group_evidence,
    _markdown_evidence_group,
)
from evidence.indexes.store import PaperRecord
from evidence.ledger import Ledger


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    lg = Ledger(tmp_path / "ledger.sqlite")
    yield lg
    lg.close()


# ── _group_evidence ─────────────────────────────────────────────────────────
def test_group_evidence_buckets_by_paper(seeded_store):
    doi_titles = {"10.1000/a": "Attention-based Sequence Models"}
    ev = [
        {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
        {"ref_type": "doi", "ref_key": "10.1000/b", "source": "openalex"},
        {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
    ]
    groups = _group_evidence(ev, seeded_store, doi_titles)
    assert len(groups) == 2
    # First-seen order: p1 paper first (from p1::c1), then the 10.1000/b DOI.
    assert groups[0][0] == "Attention-based Sequence Models"
    assert len(groups[0][1]) == 2
    assert groups[1][0] == "10.1000/b"
    assert len(groups[1][1]) == 1


def test_group_evidence_stable_order_with_unknown_paper(seeded_store):
    # Both chunks belong to a paper that is NOT in the store → one group
    # keyed by paper_id, regardless of chunk ordinal.
    from evidence.indexes.store import ChunkRecord

    seeded_store.upsert_chunk(ChunkRecord(
        chunk_id="zzz::c9", paper_id="zzz", ordinal=0, heading="H", page=1,
        text="x", tokens=1, meta={},
    ))
    seeded_store.upsert_chunk(ChunkRecord(
        chunk_id="zzz::c1", paper_id="zzz", ordinal=1, heading="K", page=2,
        text="y", tokens=1, meta={},
    ))

    doi_titles = {}
    ev = [
        {"ref_type": "chunk", "ref_key": "zzz::c9", "source": "local"},
        {"ref_type": "chunk", "ref_key": "zzz::c1", "source": "local"},
    ]
    groups = _group_evidence(ev, seeded_store, doi_titles)
    assert len(groups) == 1
    # The group key is the paper_id ("zzz"), not the full chunk key.
    assert groups[0][0] == "zzz"
    assert len(groups[0][1]) == 2


def test_group_evidence_empty(seeded_store):
    assert _group_evidence([], seeded_store, {}) == []


# ── _evidence_paper_key ─────────────────────────────────────────────────────
def test_paper_key_chunk_uses_paper_title(seeded_store):
    assert _evidence_paper_key(
        {"ref_type": "chunk", "ref_key": "p1::c1"},
        seeded_store, {},
    ) == "Attention-based Sequence Models"


def test_paper_key_doi_uses_title_or_doi(seeded_store):
    doi_titles = {"10.1000/a": "Attention-based Sequence Models"}
    # With a title in the map → title.
    assert _evidence_paper_key(
        {"ref_type": "doi", "ref_key": "10.1000/a"},
        seeded_store, doi_titles,
    ) == "Attention-based Sequence Models"
    # Unknown DOI → the bare DOI (display adds a "DOI: " prefix, but grouping
    # uses the stable identifier).
    assert _evidence_paper_key(
        {"ref_type": "doi", "ref_key": "10.9999/z"},
        seeded_store, doi_titles,
    ) == "10.9999/z"


def test_paper_key_unknown_chunk_falls_back_to_paper_id(seeded_store):
    # Chunk exists in the store but its paper does NOT → group by paper_id.
    from evidence.indexes.store import ChunkRecord

    seeded_store.upsert_chunk(ChunkRecord(
        chunk_id="ghost::c1", paper_id="ghost", ordinal=0, heading="H",
        page=None, text="x", tokens=1, meta={},
    ))
    assert _evidence_paper_key(
        {"ref_type": "chunk", "ref_key": "ghost::c1"},
        seeded_store, {},
    ) == "ghost"


# ── _evidence_section_label ─────────────────────────────────────────────────
def test_section_label_chunk_has_heading_and_page(seeded_store):
    # p1::c1 has heading "Intro" and page 1.
    assert _evidence_section_label(
        {"ref_type": "chunk", "ref_key": "p1::c1"}, seeded_store
    ) == "Intro (p.1)"


def test_section_label_doi_whole_document(seeded_store):
    assert _evidence_section_label(
        {"ref_type": "doi", "ref_key": "10.1000/a"}, seeded_store
    ) == "(whole document)"


def test_section_label_unknown_chunk_falls_back(seeded_store):
    assert _evidence_section_label(
        {"ref_type": "chunk", "ref_key": "nope::c9"}, seeded_store
    ) == "(local evidence)"


# ── _markdown_evidence_group ────────────────────────────────────────────────
def test_markdown_group_single_ref_is_one_bullet(seeded_store):
    doi_titles = {}
    lines = _markdown_evidence_group(
        [{"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"}],
        seeded_store, doi_titles,
    )
    assert len(lines) == 1
    assert lines[0].startswith("  - ")
    # The paper title + section are both present (single-ref collapse).
    assert "Attention-based Sequence Models" in lines[0]
    assert "Intro" in lines[0]


def test_markdown_group_multi_ref_is_compact_inline(seeded_store):
    """<=3 sections → single compact bullet with inline sections, plus the
    year + first author pulled from the paper record."""
    doi_titles = {}
    ev = [
        {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
        {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
    ]
    lines = _markdown_evidence_group(ev, seeded_store, doi_titles)
    # One line: bold header (title + year + author) + inline sections.
    assert len(lines) == 1
    assert lines[0].startswith("  - **")
    assert "Attention-based Sequence Models" in lines[0]
    assert "(2021)" in lines[0]
    assert "Alice Smith et al." in lines[0]
    assert "Intro (p.1)" in lines[0]
    assert "Methods (p.2)" in lines[0]


def test_markdown_group_mixed_papers(seeded_store):
    doi_titles = {"10.1000/b": "Graph Neural Networks in Practice"}
    ev = [
        {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
        {"ref_type": "doi", "ref_key": "10.1000/b", "source": "openalex"},
        {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
    ]
    lines = _markdown_evidence_group(ev, seeded_store, doi_titles)
    # p1 has 2 refs → one compact inline bullet; the DOI has 1 ref → single bullet.
    assert len(lines) == 2
    assert "Attention-based Sequence Models" in lines[0]
    assert "Intro (p.1)" in lines[0] and "Methods (p.2)" in lines[0]
    assert lines[1].startswith("  - ")
    assert "Graph Neural Networks in Practice" in lines[1]
    assert "10.1000/b" in lines[1]


def test_markdown_group_sorts_sections_by_document_order(seeded_store):
    """Sections render in document (ordinal) order, not attachment order."""
    doi_titles = {}
    # Attach Methods (ordinal 1) BEFORE Intro (ordinal 0).
    ev = [
        {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
        {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
    ]
    lines = _markdown_evidence_group(ev, seeded_store, doi_titles)
    assert len(lines) == 1
    # Intro must come before Methods in the inline list.
    assert lines[0].index("Intro (p.1)") < lines[0].index("Methods (p.2)")


def test_markdown_group_caps_long_sections(seeded_store):
    """More than 3 sections: header + bullets, capped at 5 with a 'N more' tail."""
    from evidence.indexes.store import ChunkRecord

    # Seed a paper with 7 sections.
    seeded_store.upsert_paper(
        PaperRecord(paper_id="long", title="Long Survey", path="long.pdf",
                    source_format="pdf", n_pages=7, year=2020, doi="10.1000/long",
                    authors=["Solo Author"], abstract=None, meta={})
    )
    for i in range(7):
        seeded_store.upsert_chunk(ChunkRecord(
            chunk_id=f"long::c{i}", paper_id="long", ordinal=i, heading=f"S{i+1}",
            page=i + 1, text="x", tokens=1, meta={}))

    doi_titles = {}
    ev = [
        {"ref_type": "chunk", "ref_key": f"long::c{i}", "source": "local"}
        for i in range(7)
    ]
    lines = _markdown_evidence_group(ev, seeded_store, doi_titles)
    # Header + 5 sections + one "…+2 more" tail = 7 lines.
    assert len(lines) == 7
    assert "Long Survey" in lines[0]
    assert "…+2 more" in lines[6]
    # The last 2 real sections (S6, S7) are NOT shown individually.
    assert "S6" not in "\n".join(lines)
    assert "S7" not in "\n".join(lines)


def test_markdown_group_merges_chunk_and_doi_for_same_paper(seeded_store):
    """A chunk group and a DOI group that share a paper merge into one."""
    doi_titles = {}
    # p1's own DOI (10.1000/a) as a DOI ref, plus two p1 chunks.
    ev = [
        {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
        {"ref_type": "doi", "ref_key": "10.1000/a", "source": "openalex"},
        {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
    ]
    lines = _markdown_evidence_group(ev, seeded_store, doi_titles)
    # Everything collapses into ONE group for the paper title (not a 2nd DOI line).
    assert len(lines) == 1
    assert "Attention-based Sequence Models" in lines[0]
    assert "10.1000/a" not in lines[0]  # the DOI is absorbed, not shown as its own group
    assert "(whole document)" in lines[0]  # the DOI ref contributes a section


# ── export_markdown(group_fn=...) ──────────────────────────────────────────
def test_export_markdown_uses_group_fn(ledger, seeded_store):
    from evidence.cli import _markdown_evidence_group

    doi_titles = {"10.1000/a": "Attention-based Sequence Models"}
    ledger.record(
        "pkg.train",
        "Use attention",
        evidence=[
            {"ref_type": "chunk", "ref_key": "p1::c1", "source": "local"},
            {"ref_type": "chunk", "ref_key": "p1::c2", "source": "local"},
        ],
    )

    def group_fn(ev):
        return _markdown_evidence_group(ev, seeded_store, doi_titles)

    md = ledger.export_markdown(group_fn=group_fn)
    # Grouped rendering: one compact line for the paper, with both sections inline.
    assert "**Attention-based Sequence Models" in md
    assert "Intro (p.1)" in md
    assert "Methods (p.2)" in md


def test_export_markdown_flat_when_no_group_fn(ledger):
    """Without a group_fn the legacy flat behaviour is preserved."""
    ledger.record(
        "pkg.train",
        "Use attention",
        evidence=[
            {"ref_type": "chunk", "ref_key": "p1::c1", "title": "Attention paper",
             "source": "local"},
            {"ref_type": "doi", "ref_key": "10.1000/b", "source": "openalex"},
        ],
    )
    md = ledger.export_markdown()  # no group_fn → fallback path
    assert "Attention paper" in md
    assert "10.1000/b" in md
    # No bold paper header — flat rendering.
    assert "**Attention paper**" not in md
