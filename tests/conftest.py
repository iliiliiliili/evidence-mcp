"""Shared pytest fixtures for Evidence tests.

These fixtures build a tiny in-memory-ish corpus on a temp Store and a
no-network `Models` facade, so tests run offline and deterministically.
"""

from __future__ import annotations

import pytest

from evidence.config import EvidenceConfig
from evidence.indexes.store import Store
from evidence.models.ollama import Models


@pytest.fixture
def cfg() -> EvidenceConfig:
    return EvidenceConfig()


@pytest.fixture
def models(cfg) -> Models:
    # Deterministic embedder + reranker, no Ollama calls for pure-helper tests.
    return Models(cfg)


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "store.sqlite")
    yield s
    s.close()


def _seed(store: Store) -> None:
    """Populate a small, known corpus for deterministic retrieval tests."""
    from evidence.indexes.store import ChunkRecord, PaperRecord

    p1 = PaperRecord(
        paper_id="p1", title="Attention-based Sequence Models", path="a.pdf",
        source_format="pdf", n_pages=2, year=2021, doi="10.1000/a",
        authors=["Alice Smith", "Bob Jones"], abstract="Transformers.", meta={},
    )
    p2 = PaperRecord(
        paper_id="p2", title="Graph Neural Networks in Practice", path="b.pdf",
        source_format="pdf", n_pages=1, year=2022, doi="10.1000/b",
        authors=["Carol Lee"], abstract="GNNs.", meta={},
    )
    store.upsert_paper(p1)
    store.upsert_paper(p2)
    store.upsert_chunk(ChunkRecord(
        chunk_id="p1::c1", paper_id="p1", ordinal=0, heading="Intro", page=1,
        text="The attention mechanism aligns query and key vectors to compute "
             "soft weighted combinations over the sequence of tokens.",
        tokens=20, meta={},
    ))
    store.upsert_chunk(ChunkRecord(
        chunk_id="p1::c2", paper_id="p1", ordinal=1, heading="Methods", page=2,
        text="We train the model with the Adam optimizer and a cosine schedule.",
        tokens=12, meta={},
    ))
    store.upsert_chunk(ChunkRecord(
        chunk_id="p2::c1", paper_id="p2", ordinal=0, heading="Abstract", page=1,
        text="Graph neural networks propagate messages along edges to aggregate "
             "neighbourhood information for node classification.",
        tokens=16, meta={},
    ))


@pytest.fixture
def seeded_store(store) -> Store:
    _seed(store)
    return store
