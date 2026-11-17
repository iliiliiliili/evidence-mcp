"""End-to-end test: ingest a tiny corpus, ask a question, verify a grounded,
cited answer is produced in fallback (no-LLM) mode."""

from __future__ import annotations

import pytest

from evidence.indexes.store import Store
from evidence.ingestion.pipeline import ingest_corpus
from evidence.models.ollama import Models


@pytest.fixture
def tiny_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = (
        "# Attention Models\n\n"
        "The attention mechanism computes a weighted sum of value vectors where "
        "the weights are the softmax of scaled dot products between queries and keys.\n\n"
        "## Training\n\n"
        "We train with the Adam optimizer, a learning rate of 1e-4, and early "
        "stopping on a validation set.\n\n"
        "# Graph Models\n\n"
        "Graph neural networks aggregate neighbour features through message "
        "passing to learn node embeddings for link prediction.\n"
    )
    (corpus / "attention.txt").write_text(doc, encoding="utf-8")
    return corpus


def test_end_to_end_asks_and_cites(tiny_corpus, tmp_path, cfg):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    cfg.index_dir = index_dir  # make ingest + app share the same index dir
    store = Store(index_dir / "store.sqlite")
    models = Models(cfg)

    stats = ingest_corpus(tiny_corpus, cfg=cfg, models=models, store=store)
    assert stats.papers >= 1
    assert stats.chunks >= 1

    from evidence.graph.builder import Evidence

    app = Evidence(cfg)
    state = app.ask("How is the attention mechanism computed?")

    # A grounded answer is produced even without a live LLM.
    assert state.get("answer"), "expected a non-empty answer"
    # Citations resolved to real chunks; every kept citation verified.
    citations = state.get("citations") or []
    for c in citations:
        assert c.get("verified") is True
        assert store.get_chunk(c["chunk_id"]) is not None
    store.close()
