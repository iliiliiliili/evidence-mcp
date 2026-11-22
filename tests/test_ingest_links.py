"""Tests: `evidence ingest` builds the chunk-graph by default (after ingest),
respects --no-links, and shows the model used for the link step."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from evidence.cli import app
from evidence.config import EvidenceConfig
from evidence.indexes.store import Store

runner = CliRunner()


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
                    "reason": "derives objective from",
                }
            ]
        }
        return {"message": {"content": json.dumps(payload)}}

    def embed(self, **kwargs) -> dict:
        texts = kwargs.get("input", [])
        return {"embeddings": [[0.1] * 8 for _ in texts]}


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeOllamaClient()
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: fake))
    return fake


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A minimal 2-section text corpus for ingest."""
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "paper.txt").write_text(
        "# Introduction\n\n"
        "We propose a novel method for financial sentiment analysis.\n"
        "The approach leverages knowledge graphs to enhance retrieval.\n\n"
        "## Method\n\n"
        "Our method combines LLM embeddings with graph traversal.\n"
        "We train the system end-to-end with a contrastive loss.\n",
        encoding="utf-8",
    )
    return c


@pytest.fixture
def cfg(tmp_path: Path) -> EvidenceConfig:
    c = EvidenceConfig()
    c.data_dir = tmp_path / ".evidence"
    c.index_dir = tmp_path / "index"
    c.ollama.reasoning = "reasoning"
    c.ollama.embed = "reasoning"
    c.external.enabled = False
    return c


def _write_cfg_file(tmp_path: Path, cfg: EvidenceConfig, corpus: Path) -> str:
    """Write a config YAML pointing at this corpus and index dir."""
    import yaml

    payload = {
        "data_dir": str(tmp_path / ".evidence"),
        "index_dir": str(tmp_path / "index"),
        "corpus_dir": str(corpus),
        "ollama": {
            "reasoning": "reasoning",
            "light": "reasoning",
            "embed": "reasoning",
        },
        "external": {"enabled": False},
    }
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(f)


def test_ingest_builds_links_by_default(tmp_path, corpus, fake_ollama):
    """Default ingest runs the link pass and produces chunk links in the store."""
    cfg_file = _write_cfg_file(tmp_path, None, corpus)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["ingest", str(corpus), "--config", cfg_file])
    assert result.exit_code == 0, result.output

    # The chunk-graph step was shown.
    assert "chunk-graph" in result.output.lower() or "Building" in result.output
    # The model name is shown.
    assert "reasoning" in result.output

    # The store should now have chunk links.
    store = Store(tmp_path / "index" / "store.sqlite")
    try:
        assert store.count_chunk_links() > 0
    finally:
        store.close()


def test_ingest_no_links_skips_graph(tmp_path, corpus, fake_ollama):
    """--no-links skips the graph build; store has no chunk links."""
    cfg_file = _write_cfg_file(tmp_path, None, corpus)

    result = runner.invoke(
        app, ["ingest", str(corpus), "--config", cfg_file, "--no-links"]
    )
    assert result.exit_code == 0, result.output

    # The skip message is shown.
    assert "no-links" in result.output or "skipped" in result.output.lower()

    # No chunk links in the store.
    store = Store(tmp_path / "index" / "store.sqlite")
    try:
        assert store.count_chunk_links() == 0
    finally:
        store.close()


def test_ingest_link_reason_is_short_phrase(tmp_path, corpus, fake_ollama):
    """LLM link reasons are kept to a short phrase (<=60 chars)."""
    from evidence.indexes.store import ChunkRecord, PaperRecord
    from evidence.models.ollama import Models

    store = Store(tmp_path / "index" / "store.sqlite")
    store.upsert_paper(
        PaperRecord(
            paper_id="p1", title="T", path="t.txt", source_format="txt",
            n_pages=1, year=2024, doi=None, authors=[], abstract="", meta={},
        )
    )
    texts = ["Intro chunk one.", "Second chunk about method.", "Results section."]
    for i, t in enumerate(texts):
        store.upsert_chunk(
            ChunkRecord(
                chunk_id=f"p1::c{i}", paper_id="p1", ordinal=i,
                heading="Section", page=1, text=t, tokens=5, meta={},
            )
        )

    # Force a very long reason from the model; it should be truncated to 60.
    class _LongReason:
        def __init__(self, **_):
            pass

        def list_models(self):
            return {"models": [{"name": "reasoning"}]}

        def chat(self, **kwargs):
            reason = (
                "This is a very long reason that should be truncated to a "
                "short phrase by the code"
            )
            return {"message": {"content": json.dumps({
                "links": [{
                    "src_chunk_id": "p1::c0",
                    "dst_chunk_id": "p1::c1",
                    "relation": "depends_on",
                    "confidence": 0.7,
                    "reason": reason,
                }]
            })}}

        def embed(self, **kwargs):
            return {"embeddings": [[0.1] * 8]}

    import evidence.links as links_mod
    models = Models(EvidenceConfig())

    # Monkeypatch the client factory
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=lambda **kw: _LongReason()))
    models._ollama = None
    models._providers = {}

    links_mod.build_chunk_links(store, models=models, cfg=models.cfg, paper_id="p1", reset=True)
    links = store.links_for_paper("p1")
    llm_links = [lnk for lnk in links if lnk["relation"] == "depends_on"]
    assert len(llm_links) == 1
    assert len(llm_links[0]["reason"]) <= 60
    store.close()
