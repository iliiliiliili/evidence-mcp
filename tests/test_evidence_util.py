from evidence.graph.nodes.evidence_util import chunk_hash, dedupe_evidence, external_evidence


def test_chunk_hash_stable_and_short():
    h1 = chunk_hash("some text")
    h2 = chunk_hash("  some text  ")
    assert h1 == h2  # whitespace-normalized
    assert len(h1) == 16


def test_dedupe_keeps_highest_relevance():
    a = {"chunk_id": "x", "text": "body", "relevance": 0.3}
    b = {"chunk_id": "x", "text": "body", "relevance": 0.9}
    out = dedupe_evidence([a, b])
    assert len(out) == 1
    assert out[0]["relevance"] == 0.9


def test_dedupe_sorted_desc_by_relevance():
    items = [
        {"chunk_id": "a", "text": "one", "relevance": 0.2},
        {"chunk_id": "b", "text": "two", "relevance": 0.8},
        {"chunk_id": "c", "text": "three", "relevance": 0.5},
    ]
    out = dedupe_evidence(items)
    rels = [e["relevance"] for e in out]
    assert rels == sorted(rels, reverse=True)


def test_external_evidence_shape():
    ev = external_evidence(
        work_id="W1", title="T", abstract="A", source="openalex",
        relevance=0.7, url="http://u", doi="10.1/w", year=2021, authors=["Z"],
    )
    assert ev["chunk_id"].startswith("ext::openalex::")
    assert ev["verified"] is False
    assert ev["relevance"] == 0.7
    assert ev["authors"] == ["Z"]
