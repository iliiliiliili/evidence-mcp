from evidence.indexes.store import ChunkRecord, PaperRecord


def test_paper_roundtrip(store):
    p = PaperRecord(paper_id="p", title="T", path="x.pdf", source_format="pdf",
                    year=2020, doi="10.1/x", authors=["A"], abstract="abs", meta={"k": 1})
    store.upsert_paper(p)
    got = store.get_paper("p")
    assert got.title == "T"
    assert got.authors == ["A"]
    assert got.meta == {"k": 1}


def test_chunk_roundtrip_and_ordinal(store):
    store.upsert_paper(PaperRecord(paper_id="p", title="T", path="x.pdf", source_format="pdf"))
    c = ChunkRecord(chunk_id="p::c1", paper_id="p", ordinal=3, heading="H",
                    page=2, text="hello world", tokens=2, meta={})
    store.upsert_chunk(c)
    got = store.get_chunk("p::c1")
    assert got.ordinal == 3
    assert got.text == "hello world"


def test_paper_listing_and_counts(store):
    store.upsert_paper(PaperRecord(paper_id="p1", title="A", path="a.pdf", source_format="pdf"))
    store.upsert_paper(PaperRecord(paper_id="p2", title="B", path="b.pdf", source_format="pdf"))
    store.upsert_chunk(ChunkRecord(chunk_id="p1::c", paper_id="p1", ordinal=0,
                                   heading="", page=1, text="x", tokens=1))
    assert store.count_papers() == 2
    assert store.count_chunks() == 1
    assert {p.paper_id for p in store.list_papers()} == {"p1", "p2"}
    assert store.all_chunk_ids() == ["p1::c"]


def test_kv_roundtrip(store):
    store.kv_set("index_dim", "256")
    assert store.kv_get("index_dim") == "256"
    assert store.kv_get("missing") is None
