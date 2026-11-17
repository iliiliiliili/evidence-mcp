from evidence.config import EvidenceConfig
from evidence.graph.nodes import sufficiency as S


def _cfg():
    return EvidenceConfig()


def _ev(rel, chars, year=2025, paper="p1"):
    return {
        "paper_id": paper, "chunk_id": paper + "::c", "relevance": rel,
        "text": "x" * chars, "year": year, "verified": True,
    }


def test_no_evidence_zero_coverage():
    score, _ = S._score_coverage([], "q", _cfg())
    assert score == 0.0


def test_coverage_scales_with_papers_and_items():
    cfg = _cfg()
    one = S._score_coverage([_ev(0.9, 100, paper="a"), _ev(0.9, 100, paper="a")], "q", cfg)[0]
    two = S._score_coverage([_ev(0.9, 100, paper="a"), _ev(0.9, 100, paper="b")], "q", cfg)[0]
    assert two >= one  # more papers ⇒ at least as high


def test_relevance_capped_at_max():
    cfg = _cfg()
    score, _ = S._score_relevance([_ev(1.0, 100)], cfg)
    assert score <= 20.0


def test_relevance_damps_weak_chunks():
    cfg = _cfg()
    strong = S._score_relevance([_ev(0.9, 100)], cfg)[0]
    weak = S._score_relevance([_ev(0.1, 100)], cfg)[0]
    assert strong > weak


def test_depth_scales_with_text_length():
    cfg = _cfg()
    short = S._score_depth([_ev(0.9, 100)], cfg)[0]
    long = S._score_depth([_ev(0.9, 20000)], cfg)[0]
    assert long >= short


def test_recency_content_intent_not_gated():
    score, _ = S._score_recency([_ev(0.9, 100, year=2015)], "content")
    assert score >= 10.0


def test_relevance_damp_function():
    floor, factor = 0.35, 0.5
    assert S._relevance_damp({"relevance": 0.9}, floor, factor) == 0.9
    assert S._relevance_damp({"relevance": 0.2}, floor, factor) == 0.1
