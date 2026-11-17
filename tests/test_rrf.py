from evidence.indexes.retriever import reciprocal_rank_fusion


def _ids(fused):
    return [cid for cid, _ in fused]


def test_single_ranking_preserves_order():
    r = reciprocal_rank_fusion([["a", "b", "c"]])
    assert _ids(r) == ["a", "b", "c"]


def test_fusion_prefers_items_ranked_high_in_both():
    r = reciprocal_rank_fusion([["a", "x", "y"], ["a", "w", "z"]])
    assert _ids(r)[0] == "a"


def test_larger_k_reduces_rank_penalty():
    r60 = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    r10 = dict(reciprocal_rank_fusion([["a", "b"]], k=10))
    assert (r60["a"] - r60["b"]) < (r10["a"] - r10["b"])


def test_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
