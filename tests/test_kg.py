from evidence.kg.authors import group_by_identity
from evidence.kg.citations import (
    CitationLink,
    co_citation_pairs,
    norm_doi,
    normalize_title,
)


def test_norm_doi_case_and_prefix():
    assert norm_doi("10.1000/ABC") == norm_doi("10.1000/abc")
    assert norm_doi("doi: 10.1000/xyz") == "10.1000/xyz"
    assert norm_doi("https://doi.org/10.1000/abc") == "10.1000/abc"


def test_normalize_title_lowercase_and_strip():
    a = normalize_title("  Deep Learning: A  Survey ")
    b = normalize_title("deep learning: a survey")
    assert a == b


def test_co_citation_pairs_counts_shared_targets():
    # a and b both cite X -> one co-citation pair.
    links = [
        CitationLink(source_paper_id="a", target_paper_id="X"),
        CitationLink(source_paper_id="b", target_paper_id="X"),
        CitationLink(source_paper_id="a", target_paper_id="Y"),
    ]
    pairs = co_citation_pairs(links)
    assert ("a", "b") in pairs
    assert pairs[("a", "b")] == 1


def test_group_by_identity_dedupes_case_variants():
    groups = group_by_identity(["Alice Smith", "alice smith", "Bob Jones"])
    assert len(groups) == 2
    assert {g.normalized for g in groups} == {"alice smith", "bob jones"}
