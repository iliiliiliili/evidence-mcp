"""Tests for the paper-title derivation heuristic (`evidence_util.derive_title_from_text`).

The heuristic must handle:
  - all-caps (IEEE, etc.) and title-case titles,
  - titles with commas (a title with commas must survive; an author list with
    commas must NOT),
  - single-author papers (2-word name lines as the author, not part of the
    title),
  - watermarks, journal headers, section labels, volume/issue lines,
  - middle initials in author names,
  - short (2-word) titles,
  - all-caps title continuations that are 2 words (e.g. "FLOW FIELDS").

These are pure-text tests — no store, no model.
"""

from __future__ import annotations

from evidence.graph.nodes.evidence_util import (
    _is_weak_title_stem,
    derive_title_from_text,
)


# ── _is_weak_title_stem ───────────────────────────────────────────────────
def test_weak_title_stem_fires_on_filename_like_titles():
    assert _is_weak_title_stem("") is True
    assert _is_weak_title_stem("11299") is True
    assert _is_weak_title_stem("a1b2c3") is True
    assert _is_weak_title_stem("abc123") is True


def test_weak_title_stem_fires_on_bare_ids():
    assert _is_weak_title_stem("12345") is True
    assert _is_weak_title_stem("paper01") is True


def test_weak_title_stem_not_fired_on_real_titles():
    # A title with spaces and at least one non-alphanumeric character (here,
    # a hyphen) is not "weak".
    assert _is_weak_title_stem("FinLLM-KG: Knowledge Graph-Enhanced LLMs") is False
    assert _is_weak_title_stem("Attention Mechanisms") is False


# ── derive_title_from_text: all-caps (current corpus shape) ──────────────
def test_all_caps_title_with_watermark_prefix():
    text = (
        "FOR REVIEW ONLY\n"
        "FOR REVIEW ONLY\n"
        "FINLLM-KG: KNOWLEDGE GRAPH-ENHANCED LLMS FOR FINANCIAL SENTIMENT\n"
        "ANALYSIS IN ALGORITHMIC TRADING\n"
        "Giorgos Iacovides, Oliver Hannibal, Wuyang Zhou, Danilo Mandic\n"
        "Department of Electrical and Electronic Engineering, Imperial College London, UK\n"
        "ABSTRACT\n"
        "Recent advances in Generative AI have substantially improved fi-"
    )
    t = derive_title_from_text(text)
    assert "FINLLM-KG" in t
    assert "KNOWLEDGE GRAPH-ENHANCED LLMS" in t
    assert "ANALYSIS IN ALGORITHMIC TRADING" in t
    # Must NOT include the author names or the affiliation.
    assert "Giorgos Iacovides" not in t
    assert "Imperial College" not in t
    # Must NOT include the body text.
    assert "Recent advances" not in t


def test_all_caps_two_line_title_with_2_word_continuation():
    text = (
        "FOR REVIEW ONLY\n"
        "PHYSALIGN: COORDINATING DATA AND PHYSICS IN LOW-RANK ADAPTATION OF\n"
        "FLOW FIELDS\n"
        "Zining Wang1, Enyuan Zhao1, Xiaoyu Wang1\n"
        "1 Hangzhou Institute for Advanced Study,\n"
        "University of Chinese Academy of Sciences, Hangzhou, China\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert "PHYSALIGN" in t
    assert "LOW-RANK ADAPTATION OF FLOW FIELDS" in t
    assert "Zining Wang" not in t
    assert "Hangzhou" not in t


# ── derive_title_from_text: title-case (the user's concern) ──────────────
def test_title_case_title_with_commas():
    text = (
        "Object Detection, Tracking, and Re-Identification in Dynamic Scenes\n"
        "Alice Smith, Bob Jones\n"
        "Stanford University\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Object Detection, Tracking, and Re-Identification in Dynamic Scenes"


def test_title_case_single_author_two_word_name():
    text = (
        "Attention Mechanisms for Vision Transformers\n"
        "John Doe\n"
        "MIT\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Attention Mechanisms for Vision Transformers"


def test_title_case_with_journal_header():
    text = (
        "IEEE TRANSACTIONS ON PATTERN ANALYSIS\n"
        "Attention Mechanisms for Vision Transformers\n"
        "John Doe, Jane Smith\n"
        "MIT\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Attention Mechanisms for Vision Transformers"


def test_title_case_with_volume_issue_line():
    text = (
        "Vol. 12, Issue 3, 2024\n"
        "Object Detection in Dynamic Scenes\n"
        "Alice Smith, Bob Jones\n"
        "Stanford\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert "Vol." not in t
    assert "Issue 3" not in t
    assert t == "Object Detection in Dynamic Scenes"


def test_title_case_with_middle_initials_in_authors():
    text = (
        "Attention: A Survey of Methods\n"
        "Alice A. Smith, Bob J. Jones\n"
        "Stanford University\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Attention: A Survey of Methods"
    assert "Alice A. Smith" not in t


def test_two_word_title_case_title():
    text = (
        "Object Detection\n"
        "Alice Smith, Bob Jones\n"
        "Stanford University\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Object Detection"


def test_no_author_line_title_case():
    text = (
        "Federated Learning with Heterogeneous Data\n"
        "Alice Smith, Bob Jones\n"
        "ABSTRACT\n"
        "..."
    )
    t = derive_title_from_text(text)
    assert t == "Federated Learning with Heterogeneous Data"


def test_empty_or_degenerate_text_returns_empty_string():
    assert derive_title_from_text("") == ""
    assert derive_title_from_text("   \n  \n ") == ""
    assert derive_title_from_text("ABSTRACT") == ""  # only a section label
    assert derive_title_from_text("for review only") == ""
