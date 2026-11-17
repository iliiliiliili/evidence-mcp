"""Tests for HTML ingestion (dependency-free text projection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence.ingestion.html import extract_html


def test_extract_html_text_and_headings(tmp_path: Path):
    p = tmp_path / "paper.html"
    p.write_text(
        "<html><head><title>VNN</title>"
        "<script>var x=1;</script><style>p{color:red}</style></head>"
        "<body><h1>Variational Neural Networks</h1>"
        "<p>VNNs are <b>variational</b> neural networks.</p>"
        "<ul><li>Bayesian</li></ul></body></html>",
        encoding="utf-8",
    )
    doc = extract_html(p)
    assert doc.title == "VNN"
    assert doc.n_pages == 1
    assert "# Variational Neural Networks" in doc.text
    assert "VNNs are variational neural networks." in doc.text
    assert "Bayesian" in doc.text
    # Noise stripped.
    assert "var x" not in doc.text
    assert "color:red" not in doc.text


def test_extract_html_uses_path_stem_when_no_title(tmp_path: Path):
    p = tmp_path / "notes.html"
    p.write_text("<html><body><p>hi</p></body></html>", encoding="utf-8")
    doc = extract_html(p)
    assert doc.title == "notes"
    assert "hi" in doc.text


def test_extract_html_pdf_viewer_wrapper_raises(tmp_path: Path):
    p = tmp_path / "arxiv.html"
    p.write_text(
        "<html><head><title>Paper | PDF</title></head>"
        "<body><iframe src='Paper_files/2207.01524v3.pdf'></iframe></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="PDF-viewer wrapper"):
        extract_html(p)


def test_extract_html_empty_raises(tmp_path: Path):
    p = tmp_path / "empty.html"
    p.write_text("<html><head><style>body{} </style></head><body></body></html>",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        extract_html(p)
