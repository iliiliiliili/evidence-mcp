"""Ingestion package: PDF/LaTeX extraction, chunking, page rendering."""

from .chunker import Chunk, chunk_document, split_sections
from .latex import LatexDocument, as_pdf_doc, extract_latex
from .pdf import PdfDocument, doc_id_for, extract_pdf
from .pipeline import IngestStats, ingest_corpus, ingest_item
from .render_pages import pages_for_corpus, render_pages

__all__ = [
    "Chunk",
    "chunk_document",
    "split_sections",
    "PdfDocument",
    "extract_pdf",
    "doc_id_for",
    "LatexDocument",
    "extract_latex",
    "as_pdf_doc",
    "render_pages",
    "pages_for_corpus",
    "IngestStats",
    "ingest_corpus",
    "ingest_item",
]
