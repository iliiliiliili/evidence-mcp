"""Offline ingestion pipeline (PLAN.MD "Knowledge graph construction" steps 1–2, 6).

`ingest_corpus` walks a directory of PDFs / LaTeX sources, extracts text,
chunks it, persists to the Store, and (re)builds the FAISS + BM25 indexes.
Entity extraction / Neo4j (steps 3–5) are Phase 2 and live under `kg/`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import EvidenceConfig
from ..indexes.bm25 import Bm25Index
from ..indexes.faiss_text import FaissTextIndex
from ..indexes.store import ChunkRecord, PaperRecord, Store
from ..models.ollama import Models
from .chunker import chunk_document
from .html import extract_html
from .latex import as_pdf_doc, extract_latex
from .pdf import PdfDocument, doc_id_for, extract_pdf

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    papers: int = 0
    chunks: int = 0
    failed: list[str] = field(default_factory=list)


def _extract(path: Path) -> PdfDocument:
    """Dispatch by file extension; always returns a PdfDocument-shaped object."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".tex", ".ltx", ".latex"}:
        return as_pdf_doc(extract_latex(path))
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return PdfDocument(
            path=path, text=text, pages=[text], n_pages=1, title=path.stem
        )
    raise ValueError(f"unsupported document type: {path}")


def ingest_corpus(
    corpus_dir: str | Path,
    *,
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
) -> IngestStats:
    """Ingest every supported document under `corpus_dir` (recursive)."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")

    supported = {".pdf", ".tex", ".ltx", ".latex", ".html", ".htm",
                 ".txt", ".md", ".markdown"}
    files = sorted(
        p for p in corpus_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported
    )
    if not files:
        logger.warning("no ingesting files found under %s", corpus_dir)

    stats = IngestStats()
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []

    for path in files:
        try:
            doc = _extract(path)
        except Exception as exc:  # noqa: BLE001 - keep going on a bad file
            logger.warning("extract failed for %s: %s", path, exc)
            stats.failed.append(str(path))
            continue

        paper_id = doc_id_for(path)
        paper = PaperRecord(
            paper_id=paper_id,
            title=doc.title or path.stem,
            path=str(path),
            source_format=path.suffix.lower(),
            n_pages=doc.n_pages,
            meta=dict(doc.meta),
        )
        store.upsert_paper(paper)

        # Chunk per page if we have page splits, else the whole doc.
        chunks = []
        if doc.pages and len(doc.pages) > 1:
            for i, page_text in enumerate(doc.pages, start=1):
                if not page_text.strip():
                    continue
                page_chunks = chunk_document(
                    page_text, cfg.chunking, page=i
                )
                chunks.extend(page_chunks)
        else:
            chunks = chunk_document(doc.text, cfg.chunking)

        for idx, ch in enumerate(chunks):
            ch.paper_id = paper_id
            ch.chunk_id = f"{paper_id}#c{idx}"
            # Hierarchical span metadata: which section this chunk belongs to
            # and where it sits within that section (first/last flags let
            # `get_section` stitch a span back together without truncation).
            ch.meta = {
                "heading": ch.heading,
                "section_id": ch.section_id,
                "section_heading": ch.heading,
                "is_first_in_section": ch.is_first_in_section,
                "is_last_in_section": ch.is_last_in_section,
            }
            record = ChunkRecord(
                chunk_id=ch.chunk_id,
                paper_id=paper_id,
                ordinal=idx,
                heading=ch.heading,
                page=ch.page,
                text=ch.text,
                tokens=ch.tokens,
                meta=ch.meta,
            )
            store.upsert_chunk(record)
            chunk_ids.append(ch.chunk_id)
            chunk_texts.append(ch.text)
        stats.papers += 1
        stats.chunks += len(chunks)
        logger.info("ingested %s -> %d chunks", path.name, len(chunks))

    # Rebuild indexes over the FULL corpus (cheap at this scale; swap for
    # incremental later if the corpus grows large).
    _rebuild_indexes(cfg, models, store, chunk_ids, chunk_texts)

    logger.info(
        "ingest complete: %d papers, %d chunks, %d failed",
        stats.papers, stats.chunks, len(stats.failed),
    )
    return stats


def ingest_item(
    url_or_path: str,
    *,
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
) -> dict:
    """Add a SINGLE document (local path or http(s) URL) to the corpus.

    This is the backing function for the MCP `ingest_paper` tool. It downloads
    a URL to a temp file, then reuses the same extract → chunk → index path as
    `ingest_corpus`. Returns a small report dict.

    Supports: local .pdf / .tex / .txt paths, and PDF URLs (arXiv, publisher
    OA links). LaTeX-over-HTTP is not supported (needs the .tex file on disk).
    """
    url_or_path = url_or_path.strip()
    is_url = url_or_path.startswith(("http://", "https://"))

    if is_url:
        tmp = _download(url_or_path, cfg)
    else:
        tmp = Path(url_or_path).expanduser()
        if not tmp.exists():
            raise FileNotFoundError(f"file not found: {tmp}")

    try:
        return _ingest_single_file(tmp, cfg, models, store)
    finally:
        if is_url and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _download(url: str, cfg: EvidenceConfig) -> Path:
    import os
    import tempfile

    import httpx

    suffix = ".pdf" if ".pdf" in url.lower() else ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="evidence_ingest_")
    os.close(fd)
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
    return Path(path)


def _ingest_single_file(
    path: Path,
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
) -> dict:
    doc = _extract(path)
    paper_id = doc_id_for(path)
    paper = PaperRecord(
        paper_id=paper_id,
        title=doc.title or path.stem,
        path=str(path),
        source_format=path.suffix.lower() or "bin",
        n_pages=doc.n_pages,
        meta=dict(doc.meta),
    )
    store.upsert_paper(paper)

    chunks = []
    if doc.pages and len(doc.pages) > 1:
        for i, page_text in enumerate(doc.pages, start=1):
            if not page_text.strip():
                continue
            chunks.extend(chunk_document(page_text, cfg.chunking, page=i))
    else:
        chunks = chunk_document(doc.text, cfg.chunking)

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for idx, ch in enumerate(chunks):
        ch.paper_id = paper_id
        ch.chunk_id = f"{paper_id}#c{idx}"
        ch.meta = {"heading": ch.heading}
        store.upsert_chunk(
            ChunkRecord(
                chunk_id=ch.chunk_id,
                paper_id=paper_id,
                ordinal=idx,
                heading=ch.heading,
                page=ch.page,
                text=ch.text,
                tokens=ch.tokens,
                meta=ch.meta,
            )
        )
        chunk_ids.append(ch.chunk_id)
        chunk_texts.append(ch.text)

    _rebuild_indexes(cfg, models, store, chunk_ids, chunk_texts)

    return {
        "paper_id": paper_id,
        "title": paper.title,
        "chunks": len(chunks),
        "path": str(path),
    }


def _rebuild_indexes(
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> None:
    """(Re)build the FAISS dense + BM25 sparse indexes from the store."""
    if not chunk_ids:
        logger.warning("no chunks to index — skipping index build")
        return

    # Dense (FAISS). Embedding may be expensive; only build if a real embedder.
    try:
        dim = len(models.embed_one("dim probe"))
        vectors = models.embed(chunk_texts)
        idx = FaissTextIndex(cfg.resolved_index_dir, dim)
        idx.build(chunk_ids, vectors)
        store.kv_set("index_dim", str(dim))
    except Exception as exc:  # noqa: BLE001 - dense is optional
        logger.warning("FAISS build failed (%s); BM25-only retrieval will apply", exc)

    # Sparse (BM25) — always attempt; no model needed.
    try:
        bm = Bm25Index(cfg.resolved_index_dir)
        bm.build(chunk_ids, chunk_texts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 build failed: %s", exc)
