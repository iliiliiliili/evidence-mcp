"""Save external (DOI) references into the local corpus + lazy DOI resolution.

Two complementary behaviours, both controlled by
``cfg.external.save_online_to_corpus`` (default ``True``):

* **Save** (``save_online_ref``) — when grounding produces a DOI that we did
  not already have locally, upsert at least a PaperRecord with whatever
  metadata the provider gave us (title, year, authors, abstract, PDF URL). If
  ``save_online_to_corpus`` is on **and** an OA PDF URL is present, additionally
  download + chunk + index the full document so the reference resolves to a
  readable ``Paper · Section`` offline.

* **Resolve** (``resolve_doi_meta``) — for a DOI whose title we don't already
  have (e.g. a crossref-only hit, or a citation-only ref), fetch metadata on
  first access and cache it in ``store.kv`` so the same DOI is never re-fetched
  for this corpus.

Best-effort throughout: a network or parse failure never blocks grounding — the
lighter (title+section) record is kept and the failure is logged.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ..config import EvidenceConfig
from ..indexes.bm25 import Bm25Index
from ..indexes.faiss_text import FaissTextIndex
from ..indexes.store import ChunkRecord, PaperRecord, Store
from ..models.ollama import Models
from . import crossref, openalex

logger = logging.getLogger(__name__)

_DOI_PREFIX = "https://doi.org/"
_MAX_ONLINE_CHUNKS = 400


def _normalize_doi(doi: str | None) -> str | None:
    """Lowercase + strip a ``https://doi.org/`` prefix for use as a storage key.

    DOIs are case-sensitive for *display*, so this is only ever used as a key;
    the original casing is kept separately for the PaperRecord.
    """
    if not doi:
        return None
    d = doi.strip().lower()
    if d.startswith(_DOI_PREFIX):
        d = d[len(_DOI_PREFIX):]
    return d or None


def _paper_id_for_doi(doi_key: str) -> str:
    """Stable paper_id for a normalized DOI."""
    return f"online__doi__{doi_key}"


def _work_to_meta(w: dict[str, Any]) -> dict[str, Any]:
    """Normalize a provider work dict to the minimal record we persist.

    ``doi``     — original casing (for display / PaperRecord).
    ``doi_key`` — normalized (lower, no prefix), used as a storage key.
    """
    doi_raw = (w.get("doi") or "").strip()
    doi_key = _normalize_doi(doi_raw)
    authors = w.get("authors") or []
    if isinstance(authors, dict):
        authors = list(authors.values())
    return {
        "doi": doi_raw or None,
        "doi_key": doi_key,
        "title": (w.get("title") or w.get("paper") or "").strip(),
        "year": w.get("year"),
        "authors": list(authors) if authors else [],
        "abstract": (w.get("abstract") or "").strip(),
        "pdf_url": w.get("pdf_url") or w.get("oa_pdf") or w.get("oa_url") or None,
        "source": w.get("source") or "external",
    }


def _download(url: str) -> Path:
    """Download a URL to a temp file (``.pdf`` suffix when the URL suggests one)."""
    suffix = ".pdf" if ".pdf" in url.lower() else ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="evidence_online_")
    os.close(fd)
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
    return Path(path)


def _append_to_indexes(
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
    chunk_ids: list[str],
    chunk_texts: list[str],
) -> None:
    """Incrementally add new chunks to FAISS (embed only the new ones) and
    rebuild BM25 from the whole corpus (tokenization only, no re-embed).

    Far cheaper than the pipeline's full ``_rebuild_indexes``.
    """
    if not chunk_ids:
        return
    try:
        probe = models.embed_one("dim probe")
        dim = len(probe)
        vectors = models.embed(chunk_texts)
        idx = FaissTextIndex(cfg.resolved_index_dir, dim)
        idx.add(chunk_ids, vectors)
        store.kv_set("index_dim", str(dim))
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAISS append failed for online refs (%s); BM25-only", exc)
    try:
        all_chunks = store.all_chunks()
        if not all_chunks:
            return
        bm = Bm25Index(cfg.resolved_index_dir)
        bm.build([c.chunk_id for c in all_chunks], [c.text for c in all_chunks])
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 rebuild after online save failed: %s", exc)


def _save_full(
    meta: dict[str, Any],
    paper_id: str,
    store: Store,
    cfg: EvidenceConfig,
    models: Models,
) -> dict[str, Any]:
    """Download + extract + chunk + index the full document for one DOI."""
    doi = meta.get("doi") or ""
    doi_key = meta.get("doi_key") or ""
    tmp = _download(meta["pdf_url"])  # type: ignore[arg-type]
    try:
        from ..ingestion.chunker import chunk_document
        from ..ingestion.pdf import extract_pdf

        doc = extract_pdf(tmp)
        text = doc.text
        chunks = chunk_document(text, cfg.chunking)
        if len(chunks) > _MAX_ONLINE_CHUNKS:
            logger.warning(
                "capping online paper %s at %d chunks (was %d)",
                doi_key, _MAX_ONLINE_CHUNKS, len(chunks),
            )
            chunks = chunks[:_MAX_ONLINE_CHUNKS]
        abstract = (meta.get("abstract") or (text[:2000] if text else "")).strip()
        paper = PaperRecord(
            paper_id=paper_id,
            title=meta.get("title") or doc.title or doi or doi_key,
            path=meta.get("pdf_url") or "",
            source_format="online",
            n_pages=doc.n_pages,
            year=meta.get("year"),
            doi=doi or None,
            authors=meta.get("authors") or [],
            abstract=abstract or None,
            meta={"source": meta.get("source", "external"), "doi_key": doi_key},
        )
        store.upsert_paper(paper)
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        for i, ch in enumerate(chunks):
            ch.paper_id = paper_id
            ch.chunk_id = f"{paper_id}#c{i}"
            store.upsert_chunk(
                ChunkRecord(
                    chunk_id=ch.chunk_id,
                    paper_id=paper_id,
                    ordinal=i,
                    heading=ch.heading,
                    page=ch.page,
                    text=ch.text,
                    tokens=ch.tokens,
                    meta={"heading": ch.heading,
                          "source": meta.get("source", "external")},
                )
            )
            chunk_ids.append(ch.chunk_id)
            chunk_texts.append(ch.text)
        _append_to_indexes(cfg, models, store, chunk_ids, chunk_texts)
        return {
            "saved": True, "mode": "full", "paper_id": paper_id,
            "chunks": len(chunks), "error": None,
        }
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _save_meta_only(
    meta: dict[str, Any],
    paper_id: str,
    store: Store,
    cfg: EvidenceConfig,
    models: Models,
) -> dict[str, Any]:
    """Upsert a PaperRecord + one abstract (or metadata) chunk, no download."""
    doi = meta.get("doi")
    doi_key = meta.get("doi_key")
    abstract = (meta.get("abstract") or "").strip()
    title = meta.get("title") or (doi or doi_key or "unknown paper")
    paper = PaperRecord(
        paper_id=paper_id,
        title=title,
        path=meta.get("pdf_url") or "",
        source_format="online",
        n_pages=0,
        year=meta.get("year"),
        doi=doi,
        authors=meta.get("authors") or [],
        abstract=abstract or None,
        meta={"source": meta.get("source", "external"), "doi_key": doi_key},
    )
    store.upsert_paper(paper)

    chunk_text = (
        abstract
        or f"{title}. Authors: {', '.join(meta.get('authors') or []) or 'unknown'}."
        + (f" (DOI: {doi})" if doi else "")
    )
    chunks = 0
    if chunk_text:
        chunk_id = f"{paper_id}#c0"
        heading = "Abstract" if abstract else "Metadata"
        store.upsert_chunk(
            ChunkRecord(
                chunk_id=chunk_id,
                paper_id=paper_id,
                ordinal=0,
                heading=heading,
                page=None,
                text=chunk_text,
                tokens=max(1, len(chunk_text) // 4),
                meta={"heading": heading,
                      "source": meta.get("source", "external")},
            )
        )
        try:
            _append_to_indexes(cfg, models, store, [chunk_id], [chunk_text])
        except Exception as exc:  # noqa: BLE001
            logger.warning("index append for online ref %s failed: %s", doi_key, exc)
        chunks = 1

    return {
        "saved": True, "mode": "meta", "paper_id": paper_id,
        "chunks": chunks, "error": None,
    }


def save_online_ref(
    work: dict[str, Any],
    *,
    cfg: EvidenceConfig,
    models: Models,
    store: Store,
) -> dict[str, Any]:
    """Persist one external reference into the local corpus.

    * ``cfg.external.save_online_to_corpus`` on **and** the work has a usable
      PDF URL → download + extract + chunk + index the full document.
    * Otherwise → upsert at least a PaperRecord with the metadata we do have
      (title/authors/year/abstract/PDF URL). One abstract (or metadata)
      chunk is indexed so the reference is retrievable offline.

    Idempotent: a second call for the same key is a no-op (``mode="already"``).

    Returns ``{"saved": bool, "mode": "full"|"meta"|"already"|None,
    "paper_id": str|None, "chunks": int, "error": str|None}``.
    """
    meta = _work_to_meta(work)
    doi_key = meta["doi_key"]
    report: dict[str, Any] = {
        "saved": False, "mode": None, "paper_id": None, "chunks": 0, "error": None,
    }

    # No DOI — save under the provider id if we have one.
    if doi_key is None:
        pid = (work.get("id") or work.get("paper") or "").strip()
        if not pid:
            report["error"] = "no DOI or id in work — nothing to save"
            return report
        paper_id = f"online__id__{pid}"
        report["paper_id"] = paper_id
        if store.get_paper(paper_id) is not None:
            report.update({"saved": True, "mode": "already"})
            return report
        return _save_meta_only(meta, paper_id, store, cfg, models)

    paper_id = _paper_id_for_doi(doi_key)
    report["paper_id"] = paper_id
    if store.get_paper(paper_id) is not None:
        report.update({"saved": True, "mode": "already"})
        return report

    # Flag ON + PDF URL available → full document.
    if cfg.external.save_online_to_corpus and meta.get("pdf_url"):
        try:
            return _save_full(meta, paper_id, store, cfg, models)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "full-doc save for %s failed (%s); falling back to metadata-only",
                doi_key, exc,
            )
            report["error"] = f"full-doc save failed: {exc}"

    # Metadata-only path (flag off, no PDF URL, or download failed).
    return _save_meta_only(meta, paper_id, store, cfg, models)


# ── lazy DOI resolution ────────────────────────────────────────────────────

_KV_PREFIX = "doi_meta:"


def _kv_key(doi_key: str) -> str:
    return f"{_KV_PREFIX}{doi_key}"


def resolve_doi_meta(
    doi: str,
    *,
    cfg: EvidenceConfig,
    store: Store | None = None,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Fetch metadata for a DOI (Crossref → OpenAlex) on first access and
    cache it in ``store.kv`` so the same DOI is not re-fetched.

    Lookup order (first hit wins):
      1. ``store.kv`` cache (``doi_meta:<normalized-doi>``).
      2. Local corpus (``papers`` table — from a prior ``save_online_ref``).
      3. Network: Crossref → OpenAlex (only when ``allow_network=True``).

    Returns ``{"title": str|None, "year": int|None, "doi": str|None}``.
    """
    d = _normalize_doi(doi)
    if not d:
        return {"title": None, "year": None, "doi": None}
    out: dict[str, Any] = {"title": None, "year": None, "doi": d}

    # 1) Cache.
    if store is not None:
        cached_raw = store.kv_get(_kv_key(d))
        if cached_raw:
            try:
                cached = json.loads(cached_raw)
                out.update({"title": cached.get("title"),
                            "year": cached.get("year")})
                if out["title"]:
                    return out
            except json.JSONDecodeError:
                pass

    # 2) Local corpus.
    if store is not None:
        p = store.get_paper(_paper_id_for_doi(d))
        if p is not None:
            out.update({"title": p.title, "year": p.year})
            if out["title"]:
                return out

    # 3) Network.
    if not allow_network:
        return out
    try:
        w = crossref.get_work(d)
        if w:
            title = (w.get("title") or "").strip()
            year = w.get("year")
            if title or year:
                out.update({"title": title or None, "year": year})
                if store is not None:
                    try:
                        store.kv_set(_kv_key(d), json.dumps(out))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cache DOI meta %s failed: %s", d, exc)
                if title:
                    return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOI resolve via crossref failed for %s: %s", d, exc)
    try:
        w = openalex.get_work(f"doi:{d}")
        if w:
            title = (w.get("title") or "").strip()
            year = w.get("year")
            if title or year:
                out.update({"title": title or None, "year": year})
                if store is not None:
                    try:
                        store.kv_set(_kv_key(d), json.dumps(out))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cache DOI meta %s failed: %s", d, exc)
                if title:
                    return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOI resolve via openalex failed for %s: %s", d, exc)
    return out
