"""On-disk corpus + chunk store (SQLite).

Holds the canonical metadata for every paper and chunk. The vector stores
(FAISS/BM25) are separate artefacts in the same `index_dir`; this store is the
source of truth for provenance (paper id, title, page, chunk text, DOI…) and
what the citation-integrity stage verifies against (PLAN.MD stage 11).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    path: str
    source_format: str  # "pdf" | "latex" | "json"
    n_pages: int = 0
    year: int | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    paper_id: str
    ordinal: int  # ordinal within the paper
    heading: str
    page: int | None
    text: str
    tokens: int
    meta: dict = field(default_factory=dict)


class Store:
    """A small SQLite-backed store with a friendly query surface."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                paper_id      TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                path          TEXT NOT NULL,
                source_format TEXT NOT NULL,
                n_pages       INTEGER DEFAULT 0,
                year          INTEGER,
                doi           TEXT,
                authors       TEXT,            -- JSON list
                abstract      TEXT,
                meta          TEXT             -- JSON dict
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id  TEXT PRIMARY KEY,
                paper_id  TEXT NOT NULL,
                ordinal   INTEGER NOT NULL,
                heading   TEXT,
                page      INTEGER,
                text      TEXT NOT NULL,
                tokens    INTEGER,
                meta      TEXT,                -- JSON dict
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    # ── papers ────────────────────────────────────────────────────────────
    def upsert_paper(self, p: PaperRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO papers (paper_id, title, path, source_format, n_pages,
                                year, doi, authors, abstract, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                title=excluded.title, path=excluded.path,
                source_format=excluded.source_format,
                n_pages=excluded.n_pages, year=excluded.year, doi=excluded.doi,
                authors=excluded.authors, abstract=excluded.abstract,
                meta=excluded.meta
            """,
            (
                p.paper_id,
                p.title,
                p.path,
                p.source_format,
                p.n_pages,
                p.year,
                p.doi,
                json.dumps(p.authors),
                p.abstract,
                json.dumps(p.meta),
            ),
        )
        self._conn.commit()

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE paper_id=?", (paper_id,)
        ).fetchone()
        return _row_to_paper(row) if row else None

    def list_papers(self) -> list[PaperRecord]:
        rows = self._conn.execute("SELECT * FROM papers ORDER BY title").fetchall()
        return [_row_to_paper(r) for r in rows]

    # ── chunks ────────────────────────────────────────────────────────────
    def upsert_chunk(self, c: ChunkRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO chunks (chunk_id, paper_id, ordinal, heading, page, text,
                                tokens, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                ordinal=excluded.ordinal, heading=excluded.heading,
                page=excluded.page, text=excluded.text, tokens=excluded.tokens,
                meta=excluded.meta
            """,
            (
                c.chunk_id,
                c.paper_id,
                c.ordinal,
                c.heading,
                c.page,
                c.text,
                c.tokens,
                json.dumps(c.meta),
            ),
        )
        self._conn.commit()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        for c in chunks:
            self.upsert_chunk(c)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def chunks_for_paper(self, paper_id: str) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE paper_id=? ORDER BY ordinal", (paper_id,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def all_chunk_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT chunk_id FROM chunks").fetchall()
        return [r["chunk_id"] for r in rows]

    def all_chunks(self) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks ORDER BY paper_id, ordinal"
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    def count_papers(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]

    # ── kv ────────────────────────────────────────────────────────────────
    def kv_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def close(self) -> None:
        self._conn.close()


def _row_to_paper(row: sqlite3.Row) -> PaperRecord:
    d = dict(row)
    d["authors"] = json.loads(d["authors"] or "[]")
    d["meta"] = json.loads(d["meta"] or "{}")
    return PaperRecord(**d)


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    d = dict(row)
    d["meta"] = json.loads(d["meta"] or "{}")
    return ChunkRecord(**d)
