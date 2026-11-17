"""Public, tool-facing facade over the Evidence pipeline (new plan "Server impl").

The MCP server wraps these thin, state-free functions rather than the raw graph
nodes. They thread a shared `NodeContext` (models + store + retriever + config)
so the existing nodes can be reused without running the whole LangGraph.

  - search_corpus(query)            — hybrid retrieval + rerank, citation-shaped
  - verify_claim(claim, chunk_id)   — rule-based claim↔chunk check
  - score_evidence(question, ids)   — 100-pt sufficiency on a given evidence set
  - get_paper(paper_id)             — full paper record + chunk outline
  - corpus_stats()                  — counts + last-updated
  - expand_via_graph(chunk_id, ...) — related chunks via the KG (Neo4j if avail)

Building the full `Evidence` graph is expensive (LangGraph compile). For the
read-only tools we only need the context, so we build a lightweight context
here and expose `full_pipeline()` for the heavy `answer_research` tool.
"""

from __future__ import annotations

from .config import EvidenceConfig
from .graph.nodes.citations import verify_claim as _verify_claim
from .graph.nodes.context import NodeContext, set_context
from .graph.nodes.evidence_util import evidence_from_chunk
from .graph.nodes.sufficiency import score_evidence as _score_evidence
from .indexes.retriever import Retriever
from .indexes.store import Store
from .models.ollama import Models


class AppContext:
    """Shared services (no LangGraph). Build once per server process."""

    def __init__(self, cfg: EvidenceConfig | None = None):
        self.cfg = cfg or EvidenceConfig.load()
        index_dir = self.cfg.resolved_index_dir
        index_dir.mkdir(parents=True, exist_ok=True)

        self.models = Models(self.cfg)
        self.store = Store(self.cfg.resolved_store_path)
        self.retriever = Retriever(
            index_dir, self.store, self.models, self.cfg.retrieval
        )
        dim = self.store.kv_get("index_dim")
        self.retriever.dim = int(dim) if dim else 1024

        self.ctx = NodeContext(
            cfg=self.cfg,
            models=self.models,
            store=self.store,
            retriever=self.retriever,
        )
        set_context(self.ctx)

    def close(self) -> None:
        self.store.close()


# Module-level singleton (set on first use), so tools can call functions without
# threading the context around.
_APP: AppContext | None = None


def get_app(cfg: EvidenceConfig | None = None) -> AppContext:
    global _APP
    if _APP is None or (cfg is not None and _APP.cfg is not cfg):
        if _APP is not None:
            _APP.close()
        _APP = AppContext(cfg)
    return _APP


def set_app(app: AppContext | None) -> None:
    global _APP
    _APP = app


# ── search_corpus ─────────────────────────────────────────────────────────
def retrieval_mode(app: AppContext | None = None) -> str:
    """Report which retrieval backends are actually loaded.

    Returns "hybrid" (dense FAISS + BM25), "bm25_only" (sparse only — no
    embedding model / dense index available), or "unavailable" (nothing
    indexed). Lets a client explain *why* results are the way they are, e.g.
    when running without an Ollama embedding model.
    """
    app = app or get_app()
    # Load-independent (checks index files on disk + config), so it is valid
    # before `hybrid()` has loaded the in-memory handles.
    dense, sparse = app.retriever.backends_available()
    if dense and sparse:
        return "hybrid"
    if sparse:
        return "bm25_only"
    if dense:
        return "dense_only"
    return "unavailable"


def search_corpus(query: str, *, limit: int = 8, app: AppContext | None = None) -> list[dict]:
    """Hybrid search (dense + sparse, RRF) + cross-encoder rerank.

    Returns citation-shaped chunks (title, year, section, chunk_id, score).
    Each result carries a `retrieval_mode` field ("hybrid" | "bm25_only" |
    "dense_only" | "unavailable") so a client knows how the result was
    produced — e.g. bm25_only when no embedding model is available.
    This is the workhorse retrieval tool.
    """
    app = app or get_app()
    cfg = app.cfg
    mode = retrieval_mode(app)
    top_k = max(limit * 3, cfg.retrieval.candidates // 2)
    result = app.retriever.hybrid([query], top_k=top_k)
    ranked = result.ranked[: cfg.retrieval.candidates]
    if not ranked:
        return []

    # Rerank the candidate set, then keep `limit`.
    passages = []
    ids: list[str] = []
    for cid, _score in ranked:
        text = app.retriever.text_of(cid) or ""
        passages.append(text)
        ids.append(cid)
    scores = app.models.rerank(query, passages)
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:limit]

    out: list[dict] = []
    for i in order:
        cid = ids[i]
        rec = app.store.get_chunk(cid)
        if rec is None:
            continue
        paper = app.store.get_paper(rec.paper_id)
        out.append(
            {
                "chunk_id": cid,
                "text": _truncate(rec.text, cfg.mcp.max_chunk_chars),
                "paper": paper.title if paper else "",
                "year": paper.year if paper else None,
                "section": rec.heading,
                "page": rec.page,
                "doi": paper.doi if paper else None,
                "relevance": round(float(scores[i]), 3),
                "source": "local",
                "retrieval_mode": mode,
            }
        )
    return out


# ── expand_via_graph ──────────────────────────────────────────────────────
def expand_via_graph(
    chunk_id: str,
    *,
    relation: str = "SHARES_TOPIC",
    limit: int = 5,
    app: AppContext | None = None,
) -> list[dict]:
    """Related chunks from OTHER papers via the knowledge graph.

    Falls back to a BM25/text-nearest expansion over the corpus when Neo4j is
    not available, so the tool is useful even on the default (graph-disabled)
    install. `relation` is SHARES_TOPIC | SHARES_METHOD | CITED_BY | CITES.
    """
    app = app or get_app()
    rec = app.store.get_chunk(chunk_id)
    if rec is None:
        return []

    # Try the real KG traversal (Neo4j) first.
    rows = _neo4j_related(app, rec.paper_id, relation, limit)
    if rows:
        return rows

    # Fallback: related chunks by text similarity to this chunk's paper,
    # excluding the same paper. Cheap and offline.
    seed_text = app.retriever.text_of(chunk_id) or ""
    if not seed_text:
        return []
    result = app.retriever.hybrid([seed_text], top_k=limit * 3)
    out: list[dict] = []
    for cid, _ in result.ranked:
        if len(out) >= limit:
            break
        ch = app.store.get_chunk(cid)
        if ch is None or ch.paper_id == rec.paper_id:
            continue
        p = app.store.get_paper(ch.paper_id)
        out.append(
            {
                "chunk_id": cid,
                "text": _truncate(ch.text, app.cfg.mcp.max_chunk_chars),
                "paper": p.title if p else "",
                "year": p.year if p else None,
                "section": ch.heading,
                "page": ch.page,
                "doi": p.doi if p else None,
                "relation": relation,
                "source": "graph-fallback",
            }
        )
    return out


def _neo4j_related(app, paper_id: str, relation: str, limit: int) -> list[dict]:
    """Traverse the KG (Neo4j) for related papers. Returns [] if unavailable."""
    if not app.cfg.graph_expansion.enabled:
        return []
    try:
        import neo4j  # type: ignore
    except ImportError:
        return []

    out: list[dict] = []
    driver = None
    try:
        driver = neo4j.GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "evidence")
        )
        with driver.session() as s:
            res = s.run(
                """
                MATCH (p:Paper {paper_id:$pid})
                  -[r:ON_TOPIC|USES_METHOD]-(shared:Topic|Method)
                  -[r2:ON_TOPIC|USES_METHOD]-(rel:Paper)
                WHERE rel.paper_id <> p.paper_id
                RETURN DISTINCT rel.paper_id AS pid, shared.name AS shared
                LIMIT $lim
                """,
                pid=paper_id, lim=limit,
            )
            pids = [rec["pid"] for rec in res]
        for pid in pids:
            if len(out) >= limit:
                break
            for ch in app.store.chunks_for_paper(pid)[:2]:
                if len(out) >= limit:
                    break
                p = app.store.get_paper(pid)
                out.append(
                    {
                        "chunk_id": ch.chunk_id,
                        "text": _truncate(ch.text, app.cfg.mcp.max_chunk_chars),
                        "paper": p.title if p else "",
                        "year": p.year if p else None,
                        "section": ch.heading,
                        "doi": p.doi if p else None,
                        "relation": relation,
                        "source": "graph",
                    }
                )
        return out
    except Exception:  # noqa: BLE001
        return []
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass


# ── get_paper / corpus_stats ──────────────────────────────────────────────
def get_paper(paper_id: str, *, app: AppContext | None = None) -> dict | None:
    app = app or get_app()
    p = app.store.get_paper(paper_id)
    if p is None:
        return None
    chunks = app.store.chunks_for_paper(paper_id)
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "year": p.year,
        "doi": p.doi,
        "authors": p.authors,
        "abstract": p.abstract,
        "n_pages": p.n_pages,
        "source_format": p.source_format,
        "path": p.path,
        "outline": [
            {"ordinal": c.ordinal, "section": c.heading, "page": c.page,
             "chunk_id": c.chunk_id, "tokens": c.tokens}
            for c in chunks
        ],
        "n_chunks": len(chunks),
    }


def corpus_stats(app: AppContext | None = None) -> dict:
    app = app or get_app()
    index_dir = app.cfg.resolved_index_dir
    return {
        "papers": app.store.count_papers(),
        "chunks": app.store.count_chunks(),
        "index_dir": str(index_dir),
        "faiss_present": (index_dir / "chunks.faiss").exists(),
        "bm25_present": (index_dir / "bm25.pkl").exists(),
        "embedding_dim": app.store.kv_get("index_dim"),
    }


# ── thin re-exports of the public node facades ───────────────────────────
def verify_claim(claim: str, chunk_id: str, *, app: AppContext | None = None) -> dict:
    app = app or get_app()
    set_context(app.ctx)
    return _verify_claim(claim, chunk_id)


def score_evidence(
    question: str,
    chunk_ids: list[str],
    *,
    intent: str = "content",
    app: AppContext | None = None,
) -> dict:
    app = app or get_app()
    set_context(app.ctx)
    evidence = []
    for cid in chunk_ids:
        ev = evidence_from_chunk(app.store, cid, source="rrf", relevance=0.7)
        if ev is not None:
            evidence.append(ev)
    out = _score_evidence(question, evidence, cfg=app.cfg, intent=intent)
    out["n_evidence"] = len(evidence)
    return out


# ── helpers ───────────────────────────────────────────────────────────────
def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"
