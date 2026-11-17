"""Hybrid retriever: FAISS dense + BM25 sparse fused with Reciprocal Rank Fusion.

This is the single object the `retrieve_hybrid` node talks to (PLAN.MD stage 3).
It owns loading the two sub-indexes and exposing a one-call `hybrid()` that
returns ranked `(chunk_id, score)` pairs.

Reciprocal Rank Fusion (Cormack et al., 2009):
    score(d) = Σ  1 / (k + rank_i(d))     over each ranking i
with `k` the smoothing constant (default 60).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RetrievalConfig
from ..models.ollama import Models
from .bm25 import Bm25Index
from .faiss_text import FaissTextIndex
from .store import Store

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    ranked: list[tuple[str, float]] = field(default_factory=list)  # (chunk_id, fused score)
    dense_ranked: list[str] = field(default_factory=list)
    sparse_ranked: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked id-lists via RRF. Higher fused score = better."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(
        self,
        index_dir: str | Path,
        store: Store,
        models: Models,
        cfg: RetrievalConfig,
        dim: int = 1024,
    ):
        self.index_dir = Path(index_dir)
        self.store = store
        self.models = models
        self.cfg = cfg
        self.dim = dim
        # Discover the real dim from a prior build if present.
        stored_dim = FaissTextIndex.dim_from_dir(self.index_dir)
        if stored_dim:
            self.dim = stored_dim
        self.faiss: FaissTextIndex | None = None
        self.bm25: Bm25Index | None = None

    # ── availability ──────────────────────────────────────────────────────
    def _faiss_available(self) -> bool:
        return (self.index_dir / "chunks.faiss").exists() and (
            self.index_dir / "chunk_ids.json"
        ).exists()

    def _bm25_available(self) -> bool:
        return (self.index_dir / "bm25.pkl").exists() and (
            self.index_dir / "bm25_ids.json"
        ).exists()

    def backends_available(self) -> tuple[bool, bool]:
        """(dense_available, sparse_available) based on config + index files.

        Load-independent (checks the index files on disk), so callers can use
        it before `hybrid()` has run to report which backends will be active.
        """
        dense = bool(self.cfg.use_dense and self._faiss_available())
        sparse = bool(self.cfg.use_bm25 and self._bm25_available())
        return dense, sparse

    def _load(self) -> None:
        if self.cfg.use_dense and self._faiss_available():
            try:
                self.faiss = FaissTextIndex(self.index_dir, self.dim).load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("FAISS index load failed (%s); dense path disabled", exc)
                self.faiss = None
        if self.cfg.use_bm25 and self._bm25_available():
            try:
                self.bm25 = Bm25Index(self.index_dir).load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("BM25 index load failed (%s); sparse path disabled", exc)
                self.bm25 = None

    # ── query ─────────────────────────────────────────────────────────────
    def hybrid(self, queries: list[str], *, top_k: int | None = None) -> HybridResult:
        """Run hybrid retrieval for a set of query strings.

        Each query contributes its own dense and sparse ranking; the fused
        rankings across all queries are then merged with RRF (so multiple query
        variants are combined, not just one).
        """
        self._load()
        k = top_k or self.cfg.candidates

        dense_rankings: list[list[str]] = []
        sparse_rankings: list[list[str]] = []
        for q in queries:
            if self.faiss is not None and self.cfg.use_dense:
                try:
                    vec = self.models.embed_one(q)
                    res = self.faiss.search(vec, k)
                    dense_rankings.append([cid for cid, _ in res])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("dense search failed for %r: %s", q, exc)
            if self.bm25 is not None and self.cfg.use_bm25:
                try:
                    res = self.bm25.search(q, k)
                    sparse_rankings.append([cid for cid, _ in res])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("sparse search failed for %r: %s", q, exc)

        all_rankings = dense_rankings + sparse_rankings
        if not all_rankings:
            logger.warning("no retrieval backends available — returning empty result")
            return HybridResult(
                dense_ranked=[], sparse_ranked=[]
            )

        fused = reciprocal_rank_fusion(
            all_rankings,
            k=self.cfg.rrf_k,
        )
        # Weight dense vs sparse in the final ordering (rank-based so mostly
        # tie-breaking). Simple approach: scale each source's contribution.
        return HybridResult(
            ranked=fused[:k],
            dense_ranked=[cid for r in dense_rankings for cid in r],
            sparse_ranked=[cid for r in sparse_rankings for cid in r],
        )

    def text_of(self, chunk_id: str) -> str | None:
        rec = self.store.get_chunk(chunk_id)
        return rec.text if rec else None
