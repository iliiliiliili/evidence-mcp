"""FAISS dense vector index over chunk embeddings.

The FAISS index stores embeddings in the SAME ORDER as the chunk id list passed
to it; we persist both a `.faiss` index file and a `chunk_ids.json` sidecar so
the mapping survives across runs. Retrieval returns (chunk_id, score) pairs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..models.ollama import Models

logger = logging.getLogger(__name__)


class FaissTextIndex:
    def __init__(self, index_dir: str | Path, dim: int):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index_path = self.index_dir / "chunks.faiss"
        self.ids_path = self.index_dir / "chunk_ids.json"
        self.dim_path = self.index_dir / "dim.json"

    # ── build ─────────────────────────────────────────────────────────────
    def build(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        import faiss

        if not chunk_ids or not vectors:
            raise ValueError("empty vectors — nothing to index")
        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding dim {arr.shape[1]} != index dim {self.dim}")

        index = faiss.IndexFlatIP(self.dim)  # inner product = cosine for L2-normed
        # L2-normalize so inner product == cosine similarity.
        faiss.normalize_L2(arr)
        index.add(arr)
        faiss.write_index(index, str(self.index_path))
        self.ids_path.write_text(json.dumps(chunk_ids))
        self.dim_path.write_text(json.dumps({"dim": self.dim}))
        logger.info("Built FAISS index with %d chunks (dim=%d)", len(chunk_ids), self.dim)

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        """Append vectors to an existing index (or create it if none yet).

        Cheaper than a full rebuild: no existing vectors are re-embedded. The
        on-disk index is reloaded if it isn't already in memory.
        """
        import faiss

        if not chunk_ids or not vectors:
            raise ValueError("empty vectors — nothing to index")
        arr = np.asarray(vectors, dtype="float32")
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding dim {arr.shape[1]} != index dim {self.dim}")
        faiss.normalize_L2(arr)
        if not self.index_path.exists():
            self.build(chunk_ids, vectors)
            return
        self.load()
        self.index.add(arr)
        faiss.write_index(self.index, str(self.index_path))
        existing = json.loads(self.ids_path.read_text()) if self.ids_path.exists() else []
        self.ids_path.write_text(json.dumps(existing + list(chunk_ids)))
        logger.info(
            "Appended %d chunks to FAISS index (total %d)",
            len(chunk_ids),
            len(existing) + len(chunk_ids),
        )

    # ── load ──────────────────────────────────────────────────────────────
    def load(self) -> FaissTextIndex:
        import faiss

        self._index = faiss.read_index(str(self.index_path))
        self._ids = json.loads(self.ids_path.read_text())
        if self.dim_path.exists():
            self.dim = int(json.loads(self.dim_path.read_text())["dim"])
        return self

    @classmethod
    def dim_from_dir(cls, index_dir: str | Path) -> int | None:
        p = Path(index_dir) / "dim.json"
        if p.exists():
            try:
                return int(json.loads(p.read_text())["dim"])
            except Exception:  # noqa: BLE001
                return None
        return None

    @property
    def index(self):
        if not hasattr(self, "_index"):
            self.load()
        return self._index

    @property
    def ids(self) -> list[str]:
        if not hasattr(self, "_ids"):
            self.load()
        return self._ids

    # ── query ─────────────────────────────────────────────────────────────
    def search(self, query_vec: list[float] | np.ndarray, k: int) -> list[tuple[str, float]]:
        import faiss

        arr = np.asarray([query_vec], dtype="float32")
        faiss.normalize_L2(arr)
        k = min(k, len(self.ids))
        scores, idxs = self.index.search(arr, k)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0], strict=True):
            if i < 0:
                continue
            out.append((self.ids[int(i)], float(score)))
        return out


def embed_chunks(models: Models, chunk_ids: list[str], texts: list[str]) -> list[list[float]]:
    """Embed a list of chunk texts via the configured model (with fallback)."""
    return models.embed(texts)
