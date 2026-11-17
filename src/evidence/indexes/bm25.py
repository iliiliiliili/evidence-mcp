"""BM25 sparse index (rank_bm25) over chunk texts.

Persisted as a pickle of the BM25Okapi object + a sidecar of chunk ids, so it
can be rebuilt from the store if the pickle is missing. Retrieval returns
(chunk_id, score) pairs.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from ..models.ollama import DeterministicReranker  # reuse tokenizer

logger = logging.getLogger(__name__)


class Bm25Index:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.index_dir / "bm25.pkl"
        self.ids_path = self.index_dir / "bm25_ids.json"
        self._tok = DeterministicReranker()._tokenize
        self._model = None
        self._ids: list[str] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        if not chunk_ids:
            raise ValueError("no chunks to index")
        tokenized = [self._tok(t) for t in texts]
        self._model = BM25Okapi(tokenized)
        self._ids = list(chunk_ids)
        with self.model_path.open("wb") as fh:
            pickle.dump(self._model, fh)
        self.ids_path.write_text(json.dumps(self._ids))
        logger.info("Built BM25 index with %d chunks", len(chunk_ids))

    def load(self) -> Bm25Index:
        with self.model_path.open("rb") as fh:
            self._model = pickle.load(fh)
        self._ids = json.loads(self.ids_path.read_text())
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = self._tok(query)
        if not tokens:
            return []
        scores = self.model.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], float(scores[i])) for i in order]
