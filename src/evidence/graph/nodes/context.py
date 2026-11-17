"""Dependency registry for graph nodes.

Nodes are pure functions of `(state)` but need shared services (models,
retriever, store, config). We thread them through a single `NodeContext` that
is bound once at graph-build time and read by every node. Keeping nodes free of
constructor args makes them trivially testable and keeps the LangGraph builder
declarative.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import EvidenceConfig
from ...indexes.retriever import Retriever
from ...indexes.store import Store
from ...models.ollama import Models


@dataclass
class NodeContext:
    cfg: EvidenceConfig
    models: Models
    store: Store
    retriever: Retriever


# Module-level singleton set during graph construction.
_CONTEXT: NodeContext | None = None


def set_context(ctx: NodeContext) -> None:
    global _CONTEXT
    _CONTEXT = ctx


def get_context() -> NodeContext:
    if _CONTEXT is None:
        raise RuntimeError(
            "NodeContext not initialised. Build the graph via "
            "evidence.graph.builder.build_graph(...) first, or call "
            "evidence.graph.nodes.context.set_context(...) in tests."
        )
    return _CONTEXT
