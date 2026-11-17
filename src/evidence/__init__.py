"""Evidence — open, local-first, evidence-gated agentic RAG for technical literature.

A LangGraph reimplementation of *Evidence: Evidence-Gated Multimodal Agentic RAG
for Technical Literature Reasoning* (arXiv:2606.01613).

Core loop is fully local (Ollama models, FAISS/BM25, no cloud API dependencies).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
