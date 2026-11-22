"""Model wrappers (Ollama chat/embed/rerank + deterministic fallbacks)."""

from .ollama import (
    CrossEncoderReranker,
    DeterministicEmbedder,
    DeterministicReranker,
    EmbedBackend,
    Models,
    OllamaModel,
    OpenAIModel,
    extract_json,
)

__all__ = [
    "Models",
    "OllamaModel",
    "OpenAIModel",
    "EmbedBackend",
    "CrossEncoderReranker",
    "DeterministicEmbedder",
    "DeterministicReranker",
    "extract_json",
]
