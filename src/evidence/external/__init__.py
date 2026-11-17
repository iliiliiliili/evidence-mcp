"""External academic search providers (Phase 3)."""

from . import crossref, openalex, semantic_scholar
from .backoff import backoff, retry_on_error

__all__ = ["openalex", "semantic_scholar", "crossref", "backoff", "retry_on_error"]
