"""Knowledge-graph construction (Phase 2, PLAN.MD offline pipeline steps 3–6).

- `entities`:  per-paper (Author, Topic, Method) extraction
- `authors`:   OpenAlex author dedupe / validation
- `citations`: intra-corpus citation + co-citation resolution
- `neo4j`:     Neo4j loader (Paper/Author/Topic/Method/CITES schema)
- `pipeline`:  `build_kg(...)` — runs the whole thing over an ingested Store
"""

from .authors import AuthorIdentity, group_by_identity, validate_authors
from .citations import (
    CitationLink,
    co_citation_pairs,
    extract_dois,
    match_corpus,
    norm_doi,
    resolve_citations_for_corpus,
)
from .entities import PaperEntities, extract_entities
from .pipeline import KgReport, build_kg

__all__ = [
    "AuthorIdentity",
    "group_by_identity",
    "validate_authors",
    "CitationLink",
    "co_citation_pairs",
    "extract_dois",
    "match_corpus",
    "norm_doi",
    "resolve_citations_for_corpus",
    "PaperEntities",
    "extract_entities",
    "KgReport",
    "build_kg",
]
