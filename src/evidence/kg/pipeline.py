"""Knowledge-graph build pipeline (PLAN.MD offline pipeline, steps 3–6).

Ties together entity extraction, citation resolution, author validation, and
(optional) Neo4j loading. Designed to run over the ingested `Store` so it
reuses the exact papers/chunks that Phase 1 indexes.

Usage (from the CLI or a script):

    from evidence.kg import pipeline
    report = pipeline.build_kg(store, models=models, load_neo4j=False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..indexes.store import Store
from ..models.ollama import Models
from . import authors as author_mod
from . import citations as citation_mod
from . import entities as entity_mod
from . import neo4j as neo4j_mod
from .authors import AuthorIdentity
from .citations import CitationLink
from .entities import PaperEntities

logger = logging.getLogger(__name__)


@dataclass
class KgReport:
    """Summary of a knowledge-graph build run."""

    papers_processed: int = 0
    entities: dict[str, PaperEntities] = field(default_factory=dict)
    authors: dict[str, list[AuthorIdentity]] = field(default_factory=dict)
    citations: list[CitationLink] = field(default_factory=list)
    co_citations: dict = field(default_factory=dict)
    neo4j: neo4j_mod.KgStats | None = None
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "papers_processed": self.papers_processed,
            "entities": {
                pid: {"topics": len(e.topics), "methods": len(e.methods), "authors": len(e.authors)}
                for pid, e in self.entities.items()
            },
            "authors": {pid: len(a) for pid, a in self.authors.items()},
            "citations": len(self.citations),
            "co_citation_pairs": len(self.co_citations),
            "neo4j": (
                {
                    "papers": self.neo4j.papers,
                    "cites": self.neo4j.cites,
                    "failed": len(self.neo4j.failed),
                }
                if self.neo4j
                else None
            ),
            "skipped": self.skipped,
        }


def build_kg(
    store: Store,
    *,
    models: Models | None = None,
    cfg=None,
    load_neo4j: bool = False,
    neo4j_uri: str = neo4j_mod._DEFAULT_URI,
    neo4j_auth: tuple[str, str] = neo4j_mod._DEFAULT_AUTH,
    author_online: bool = False,
) -> KgReport:
    """Build the knowledge graph over the papers in `store`.

    Steps (PLAN.MD offline pipeline):
      3. entity extraction          -> `entities`
      4. OpenAlex author validation -> `authors` (offline dedupe by default)
      5. intra-corpus citation res. -> `citations` + `co_citations`
      6. (optional) Neo4j load      -> `neo4j` stats
    """
    report = KgReport()
    papers = store.list_papers()
    if not papers:
        logger.warning("build_kg: no papers in store; nothing to do")
        return report

    # Preload each paper's chunk texts (bounded) for entity / reference mining.
    paper_chunks: dict[str, list[str]] = {}
    for p in papers:
        chunks = store.chunks_for_paper(p.paper_id)
        paper_chunks[p.paper_id] = [c.text for c in chunks[:20]]

    # Step 3 — entities.
    for p in papers:
        try:
            text = "\n\n".join(paper_chunks.get(p.paper_id, [])) or p.abstract or p.title
            ent = entity_mod.extract_entities(
                paper_id=p.paper_id,
                title=p.title,
                text=text,
                authors=p.authors,
                models=models,
                cfg=cfg,
            )
            report.entities[p.paper_id] = ent
            report.papers_processed += 1
        except Exception as exc:  # noqa: BLE001 - keep going
            logger.warning("entity extraction failed for %s: %s", p.paper_id, exc)
            report.skipped.append(p.paper_id)

    # Step 4 — author validation (offline dedupe by default).
    for p in papers:
        ent = report.entities.get(p.paper_id)
        names = list(ent.authors) if ent else list(p.authors)
        ext = getattr(cfg, "external", None) if cfg is not None else None
        mailto = getattr(ext, "openalex_mailto", None)
        try:
            idents = author_mod.validate_authors(names, online=author_online, mailto=mailto)
            report.authors[p.paper_id] = idents
        except Exception as exc:  # noqa: BLE001
            logger.warning("author validation failed for %s: %s", p.paper_id, exc)
            report.authors[p.paper_id] = author_mod.group_by_identity(names)

    # Step 5 — intra-corpus citation resolution.
    papers_for_citations = [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "doi": p.doi,
            "text": "\n\n".join(paper_chunks.get(p.paper_id, [])),
        }
        for p in papers
    ]
    try:
        report.citations = citation_mod.resolve_citations_for_corpus(papers_for_citations)
        report.co_citations = citation_mod.co_citation_pairs(report.citations)
    except Exception as exc:  # noqa: BLE001
        logger.warning("citation resolution failed: %s", exc)

    # Step 6 — optional Neo4j load.
    if load_neo4j:
        try:
            neo_papers = [
                {"paper_id": p.paper_id, "title": p.title, "year": p.year, "doi": p.doi}
                for p in papers
            ]
            report.neo4j = neo4j_mod.build_knowledge_graph(
                papers=neo_papers,
                entities_by_id=report.entities,
                authors_by_id=report.authors,
                citations=report.citations,
                uri=neo4j_uri,
                auth=neo4j_auth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j load failed: %s", exc)

    return report


__all__ = [
    "KgReport",
    "build_kg",
    "AuthorIdentity",
    "CitationLink",
    "PaperEntities",
]
