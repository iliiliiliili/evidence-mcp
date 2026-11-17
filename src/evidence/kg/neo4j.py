"""Neo4j knowledge-graph loader (PLAN.MD offline pipeline, steps 3–5, 6).

Builds the graph schema the `expand_via_graph` node (stage 5) traverses:

    (:Paper {paper_id, title, year, doi})
      -[:BY]->          (:Author {name, openalex_id})
      -[:ON_TOPIC]->    (:Topic  {name})
      -[:USES_METHOD]-> (:Method {name})
      -[:CITES]->       (:Paper {paper_id})

`Topic` / `Method` act as the shared hubs: two papers are "related" when they
both connect to the same hub (the Cypher in `expand_graph` walks
`Paper—Topic|Method—Paper`). All writes are idempotent (`MERGE`).

Requires the `neo4j` driver (optional extra) and a running instance. The
connection is injected so tests / the CLI can point at any endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .authors import AuthorIdentity
from .citations import CitationLink
from .entities import PaperEntities

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://localhost:7687"
_DEFAULT_AUTH = ("neo4j", "evidence")


@dataclass
class KgStats:
    papers: int = 0
    authors: int = 0
    topics: int = 0
    methods: int = 0
    cites: int = 0
    failed: list[str] = field(default_factory=list)


def _get_driver(uri: str, auth: tuple[str, str]):
    try:
        import neo4j  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The 'neo4j' package is not installed. "
            "Install with `pip install neo4j` or `pip install evidence[neo4j]`."
        ) from exc
    return neo4j.GraphDatabase.driver(uri, auth=auth)


def _ensure_schema(session) -> None:
    session.run(
        """
        CREATE CONSTRAINT paper_id_unique IF NOT EXISTS
            FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;
        CREATE CONSTRAINT author_norm_unique IF NOT EXISTS
            FOR (a:Author) REQUIRE a.normalized IS UNIQUE;
        CREATE CONSTRAINT topic_name_unique IF NOT EXISTS
            FOR (t:Topic) REQUIRE t.name IS UNIQUE;
        CREATE CONSTRAINT method_name_unique IF NOT EXISTS
            FOR (m:Method) REQUIRE m.name IS UNIQUE;
        CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:Paper) ON (p.title);
        """
    )


def load_paper(
    session,
    *,
    entities: PaperEntities,
    authors: list[AuthorIdentity] | None = None,
) -> None:
    """Load one paper + its entity edges (idempotent)."""
    session.run(
        """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title, p.year = $year, p.doi = $doi
        """,
        paper_id=entities.paper_id,
        title=entities.title,
        year=None,
        doi=None,
    )
    for name in entities.topics:
        session.run(
            "MERGE (t:Topic {name: $name}) "
            "MERGE (p:Paper {paper_id: $pid})-[:ON_TOPIC]->(t)",
            name=name,
            pid=entities.paper_id,
        )
    for name in entities.methods:
        session.run(
            "MERGE (m:Method {name: $name}) "
            "MERGE (p:Paper {paper_id: $pid})-[:USES_METHOD]->(m)",
            name=name,
            pid=entities.paper_id,
        )
    for a in authors or []:
        session.run(
            """
            MERGE (au:Author {normalized: $norm})
            SET au.name = $name, au.openalex_id = $oid
            MERGE (p:Paper {paper_id: $pid})-[:BY]->(au)
            """,
            norm=a.normalized,
            name=a.display_name,
            oid=a.openalex_id,
            pid=entities.paper_id,
        )


def load_citations(session, links: list[CitationLink]) -> None:
    for link in links:
        if not link.target_paper_id:
            continue
        session.run(
            """
            MERGE (src:Paper {paper_id: $src})
            MERGE (tgt:Paper {paper_id: $tgt})
            MERGE (src)-[:CITES]->(tgt)
            SET rel.method = $method, rel.confidence = $conf,
                rel.target_doi = $doi
            """,
            src=link.source_paper_id,
            tgt=link.target_paper_id,
            method=link.method,
            conf=link.confidence,
            doi=link.target_doi,
        )


def build_knowledge_graph(
    *,
    papers: list[dict],
    entities_by_id: dict[str, PaperEntities],
    authors_by_id: dict[str, list[AuthorIdentity]],
    citations: list[CitationLink],
    uri: str = _DEFAULT_URI,
    auth: tuple[str, str] = _DEFAULT_AUTH,
) -> KgStats:
    """Load the full knowledge graph into Neo4j.

    `papers`: list of dicts with `paper_id`, `title`, `year` (optional),
    `doi` (optional). `entities_by_id` / `authors_by_id` / `citations` come
    from the `entities`, `authors`, and `citations` modules.
    """
    stats = KgStats()
    driver = _get_driver(uri, auth)
    try:
        with driver.session() as session:
            _ensure_schema(session)
            for p in papers:
                pid = p["paper_id"]
                ent = entities_by_id.get(pid)
                if ent is None:
                    stats.failed.append(pid)
                    continue
                # Attach year/doi if present.
                try:
                    session.run(
                        "MATCH (p:Paper {paper_id: $pid}) "
                        "SET p.year = $year, p.doi = $doi",
                        pid=pid,
                        year=p.get("year"),
                        doi=p.get("doi"),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("setting year/doi failed for %s: %s", pid, exc)
                load_paper(
                    session,
                    entities=ent,
                    authors=authors_by_id.get(pid),
                )
                stats.papers += 1
            for links in authors_by_id.values():
                stats.authors += len(links)
            for ent in entities_by_id.values():
                stats.topics += len(ent.topics)
                stats.methods += len(ent.methods)
            load_citations(session, citations)
            stats.cites = len(citations)
    finally:
        driver.close()
    logger.info(
        "knowledge graph built: %d papers, %d authors, %d topics, %d methods, %d cites",
        stats.papers, stats.authors, stats.topics, stats.methods, stats.cites,
    )
    return stats


def query_related_papers(
    paper_id: str,
    *,
    limit: int = 10,
    uri: str = _DEFAULT_URI,
    auth: tuple[str, str] = _DEFAULT_AUTH,
) -> list[dict]:
    """Return papers sharing a topic/method hub with `paper_id` (stage 5 helper)."""
    driver = _get_driver(uri, auth)
    try:
        with driver.session() as session:
            res = session.run(
                """
                MATCH (p:Paper {paper_id: $pid})
                  -[r:ON_TOPIC|USES_METHOD]-(shared:Topic|Method)
                  -[r2:ON_TOPIC|USES_METHOD]-(rel:Paper)
                WHERE rel.paper_id <> p.paper_id
                RETURN DISTINCT rel.paper_id AS paper_id,
                               rel.title AS title,
                               shared.name AS shared
                LIMIT $limit
                """,
                pid=paper_id,
                limit=limit,
            )
            return [dict(r) for r in res]
    finally:
        driver.close()
