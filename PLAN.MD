# TechRAG MCP Server — Tool Catalog & Client Setup

Exposes the TechRAG evidence pipeline (retrieval, graph expansion, external
literature search, evidence-gated QA) as MCP tools, so any MCP-capable client
(OpenCode, VS Code Copilot, Cline, Goose, Claude Code) can use the paper corpus
inside its own coding loop.

## Design principles

1. **Thin server, thick pipeline.** The MCP server is a wrapper over the existing
   TechRAG modules (`indexes/`, `external/`, `kg/`, `graph/`). No business logic here.
2. **Tools for agents, resources for context, prompts for workflows.** Agents get
   tools; IDEs get corpus stats as a resource; users get canned research prompts.
3. **Citation-first return shapes.** Every tool result carries paper title, year,
   chunk ID and (where possible) DOI — so the calling agent can cite sources in
   code comments, PRs, or docs.
4. **Read-only by default.** The only state-changing tools (`ingest_pdf`,
   `ingest_url`) are optional and require an explicit enable flag. Ingestion of
   new papers = corpus growth via citation expansion, gated by config.

## Tool catalog (mapped to pipeline stages)

| Tool | Pipeline stage | Purpose | Mutating? |
|---|---|---|---|
| `search_corpus` | 3–4 (hybrid retrieval + rerank) | Hybrid text search over paper chunks; returns cited chunks. The workhorse. | No |
| `search_figures` | 6 (visual retrieval) | Visual page retrieval — find figures/tables/diagrams matching a query (Phase 4). | No |
| `expand_via_graph` | 5 (graph expansion) | Given a chunk, traverse topic/method/co-citation edges in Neo4j; return related chunks. | No |
| `check_evidence` | 7 (sufficiency gate) | Score a proposed answer/evidence set against the 100-pt rubric; returns per-dimension scores + verdict. | No |
| `search_literature` | 9 (external search) | OpenAlex/S2/Crossref search *outside* the corpus; returns metadata + OA PDF URL. | No |
| `expand_citations` | 9 (citation expansion) | Forward/backward citations of a paper via OpenAlex/S2, incl. citation contexts. | No |
| `ingest_paper` | offline (ingestion) | Add a PDF/LaTeX/arXiv URL to the corpus and index it. **Optional, flag-gated.** | Yes |
| `answer_research` | 1–13 (full pipeline) | Run the whole evidence-gated TechRAG graph on a research question; returns cited answer + evidence map. Heavy — document the cost in the description. | No |
| `get_paper` | — | Fetch full metadata + section map + reference list of a corpus paper by ID/DOI. | No |
| `verify_claim` | 11 (citation integrity) | Check whether a specific claim maps to a real indexed chunk (rule-based, no LLM). | No |
| `lookup_evidence` | Evidence ledger | **Cache-first:** get the stored decision record (rationale, alternatives, citations, status) for a function/class/API without running retrieval. Check this BEFORE search_corpus/answer_research. | No |
| `record_evidence` | Evidence ledger | Write/update a decision record for a symbol after a justified change. PR shows the evidence diff next to the code diff. | Yes |
| `audit_evidence` | Evidence ledger | Scan code fingerprints; list records marked `stale_code`/`stale_evidence`/`orphaned`; re-checks only invalidated records. | No |
| `impact_analysis` | Evidence ledger | Given a symbol or evidence chunk, list all other decisions citing the same evidence — contradiction/change propagation map. | No |

Resources: `papers://corpus/stats` (counts, last updated), `papers://{id}/outline`.

Prompts: `techrag.literature-review` (topic → structured review workflow),
`techrag.related-work` (draft text → find contradicting/supporting prior work).

## Server implementation

```
techrag/
  src/techrag/mcp/
    server.py          # FastMCP instance + tool registration
    tools/             # one module per tool, thin wrappers
    schemas.py         # TypedDicts for return shapes (converted to JSON)
```

```python
# src/techrag/mcp/server.py
"""TechRAG MCP server. Run: techrag-mcp (stdio) — see cli.py wiring."""
import os
from mcp.server.fastmcp import FastMCP

from techrag.indexes import faiss_text, bm25, neo4j_store
from techrag.external import openalex, semantic_scholar, crossref
from techrag.graph import builder as pipeline          # full LangGraph pipeline
from techrag.ingestion import pdf as ingest_pdf_mod
from techrag.models.ollama import reranker, embeddings
from techrag.ledger import store as ledger            # evidence ledger

mcp = FastMCP("techrag")
ALLOW_INGEST = os.environ.get("TECHRAG_ALLOW_INGEST", "0") == "1"


@mcp.tool()
def search_corpus(query: str, limit: int = 8) -> list[dict]:
    """Hybrid search over the local paper corpus (dense FAISS + BM25, RRF-fused,
    cross-encoder reranked). Returns text chunks from scientific papers with
    paper title, year, section, chunk ID, and relevance score. Use this for any
    question about methods, equations, prior work, or technical background
    instead of guessing from memory. Cost: one local rerank, ~1-2s."""
    fused = _hybrid_fuse(query, candidates=40)
    top = reranker.rerank(query, fused, k=limit)
    return [_chunk_to_citation(c) for c in top]


@mcp.tool()
def expand_via_graph(chunk_id: str, relation: str = "SHARES_TOPIC",
                    limit: int = 5) -> list[dict]:
    """Find chunks from OTHER papers connected to this chunk's paper through the
    knowledge graph: same topic, same method, citing, or cited-by. Use after
    search_corpus when the question needs related work beyond the retrieved
    paper. relation: SHARES_TOPIC | SHARES_METHOD | CITED_BY | CITES."""
    rows = neo4j_store.traverse(chunk_id, relation, limit=limit)
    return [_chunk_to_citation(r) for r in rows]


@mcp.tool()
def search_literature(query: str, intent: str = "content",
                      limit: int = 10) -> list[dict]:
    """Search OUTSIDE the local corpus via OpenAlex/Semantic Scholar/Crossref.
    Returns paper metadata with year, citation count, and an open-access PDF URL
    when available. intent: content | bibliometric | trend | current-world —
    controls sort/strategy. Use for recent work the corpus may not contain."""
    if intent == "current-world":
        works = openalex.search(query, sort="publication_date:desc", limit=limit)
    else:
        works = openalex.search(query, sort="cited_by_count:desc", limit=limit)
    return [_work_to_citation(w) for w in works]


@mcp.tool()
def expand_citations(doi_or_id: str, direction: str = "citing",
                     limit: int = 10) -> list[dict]:
    """Find works CITED BY or CITING a paper (OpenAlex, with Semantic Scholar
    citation contexts as fallback for AI/ML papers). direction: citing | cited.
    Returns citing works with year, citation count, and OA PDF URL."""
    if direction == "citing":
        works = openalex.works_citing(doi_or_id, limit=limit)
        contexts = semantic_scholar.citation_contexts(doi_or_id, limit=limit)
        return [_merge(w, c) for w, c in zip(works, contexts)]
    return [_work_to_citation(w)
            for w in openalex.referenced_by(doi_or_id, limit=limit)]


@mcp.tool()
def verify_claim(claim: str, chunk_id: str) -> dict:
    """Rule-based citation check: does the claim (entailment against the chunk
    text, local NLI model) follow from the indexed chunk? Returns
    SUPPORTED | PARTIAL | UNSUPPORTED with the exact chunk text. Use before
    citing a source for a strong claim in code, docs, or PRs."""
    return pipeline.verify_claim(claim, chunk_id)


@mcp.tool()
def check_evidence(question: str, chunk_ids: list[str]) -> dict:
    """Score an evidence set against the 100-point sufficiency rubric
    (coverage, relevance, depth, recency, consistency). Returns per-dimension
    scores, a verdict (sufficient | retry | need_external), and suggestions for
    missing evidence. Use to decide whether to retrieve more before answering."""
    return pipeline.score_sufficiency(question, chunk_ids)


@mcp.tool()
def get_paper(paper_id: str) -> dict:
    """Full metadata for a corpus paper: title, year, authors (OpenAlex-validated),
    abstract, section outline, reference list, and topic/method graph nodes."""
    return neo4j_store.paper_record(paper_id)


@mcp.tool()
def answer_research(question: str) -> dict:
    """Run the FULL TechRAG evidence-gated pipeline on a research question:
    intent classification, hybrid retrieval, graph expansion, external search,
    citation verification, Planner-Researcher-Writer-Critic generation.
    Returns a cited answer plus an evidence map (claim -> chunk). Heavy:
    tens of local-model calls, may take minutes. Prefer search_corpus for
    quick lookups; use this for open research questions."""
    result = pipeline.ask(question)
    return {"answer": result["answer"], "citations": result["citations"],
            "evidence_map": result["evidence_map"], "qc": result["qc_report"]}


@mcp.tool()
def ingest_paper(url_or_path: str) -> dict:
    """Add a PDF (local path) or arXiv/OpenAlex URL to the corpus: download,
    parse, chunk, embed, extract entities, resolve citations, update indexes.
    MUTATES the corpus — only call when the user explicitly asks to add a paper
    to their library."""
    if not ALLOW_INGEST:
        return {"error": "Ingestion disabled. Set TECHRAG_ALLOW_INGEST=1."}
    return ingest_pdf_mod.ingest(url_or_path)


```python
# --- Evidence ledger tools (cache-first layer) ---

@mcp.tool()
def lookup_evidence(symbol: str) -> dict | None:
    """Return the stored evidence record for a function/class/API: the design
    decision, its rationale, alternatives considered, supporting citations
    (paper + chunk), and staleness status. CHECK THIS FIRST for any question
    about why code is written the way it is, or before modifying a symbol —
    a current record means no retrieval or analysis is needed. Returns null
    if no record exists (then use search_corpus). Cost: single SQLite lookup."""
    return ledger.lookup(symbol)


@mcp.tool()
def record_evidence(symbol: str, decision: str,
                    evidence: list[dict],
                    alternatives: list[dict] | None = None) -> dict:
    """Write/update the decision record for a symbol after making a justified
    change (e.g. choosing an algorithm or API because of retrieved evidence).
    evidence items reference corpus chunk IDs or external DOIs from prior
    search_corpus/search_literature/expand_citations calls. MUTATES the ledger."""
    return ledger.record(symbol, decision, evidence, alternatives)


@mcp.tool()
def audit_evidence(scope: str = "repo") -> dict:
    """Scan code fingerprints and report decision records whose code or cited
    evidence has drifted: stale_code (symbol changed since the record was
    written), stale_evidence (cited chunk changed or a new corpus paper
    contradicts it), orphaned (symbol removed/renamed). Only invalidated
    records are queued for targeted re-analysis. Cost: hashing, no LLM."""
    return ledger.audit(scope)


@mcp.tool()
def impact_analysis(symbol_or_chunk: str) -> dict:
    """Given a symbol (or an evidence chunk ID), list every OTHER decision
    record that cites the same evidence. Use before changing a method, class,
    or API: shows which other decisions would inherit a contradiction."""
    return ledger.impact(symbol_or_chunk)


@mcp.resource("papers://corpus/stats")
def corpus_stats() -> dict:
    return neo4j_store.corpus_stats()


def _chunk_to_citation(c: dict) -> dict:
    return {"chunk_id": c["chunk_id"], "text": c["text"],
            "paper": c["paper_title"], "year": c["year"],
            "section": c.get("section"), "doi": c.get("doi"),
            "relevance": round(c["score"], 3)}


def _work_to_citation(w: dict) -> dict:
    return {"paper": w["display_name"], "year": w["publication_year"],
            "doi": w.get("doi"), "cited_by": w.get("cited_by_count"),
            "oa_pdf": (w.get("open_access") or {}).get("oa_url"),
            "source": w.get("_source")}


def _hybrid_fuse(query: str, candidates: int) -> list[dict]:
    dense = faiss_text.search(embeddings.embed(query), k=candidates)
    sparse = bm25.search(query, k=candidates)
    return _rrf(dense, sparse)


def _rrf(*lists, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion, deduped by chunk_id."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            cid = item["chunk_id"]
            items.setdefault(cid, item)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(items.values(),
                  key=lambda it: scores[it["chunk_id"]], reverse=True)


if __name__ == "__main__":
    mcp.run()  # stdio transport — works everywhere, runs fully local
```

Return-shape notes: keep tool outputs **compact** (truncate chunk text at ~1500
chars, offer a `get_chunk`-style detail path via `get_paper`), always include
`chunk_id` so agents can chain `verify_claim`, and surface provenance
(`source: local | openalex | s2`) so callers know what's inside vs. outside the corpus.

## Client setup

### OpenCode

`opencode.json` (project) or `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "techrag": {
      "type": "local",
      "command": ["uv", "run", "techrag-mcp"],
      "env": {"TECHRAG_ALLOW_INGEST": "0"}
    }
  }
}
```

### VS Code Copilot

Yes — Copilot agent mode is MCP-native. Two config formats:

**VS Code format** (IntelliSense, richer features) — `.vscode/mcp.json`:

```json
{
  "servers": {
    "techrag": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "techrag-mcp"],
      "env": {"TECHRAG_ALLOW_INGEST": "0"}
    }
  }
}
```

**Portable Copilot format** (shared with other Copilot harnesses) — `.mcp.json`
at workspace root, or `~/.copilot/mcp-config.json` for user level:

```json
{
  "mcpServers": {
    "techrag": {
      "command": "uv",
      "args": ["run", "techrag-mcp"]
    }
  }
}
```

Then: open Copilot Chat → Agent (or the Copilot harness) → the server starts and
its tools appear; confirm tool use when prompted. A `dev` block in the VS Code
format (`"dev": {"watch": "src/techrag/mcp/**/*.py", "debug": true}`) gives you
auto-restart and Python debugging while developing the server.

Caveats for the Copilot path:

- **Cloud agent / code review** (GitHub-hosted Copilot): repo-level MCP config
  supports **tools only** — no resources or prompts — and no OAuth remote servers.
  A local stdio server therefore works in VS Code desktop, but not in the
  GitHub-hosted cloud agent. For cloud use you'd deploy TechRAG as a remote
  HTTP MCP server (FastMCP `mcp.run(transport="streamable-http")` behind auth).
- Workspace files start only after you approve trust; tools are then called
  autonomously (subject to the client's approval settings).

### Other clients (same server, no code changes)

Cline/Goose/Claude Code all use the stdio config shape above with minor schema
naming differences (`mcpServers` everywhere; Claude Code accepts the portable
`.mcp.json` directly). One server, every agent.

## Testing

```bash
# Inspector: interactive tool-calling UI, no agent needed
npx @modelcontextprotocol/inspector uv run techrag-mcp

# Smoke test the contract
mcp-cli call techrag search_corpus --args '{"query": "slip angle estimation", "limit": 3}'
```

Checklist before shipping: every tool has a cost/latency hint in its description;
descriptions say *when to use* the tool (not just what it does — models choose on
description text); no tool exceeds ~2KB of returned text without a truncation
escape hatch; `answer_research` is clearly marked heavy so clients don't call it
casually; and `lookup_evidence` descriptions direct agents to the cache-first
flow (ledger before retrieval) so the pipeline only runs on genuine misses.