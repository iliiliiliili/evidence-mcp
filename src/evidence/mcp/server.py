"""Evidence MCP server (new plan: `mcp/server.py`).

Run as stdio via the `evidence-mcp` entry point. Thin registration layer over
`evidence.mcp.tools` (the real logic) so the tools stay unit-testable without the
`mcp` package installed.

Transport is stdio by default (works in every local MCP client, fully local).
For cloud agents, deploy as `streamable-http` behind auth (see PLAN caveats).
"""

from __future__ import annotations

import logging

from .. import facade
from . import tools

logger = logging.getLogger(__name__)


def create_server(name: str = "evidence"):
    """Build and return the FastMCP server with all tools registered.

    Kept separate from `main()` so tests can import the server object without
    starting the stdio loop.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The 'mcp' package is not installed. Install with "
            "`pip install mcp` or `pip install evidence[mcp]`."
        ) from exc

    mcp = FastMCP(name)

    # ── Read-only retrieval / evidence tools ──────────────────────────────
    @mcp.tool()
    def search_corpus(query: str, limit: int = 8) -> list[dict]:
        """Hybrid text search over the local paper corpus (dense FAISS + BM25,
        RRF-fused, cross-encoder reranked). Returns cited chunks with paper
        title, year, section, chunk ID, and relevance score. THE WORKHORSE —
        use for any question about methods, equations, prior work, or technical
        background instead of guessing from memory. Cost: one local rerank,
        ~1–2s. Read-only.

        Check `lookup_evidence` first if your question is about *why a specific
        piece of code is written the way it is* — a cached record may make
        retrieval unnecessary."""
        return tools.search_corpus(query, limit=limit)

    @mcp.tool()
    def expand_via_graph(
        chunk_id: str, relation: str = "SHARES_TOPIC", limit: int = 5
    ) -> list[dict]:
        """Find chunks from OTHER papers connected to this chunk's paper through
        the knowledge graph (same topic, same method, citing, or cited-by). Use
        AFTER `search_corpus` when you need related work beyond the retrieved
        paper. relation: SHARES_TOPIC | SHARES_METHOD | CITED_BY | CITES.
        Falls back to a corpus text-nearest expansion when Neo4j is unavailable.
        Read-only, ~1s."""
        return tools.expand_via_graph(chunk_id, relation=relation, limit=limit)

    @mcp.tool()
    def search_figures(query: str, limit: int = 5) -> dict:
        """Visual page retrieval — find figures/tables/diagrams matching a query
        (stage 6, Phase 4). Returns a page-image match list when the MUVERA/
        ColSmol page index is built; otherwise a clear 'not available' envelope
        so you fall back to `search_corpus`. Read-only."""
        return tools.search_figures(query, limit=limit)

    @mcp.tool()
    def check_evidence(question: str, chunk_ids: list[str], intent: str = "content") -> dict:
        """Score an evidence set against the 100-point sufficiency rubric
        (coverage, relevance, depth, recency, consistency). Returns per-dimension
        scores, a total, a verdict (sufficient | retry), and suggestions for
        missing evidence. Use to decide whether to retrieve MORE before
        answering. `chunk_ids` come from prior `search_corpus` calls. Read-only,
        no LLM (deterministic rubric), ~50ms."""
        return tools.check_evidence(question, chunk_ids, intent=intent)

    @mcp.tool()
    def verify_claim(claim: str, chunk_id: str) -> dict:
        """Rule-based citation check: does `claim` follow from the indexed
        `chunk_id`? Returns SUPPORTED | PARTIAL | UNSUPPORTED with the exact
        chunk text. Use BEFORE citing a source for a strong claim in code,
        docs, or PRs. Read-only, deterministic, no LLM, ~5ms."""
        return tools.verify_claim(claim, chunk_id)

    @mcp.tool()
    def get_paper(paper_id: str) -> dict:
        """Full metadata for a corpus paper: title, year, authors, abstract,
        section outline, reference list, and chunk count. Use to expand a
        citation from `search_corpus` into a full reference. Read-only, ~10ms."""
        return tools.get_paper(paper_id)

    # ── External literature (stage 9) ─────────────────────────────────────
    @mcp.tool()
    def search_literature(
        query: str, intent: str = "content", limit: int = 10
    ) -> list[dict]:
        """Search OUTSIDE the local corpus via OpenAlex / Semantic Scholar /
        Crossref. Returns paper metadata with year, citation count, and an
        open-access PDF URL when available. intent: content | bibliometric →
        most-cited; trend → year order; current_world → newest first. Use for
        recent work the corpus may not contain. Network: ~1s (polite-pool
        rate-limited)."""
        return tools.search_literature(
            query, intent=intent, limit=limit, cfg=facade.get_app().cfg
        )

    @mcp.tool()
    def expand_citations(
        doi_or_id: str, direction: str = "citing", limit: int = 10
    ) -> list[dict]:
        """Find works CITED BY or CITING a paper (OpenAlex, with Semantic
        Scholar citation contexts as fallback). direction: 'citing' (forward)
        | 'cited' (backward). Returns citing works with year, citation count,
        and OA PDF URL. Use to walk the citation graph outward from a paper.
        Network: ~1–2s."""
        return tools.expand_citations(
            doi_or_id, direction=direction, limit=limit, cfg=facade.get_app().cfg
        )

    # ── Ingestion (mutating, flag-gated) ──────────────────────────────────
    @mcp.tool()
    def ingest_paper(url_or_path: str) -> dict:
        """Add a PDF (local path) or an arXiv/OpenAlex PDF URL to the corpus:
        download, parse, chunk, embed, index. MUTATES the corpus — only call
        when the user explicitly asks to add a paper to their library. Gated by
        `mcp.allow_ingest` (default off). Cost: embedding + index rebuild, ~10s."""
        return tools.ingest_paper(url_or_path)

    # ── Full pipeline (heavy) ─────────────────────────────────────────────
    @mcp.tool()
    def answer_research(question: str) -> dict:
        """Answer a research question from the corpus.

        Two modes (set by `mcp.prefer_client_generation` in the config):

        - **bundle mode** (flag on, recommended when you already have a strong
          model): returns an *evidence bundle* — retrieved chunks, a
          sufficiency score, and explicit guidance — for YOU to synthesise a
          cited answer. Cheap: retrieval + a deterministic score only; NO local
          generation models are run.

        - **full mode** (flag off, default): runs the local LangGraph pipeline
          (Planner/Writer/Critic) and returns a finished, cited `answer`.
          HEAVY: many local-model calls, may take minutes.

        For quick lookups prefer `search_corpus`; use this for open research
        questions."""
        return tools.answer_research(question)

    # ── Evidence ledger (cache-first layer) ───────────────────────────────
    @mcp.tool()
    def lookup_evidence(symbol: str) -> dict | None:
        """Return the stored decision record for a function/class/API: the
        design decision, its rationale, alternatives considered, supporting
        citations (paper + chunk / DOI), and staleness status. **CHECK THIS
        FIRST** for any question about why code is written the way it is, or
        before modifying a symbol — a current record means NO retrieval or
        analysis is needed. Returns null if no record exists (then use
        `search_corpus`). Cost: single indexed SQLite lookup, <5ms."""
        return tools.lookup_evidence(symbol)

    @mcp.tool()
    def record_evidence(
        symbol: str,
        decision: str,
        evidence: list[dict] | None = None,
        alternatives: list[dict] | None = None,
        rationale: str = "",
        code_fingerprint: str = "",
        depends_on: list[str] | None = None,
        cited: list[dict] | None = None,
    ) -> dict:
        """Write/update the decision record for a symbol after a justified
        change (e.g. choosing an algorithm or API because of retrieved
        evidence). `evidence` items reference corpus chunk IDs or external DOIs
        from prior `search_corpus` / `search_literature` / `expand_citations`
        calls.

        `depends_on` lists OTHER decisions this one is derived from (e.g.
        "use PyTorch" depends on "use Python"). This is what lets
        `impact_analysis` / `audit_evidence` propagate a root decision's change
        transitively through every decision built on it — even when they cite
        different evidence.

        `cited` lists the papers/methods the decision rests on (name, authors,
        optional DOI). Keep these distinct from `evidence` so the reviewer can
        see what you intended to cite even when a live lookup did not resolve
        a DOI. e.g. `[{"name": "Dropout as a Bayesian Approximation",
        "authors": "Gal & Ghahramani", "doi": "10.48550/arXiv.1505.05416"}]`.

        MUTATES the ledger. Cost: one SQLite write, <5ms."""
        return tools.record_evidence(
            symbol,
            decision,
            evidence=evidence,
            alternatives=alternatives,
            rationale=rationale,
            code_fingerprint=code_fingerprint,
            depends_on=depends_on,
            cited=cited,
        )

    @mcp.tool()
    def populate_scope(
        scope: str = ".",
        limit: int = 8,
        force: bool = False,
        external: bool | None = None,
    ) -> dict:
        """Autonomously draft, ground, and record design decisions for a
        directory of Python code (the same flow as `evidence populate` in the
        CLI). For each top-level symbol: draft a decision record (decision,
        rationale, alternatives, depends_on, and `cited` — the papers/methods
        the decision rests on), ground it in real evidence (local corpus +
        external literature, in parallel when `external` is enabled), and
        record the result in the ledger.

        Use this instead of calling `record_evidence` one-symbol-at-a-time
        when the user asks to "populate", "draft decisions for X", or "record
        why this code is written the way it is".

        `scope`: directory of Python (default CWD). `limit`: max symbols to
        populate (default 8; pass 0 for unlimited). `force`: re-draft symbols
        that already have a record. `external`: enable/disable external
        literature search (None = follow `cfg.external.enabled`).

        MUTATES the ledger (one write per symbol). Cost: one draft + one
        ground per symbol (~1–3 s each, network-bound when external). Returns
        a summary including per-symbol drafts, audit, and conflicts."""
        return tools.populate_scope(
            scope, limit=limit, force=force, external=external
        )

    @mcp.tool()
    def populate_symbol(
        symbol: str,
        decision: str,
        rationale: str = "",
        alternatives: list[dict] | None = None,
        depends_on: list[str] | None = None,
        cited: list[dict] | None = None,
        scope: str = ".",
        external: bool | None = None,
    ) -> dict:
        """Record a design decision for ONE symbol, where YOU (the model in
        this conversation) wrote the decision text — no local reasoning model
        is run. This tool handles the **grounding + ledger write**, so the
        decision is still grounded in real evidence and appears in the ledger.

        Use this when the device can't run a local model but you still want an
        evidence-grounded, ledger-recorded decision. You supply the decision
        (you already have the code in context); the tool then:
          1. fingerprints `symbol` (from `scope`),
          2. grounds it in real evidence (local corpus + external literature),
          3. resolves `cited` entries to DOIs,
          4. writes the record (same path as `populate_scope` /
             `record_evidence`).

        Args: `symbol` (qualified name, e.g. `nn.MCDDropout`), `decision`
        (what was chosen), `rationale` (why), `alternatives`
        (`[{"name","why_rejected"}]`), `depends_on` (symbols it builds on),
        `cited` (`[{"name","authors","doi"?}]` — the works it rests on),
        `scope` (dir of Python to fingerprint, default CWD), `external`
        (enable/disable external literature; None = follow config).

        MUTATES the ledger. Cost: retrieval (+ optional network), NO local LLM.
        Prefer this over `populate_scope` when you can't afford a local model;
        `populate_scope` additionally drafts decisions with the local model."""
        return tools.populate_symbol(
            symbol,
            decision,
            rationale=rationale,
            alternatives=alternatives,
            depends_on=depends_on,
            cited=cited,
            scope=scope,
            external=external,
        )

    @mcp.tool()
    def audit_evidence(scope: str = "repo") -> dict:
        """Scan code fingerprints and report decision records whose code or
        cited evidence has drifted: `stale_code` (symbol changed since the
        record was written), `stale_evidence` (a cited chunk is no longer in
        the index), `stale_dependency` (a decision this record transitively
        depends on was invalidated), `orphaned` (symbol removed/renamed). Only
        invalidated records are re-queued. Cost: AST scan + hashing, no LLM,
        ~1–5s on a normal repo."""
        return tools.audit_evidence(scope=scope)

    @mcp.tool()
    def impact_analysis(symbol_or_chunk: str) -> dict:
        """Given a symbol (or an evidence chunk ID / DOI), list every OTHER
        decision record that cites the same evidence OR declares a `depends_on`
        dependency on it. Use before changing a method, class, or API — shows
        which other decisions would inherit a contradiction, transitively.
        Read-only, ~10ms."""
        return tools.impact_analysis(symbol_or_chunk)

    @mcp.tool()
    def export_evidence(format: str = "markdown") -> dict:
        """Render the whole evidence ledger as a readable, diff-able artifact.

        `format` is "markdown" (human-readable) or "json" (machine-readable).
        The returned `content` is meant to be written to a file in your repo
        (e.g. `docs/evidence/ledger.md`) and committed, so `git diff` shows
        exactly which evidence links and decisions changed. READ-ONLY: does
        not mutate the ledger. Cost: full-table scan + render, ~10ms for a
        normal ledger."""
        return tools.export_evidence(format=format)

    # ── Resources ─────────────────────────────────────────────────────────
    @mcp.resource("papers://corpus/stats")
    def corpus_stats_resource() -> dict:
        """Corpus statistics: paper/chunk counts, index presence, embedding dim."""
        return tools.corpus_stats()

    return mcp


def main() -> None:
    """stdio entry point (bound as `evidence-mcp`).

    Accepts an optional `--config <path>` (or reads `EVIDENCE_CONFIG`) to pin
    which project this instance serves. This matters for clients with a *shared*
    MCP gateway (e.g. VS Code): they de-duplicate identical launch commands into
    ONE process serving ALL workspaces. By giving each project a distinct
    `--config` argument we force a separate, correctly-scoped process per
    project, so each workspace's `evidence_*` tools hit *its own* corpus.
    """
    import os
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    argv = sys.argv[1:]
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            os.environ["EVIDENCE_CONFIG"] = argv[i + 1]
    server = create_server()
    server.run()  # stdio transport by default


if __name__ == "__main__":
    main()
