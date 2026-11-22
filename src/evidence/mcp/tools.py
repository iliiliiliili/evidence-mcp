"""Thin tool implementations (new plan: `mcp/tools/` — one fn per tool).

Each function takes explicit parameters (query/limit + the shared `AppContext`
or `Ledger`) and returns a plain, JSON-serializable dict/list. Keeping the logic
here — and OUT of `server.py` — means the tools are unit-testable without the
`mcp` package installed, and the FastMCP layer stays a trivial registration shim.
"""

from __future__ import annotations

import re
from typing import Any

from .. import facade
from ..config import EvidenceConfig
from ..external import crossref, openalex, semantic_scholar
from ..ledger import Ledger


# ─────────────────────────────────────────────────────────────────────────
# Retrieval (stages 3–6)
# ─────────────────────────────────────────────────────────────────────────
def search_corpus(query: str, limit: int = 8, app: facade.AppContext | None = None) -> list[dict]:
    """Hybrid search over the local paper corpus. See facade.search_corpus."""
    return facade.search_corpus(query, limit=limit, app=app)


def expand_via_graph(
    chunk_id: str, relation: str = "SHARES_TOPIC", limit: int = 5,
    app: facade.AppContext | None = None,
) -> list[dict]:
    """Related chunks from OTHER papers via the KG. See facade.expand_via_graph."""
    return facade.expand_via_graph(chunk_id, relation=relation, limit=limit, app=app)


def search_figures(query: str, limit: int = 5, app: facade.AppContext | None = None) -> dict:
    """Visual page retrieval (stage 6, Phase 4).

    Returns a page-image match list when the MUVERA/ColSmol page index has been
    built; otherwise a clear "not available" envelope so the agent knows to fall
    back to `search_corpus`.
    """
    index_dir = app.cfg.resolved_index_dir
    page_index = index_dir / "pages_muvera.faiss"
    if not (page_index.exists() and app.cfg.visual.enabled):
        return {
            "available": False,
            "reason": (
                "Visual page index not built (Phase 4, `visual.enabled`). "
                "Use `search_corpus` for text-based retrieval."
            ),
            "results": [],
        }
    # Real implementation would run the MUVERA ANN + MaxSim here. The scaffold
    # returns a graceful, documented envelope.
    return {
        "available": True,
        "reason": "visual index present",
        "results": [],
        "note": "visual retrieval scaffolded; model inference not wired yet.",
    }


# ─────────────────────────────────────────────────────────────────────────
# Evidence gates (stage 7, 11)
# ─────────────────────────────────────────────────────────────────────────
def check_evidence(
    question: str,
    chunk_ids: list[str],
    intent: str = "content",
    app: facade.AppContext | None = None,
) -> dict:
    """Score an evidence set against the 100-pt rubric. See facade.score_evidence."""
    return facade.score_evidence(question, chunk_ids, intent=intent, app=app)


def verify_claim(claim: str, chunk_id: str, app: facade.AppContext | None = None) -> dict:
    """Rule-based claim↔chunk check. See facade.verify_claim."""
    return facade.verify_claim(claim, chunk_id, app=app)


def get_paper(paper_id: str, app: facade.AppContext | None = None) -> dict | None:
    """Full paper record + section outline + parsed references.

    `paper_id` may be an exact store id or a title substring. The `title` is
    the derived display title, and `references` is the paper's bibliography
    split into individual entries. See facade.get_paper."""
    return facade.get_paper(paper_id, app=app)


def get_chunk(chunk_id: str, app: facade.AppContext | None = None) -> dict | None:
    """The FULL, untruncated text of one chunk + its provenance.

    Use to read an exact passage (a table row, a definition, a reference entry)
    that `search_corpus` returned truncated. Returns the full text, paper/
    section/page, and neighbouring chunks for local context. See facade."""
    return facade.get_chunk(chunk_id, app=app)


def get_references(paper_id: str, app: facade.AppContext | None = None) -> list[str]:
    """The paper's reference list as individual, untruncated entries.

    Structured (split per-entry) and NOT truncated — the clean way to read a
    bibliography, instead of fishing through `search_corpus` chunks. `paper_id`
    may be an exact id or a title substring. See facade.get_references."""
    return facade.get_references(paper_id, app=app)


def get_sections(paper_id: str, app: facade.AppContext | None = None) -> list[dict]:
    """A paper's sections (hierarchical spans) in document order.

    Each section lists its `chunk_ids`, first/last chunk, heading, page, and
    token count — a coarse-to-fine navigation outline. Use with `get_section`
    to read one section as a single untruncated unit. See facade."""
    return facade.get_sections(paper_id, app=app)


def get_section(
    paper_id: str, section_id: str, app: facade.AppContext | None = None
) -> dict | None:
    """One section as a single untruncated unit (all its chunks stitched).

    The hierarchical counterpart of `get_chunk`. `section_id` may be the exact
    id from `get_sections` or a heading substring (e.g. "Method", "Results").
    Returns the full section text in document order plus the chunk ids that
    make it up. See facade.get_section."""
    return facade.get_section(paper_id, section_id, app=app)


def build_chunk_links(
    paper_id: str | None = None,
    app: facade.AppContext | None = None,
    reset: bool = True,
) -> dict:
    """(Re)build the chunk-graph over the corpus (or one paper).

    Emits deterministic links (adjacent "continuation", same-section "see_also")
    and, when a local reasoning model is available, semantic LLM links
    (see_also / depends_on / cites / contrasts / example) with reasons and
    confidence. Fail-soft: a model error just yields fewer links.

    Returns a report: {papers_processed, links_emitted, llm_links,
    deterministic_links, skipped}. See evidence.links.build_chunk_links."""
    from ..links import build_chunk_links as _build

    app = app or facade.get_app()
    cfg = app.cfg
    report = _build(
        app.store, models=app.models, cfg=cfg, paper_id=paper_id, reset=reset
    )
    return report.summary()


def linked_chunks(
    chunk_id: str,
    direction: str = "both",
    min_confidence: float = 0.0,
    app: facade.AppContext | None = None,
) -> dict:
    """Neighbour chunks of `chunk_id`, with the relation + reason of each link.

    `direction`: "out" (chunks this one points to), "in" (chunks pointing to
    it), or "both" (default). Each neighbour carries a preview of its text so
    a client can show the target inline without a second round-trip.
    See evidence.links.linked_chunks."""
    from ..links import linked_chunks as _linked

    app = app or facade.get_app()
    return _linked(
        chunk_id,
        store=app.store,
        direction=direction,
        min_confidence=min_confidence,
    )


def links_for_paper(
    paper_id: str,
    min_confidence: float = 0.0,
    app: facade.AppContext | None = None,
) -> dict:
    """All chunk-links whose source chunk belongs to `paper_id`, confidence-ordered.

    Cross-paper links (e.g. a "cites" edge into another paper's chunk) are
    included, each with a short preview of both endpoints so a reader can
    follow the edge without extra calls. See evidence.links.links_for_paper."""
    from ..links import links_for_paper as _links_paper

    app = app or facade.get_app()
    return _links_paper(
        paper_id, store=app.store, min_confidence=min_confidence
    )


# ─────────────────────────────────────────────────────────────────────────
# External literature (stage 9)
# ─────────────────────────────────────────────────────────────────────────
def search_literature(
    query: str,
    intent: str = "content",
    limit: int = 10,
    cfg: EvidenceConfig | None = None,
) -> list[dict]:
    """Search OUTSIDE the corpus via OpenAlex/S2/Crossref.

    `intent` controls sort/strategy: content|bibliometric → most cited;
    trend → publication-year order; current_world → newest first.

    **Identifier fast-path:** if `query` already contains a DOI or an arXiv ID
    (e.g. "arXiv:2510.05710" or "10.1000/xyz"), it is resolved DIRECTLY to that
    work (Crossref → OpenAlex) instead of a fuzzy keyword search. This is what
    makes niche citations resolvable — keyword search surfaces generic surveys
    for specific prior work, but an identifier pins the exact artifact.

    **Search ≠ citation resolver:** for an open-ended *query* the top result is
    a legitimately "most cited / most relevant" hit and is the right answer.
    When you are instead trying to resolve one *specific* prior citation (e.g. a
    reference-list entry with no identifier), do NOT assume the top result is
    that work — verify the title before trusting it. `review_paper` does exactly
    this: it scores each candidate against the reference title and reports a
    low-confidence match as an unverified `candidate` hint rather than a hit.
    """
    cfg = cfg or EvidenceConfig.load()

    # Fast-path: resolve an explicit DOI / arXiv identifier directly.
    ident = _extract_identifier(query)
    if ident is not None:
        work = _resolve_identifier(ident, cfg)
        if work:
            return [_work_to_citation({**work, "resolved_via": ident,
                                       "match": "identifier"})]

    mailto = cfg.external.openalex_mailto
    s2_key = cfg.external.s2_api_key

    if intent == "current_world":
        sort = "publication_date:desc"
    else:
        sort = "cited_by_count:desc"

    works: list[dict] = []
    if "openalex" in cfg.external.providers:
        try:
            works = openalex.search_works(query, per_page=limit, mailto=mailto, sort=sort)
            for w in works:
                w["source"] = "openalex"
        except Exception:  # noqa: BLE001 - provider is optional
            works = []
    if not works and "s2" in cfg.external.providers:
        try:
            works = semantic_scholar.search_papers(query, limit=limit, api_key=s2_key)
            for w in works:
                w["source"] = "s2"
        except Exception:  # noqa: BLE001
            works = []
    if not works and "crossref" in cfg.external.providers:
        try:
            works = crossref.search(query, rows=limit, mailto=mailto)
            for w in works:
                w["source"] = "crossref"
        except Exception:  # noqa: BLE001
            works = []

    return [_work_to_citation(w) for w in works[:limit]]


def expand_citations(
    doi_or_id: str,
    direction: str = "citing",
    limit: int = 10,
    cfg: EvidenceConfig | None = None,
) -> list[dict]:
    """Forward/backward citations of a paper (OpenAlex + S2 contexts fallback).

    direction: "citing" (works that cite it) | "cited" (works it cites).
    """
    cfg = cfg or EvidenceConfig.load()
    mailto = cfg.external.openalex_mailto
    s2_key = cfg.external.s2_api_key

    works: list[dict] = []
    if direction == "citing":
        if "openalex" in cfg.external.providers:
            try:
                works = openalex.works_citing(doi_or_id, limit=limit, mailto=mailto)
                for w in works:
                    w["source"] = "openalex"
            except Exception:  # noqa: BLE001
                works = []
        if not works and "s2" in cfg.external.providers:
            try:
                works = semantic_scholar.citation_contexts(
                    doi_or_id, limit=limit, api_key=s2_key
                )
                for w in works:
                    w["source"] = "s2"
            except Exception:  # noqa: BLE001
                works = []
    else:  # "cited" / referenced_by
        if "openalex" in cfg.external.providers:
            try:
                works = openalex.referenced_by(doi_or_id, limit=limit, mailto=mailto)
                for w in works:
                    w["source"] = "openalex"
            except Exception:  # noqa: BLE001
                works = []

    return [_work_to_citation(w) for w in works[:limit]]


# ─────────────────────────────────────────────────────────────────────────
# Ingestion (flag-gated, mutating)
# ─────────────────────────────────────────────────────────────────────────
def ingest_paper(url_or_path: str, app: facade.AppContext | None = None) -> dict:
    """Add a PDF path / URL to the corpus. MUTATES — gated by `mcp.allow_ingest`."""
    if not app.cfg.mcp.allow_ingest:
        return {
            "error": (
                "Ingestion disabled. Set `mcp.allow_ingest: true` in config "
                "or EVIDENCE_MCP_ALLOW_INGEST=1."
            )
        }
    from ..ingestion import ingest_item

    return ingest_item(url_or_path, cfg=app.cfg, models=app.models, store=app.store)


# ─────────────────────────────────────────────────────────────────────────
# Full pipeline (heavy)
# ─────────────────────────────────────────────────────────────────────────
def answer_research(question: str, app: facade.AppContext | None = None) -> dict:
    """Run the FULL evidence-gated pipeline.

    Two modes (selected by `mcp.prefer_client_generation`):

    - **bundle mode** (flag on): no local reasoning models are run. Returns a
      raw *evidence bundle* — retrieved chunks, a sufficiency score, and explicit
      guidance — for the CALLING model (e.g. VS Code Copilot) to synthesise a
      cited answer. Use this when the MCP client already has a capable model;
      it avoids running Evidence's local Planner/Writer/Critic stack.

    - **full mode** (flag off, default): runs the local LangGraph pipeline and
      returns a finished, cited `answer`. Use for headless / no-client-LLM use.
    """
    app = app or facade.get_app()
    if app.cfg.mcp.prefer_client_generation:
        return _research_bundle(question, app)

    # Full local pipeline (heavy — many local-model calls).
    from ..graph.builder import Evidence

    tr = Evidence(app.cfg)
    result = tr.ask(question)
    return {
        "mode": "full",
        "answer": result.get("answer", ""),
        "intent": result.get("intent"),
        "citations": result.get("citations", []),
        "citations_dropped": result.get("citations_dropped", []),
        "sufficiency": result.get("sufficiency", {}),
        "evidence_map": result.get("evidence_map", {}),
        "qc_report": result.get("qc_report", {}),
        "sufficiency_decision": result.get("sufficiency_decision"),
    }


# ─────────────────────────────────────────────────────────────────────────
# Scientific review (structured peer-review of ONE paper)
# ─────────────────────────────────────────────────────────────────────────
def review_paper(
    paper_id: str,
    *,
    external: bool | None = None,
    resolve_limit: int = 25,
    cfg: EvidenceConfig | None = None,
) -> dict:
    """Structured scientific review of a single corpus paper.

    Follows a fixed five-part structure:

      1. **summary** — what the paper is about (problem, contribution, method,
         headline result).
      2. **strengths** / **weaknesses** — the paper's merits and shortcomings.
      3. **incorrect_statements** — statements that look factually wrong,
         internally contradictory, or misstate a method/theorem (each flagged
         with a ``confidence`` of high/medium/low and a ``chunk_id`` — these
         are *candidates to check*, not verdicts).
      4. **undefined_terms** — terms/acronyms used without a definition
         (model-identified + a deterministic acronym scan).
      5. **references** — the paper's COMPLETE reference list (parsed from the
         document, not model-enumerated, so nothing is silently dropped), each
         with a ``relevance`` (high/medium/low) and ``role``
         (method/comparison/dataset/…). Relevance/role are best-effort model
         annotations layered onto the deterministic list. When `external` is
         enabled each is checked against OpenAlex/S2/Crossref. Because a search
         is not a citation resolver, ``resolved.found`` is set only when a
         candidate's title matches the reference; a plausible-but-unconfirmed
         top hit is reported as an unverified ``resolved.candidate`` hint
         (with ``resolved.match_confidence``) rather than a false "found".
      6. **author_conflicts** — a deterministic (no-model) list flagging any
         reference whose author string overlaps the paper's OWN authors,
         surfacing the "same-group citing its own prior work" issue
         automatically. Each entry carries ``ref_index``, ``ref_title``,
         ``overlapping_authors``, and ``kind`` (``self-citation`` when all of
         the paper's authors appear in the reference, ``author-overlap`` when
         only some do).

    `undefined_terms` is curated: banner/watermark all-caps noise
    (e.g. "FOR REVIEW ONLY", paper-type acronyms) is suppressed so the list
    holds only genuinely undefined terms.

    `paper_id` may be an exact store id or a title substring (most-chunked match
    wins). `external` overrides `cfg.external.enabled` for reference lookups
    (None = follow config). `resolve_limit` caps how many references are
    checked externally (high relevance first). Read-only: never mutates the
    corpus or ledger. Cost: two local model calls (+ optional network); a
    deterministic fallback keeps it working with no model.
    """
    from ..review import review_paper as _review

    return _review(
        paper_id,
        external=external,
        resolve_limit=resolve_limit,
        cfg=cfg,
    )


def _research_bundle(question: str, app: facade.AppContext) -> dict:
    """Assemble an evidence bundle for the calling model to generate from.

    Thin + cheap: retrieval + a deterministic sufficiency score only. No local
    generation models are invoked — the client's model writes the answer.
    """
    limit = max(app.cfg.mcp.default_search_limit, 8)
    chunks = search_corpus(question, limit=limit, app=app)

    # Deterministic sufficiency over the retrieved set (no LLM).
    chunk_ids = [c["chunk_id"] for c in chunks]
    try:
        sufficiency = facade.score_evidence(
            question, chunk_ids, intent="content", app=app
        )
    except Exception:  # noqa: BLE001 - scoring is best-effort
        sufficiency = {"error": "sufficiency scoring unavailable"}

    return {
        "mode": "bundle",
        "question": question,
        "instruction": (
            "Synthesise a clear, well-organised answer to the question from the "
            "chunks below. Cite specific chunk_ids inline (e.g. [p3::c12]). "
            "Only state claims the chunks support; if the evidence is "
            "insufficient, say what is missing and use `search_corpus` with a "
            "different query before answering. Do NOT invent citations."
        ),
        "sufficiency": sufficiency,
        "chunks": chunks,
        "next_steps": [
            "If `sufficiency.verdict` is not 'sufficient', call `search_corpus` "
            "with a reformulated query.",
            "Verify any load-bearing claim with `verify_claim(claim, chunk_id)`.",
            "Record the decision with `record_evidence(symbol, decision, "
            "evidence=[...])` once settled.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# Evidence ledger (cache-first layer)
# ─────────────────────────────────────────────────────────────────────────
def lookup_evidence(symbol: str, ledger: Ledger | None = None) -> dict | None:
    """Stored decision record for a symbol, or None. CHECK THIS FIRST."""
    from ..ledger import get_ledger

    ledger = ledger or get_ledger()
    return ledger.lookup(symbol)


def record_evidence(
    symbol: str,
    decision: str,
    evidence: list[dict] | None = None,
    alternatives: list[dict] | None = None,
    *,
    rationale: str = "",
    code_fingerprint: str = "",
    depends_on: list[str] | None = None,
    cited: list[dict] | None = None,
    ledger: Ledger | None = None,
) -> dict:
    """Write/update the decision record for a symbol. MUTATES the ledger.

    `depends_on` lists OTHER decisions this one is derived from (e.g. "use
    PyTorch" depends on "use Python"). This is what lets `impact_analysis` /
    `audit_evidence` propagate a root decision's change transitively through
    every decision built on it — even when they cite different evidence.

    `cited` lists the papers/methods the decision *rests on* (name, authors,
    optional DOI) — kept separate from `evidence` so the reviewer can see what
    you intended to cite even when a live lookup did not resolve a DOI."""
    from ..ledger import get_ledger

    ledger = ledger or get_ledger()
    return ledger.record(
        symbol,
        decision,
        evidence=evidence,
        alternatives=alternatives,
        rationale=rationale,
        code_fingerprint=code_fingerprint,
        depends_on=depends_on,
        cited=cited,
    )


def populate_symbol(
    symbol: str,
    decision: str,
    rationale: str = "",
    alternatives: list[dict] | None = None,
    depends_on: list[str] | None = None,
    cited: list[dict] | None = None,
    *,
    scope: str = ".",
    external: bool | None = None,
    code_fingerprint: str = "",
    cfg: Any | None = None,
    ledger: Ledger | None = None,
) -> dict:
    """Record a design decision for ONE symbol, where the *caller* (you) wrote
    the decision text, and this tool handles **grounding + the ledger write**.

    Use this when you do NOT want to run a local reasoning model (e.g. the
    device can't afford one) but you still want the decision grounded in real
    evidence and recorded in the ledger. You (the model driving the
    conversation) supply `decision` / `rationale` / `alternatives` /
    `depends_on` / `cited` — you already have the code in context. This tool
    then:

      1. computes the code fingerprint for `symbol` (from `scope`),
      2. grounds it in real evidence — local corpus + external literature
         (in parallel when `external` is enabled) — attaching only refs that a
         live search actually returned,
      3. resolves any `cited` entries to DOIs,
      4. writes the record via the same `record()` path (so audit / impact /
         `export_evidence` all see it identically).

    `scope`: directory of Python to fingerprint `symbol` against (default CWD).
    `external`: enable/disable external literature (None = follow config).

    MUTATES the ledger. Cost: retrieval (+ optional network), no local LLM.
    Contrast with `populate_scope`, which ALSO drafts the decision using the
    local reasoning model (autonomous, one call for a whole folder)."""
    from ..config import EvidenceConfig
    from ..ledger import get_ledger
    from ..ledger.fingerprint import scan_repo
    from ..populate import PopulateService

    cfg = cfg or EvidenceConfig.load()
    ledger = ledger or get_ledger(cfg)
    svc = PopulateService(cfg, ledger=ledger)

    # 1) Fingerprint (so `audit` can track drift for this symbol).
    fp = code_fingerprint
    if not fp:
        try:
            fp = scan_repo(scope).get(symbol, "")
        except Exception:  # noqa: BLE001 - fingerprinting is best-effort
            fp = ""

    # 2-3) Ground (local + external, in parallel) + resolve `cited`.
    evidence, notes = svc._ground(
        symbol, decision, external=external, cited=cited or []
    )

    # 4) Record (same path as populate_scope / record_evidence).
    rec = ledger.record(
        symbol,
        decision,
        evidence=evidence,
        alternatives=alternatives or [],
        rationale=rationale,
        code_fingerprint=fp,
        depends_on=depends_on or [],
        cited=cited or [],
    )
    return {**rec, "notes": notes, "grounded": bool(evidence)}


def populate_scope(
    scope: str = ".",
    *,
    limit: int = 8,
    force: bool = False,
    external: bool | None = None,
    cfg: Any | None = None,
) -> dict:
    """Autonomously draft, ground, and record design decisions for a code scope.

    Scans `scope` (a directory of Python) for top-level symbols, uses the
    local reasoning model to draft a decision record per symbol (decision,
    rationale, alternatives, depends_on, and `cited` — the papers/methods
    the decision rests on), grounds each in real evidence (local corpus +
    external literature, in parallel when `external` is enabled), and records
    the result in the ledger.

    MUTATES the ledger (one write per symbol). Cost: one draft + one ground
    per symbol (~1–3 s each depending on model + network). Use this instead of
    calling `record_evidence` symbol-by-symbol when the user asks to
    "populate", "draft decisions", or "record why this code is the way it is".

    Returns a summary: scanned / recorded / skipped / audit / conflicts.
    For each recorded symbol, the returned entry has the same shape as the
    output of `record_evidence` plus a `notes` list (e.g. warnings about
    local-only evidence when external was off).
    """
    from ..config import EvidenceConfig
    from ..ledger import get_ledger
    from ..populate import PopulateService

    cfg = cfg or EvidenceConfig.load()
    ledger = get_ledger(cfg)
    svc = PopulateService(cfg, ledger=ledger)
    report = svc.run(
        scope,
        max_symbols=max(1, limit),
        force=force,
        dry_run=False,
        external=external,
    )
    return {
        "scope": report.scope,
        "scanned": report.scanned,
        "recorded": [
            {
                "symbol": pr.symbol,
                "decision": pr.decision,
                "rationale": pr.rationale,
                "alternatives": pr.alternatives,
                "depends_on": pr.depends_on,
                "cited": pr.cited,
                "evidence": pr.evidence,
                "grounded": pr.grounded,
                "notes": pr.notes,
            }
            for pr in report.recorded
        ],
        "skipped": report.skipped,
        "audit": report.audit,
        "conflicts": [
            {"kind": c.kind, "symbols": c.symbols, "detail": c.detail}
            for c in report.conflicts
        ],
    }


def audit_evidence(
    scope: str = "repo",
    ledger: Ledger | None = None,
    app: Any | None = None,
) -> dict:
    """Scan code fingerprints; list stale_code / stale_evidence / orphaned."""
    from ..ledger import get_ledger

    ledger = ledger or get_ledger()
    app = app or facade.get_app()
    cfg = app.cfg
    return ledger.audit(
        scope,
        store=app.store,
        algo=cfg.ledger.fingerprint_algo,
        window=cfg.ledger.fingerprint_window,
    )


def impact_analysis(symbol_or_chunk: str, ledger: Ledger | None = None) -> dict:
    """All OTHER decisions citing the same evidence (change propagation)."""
    from ..ledger import get_ledger

    ledger = ledger or get_ledger()
    return ledger.impact(symbol_or_chunk)


def export_evidence(
    format: str = "markdown",
    ledger: Ledger | None = None,
) -> dict:
    """Render the whole ledger as a readable, diff-able artifact.

    `format` is "markdown" (human-readable) or "json" (machine-readable).
    Returns `{"format": ..., "content": ..., "record_count": n}` so a consumer
    can write `content` to a file in their repo (e.g. `docs/evidence/ledger.md`)
    and commit it — giving a `git diff`-reviewable view of the evidence links
    and the reasoning behind each decision. READ-ONLY: does not mutate the
    ledger."""
    from ..ledger import get_ledger

    ledger = ledger or get_ledger()
    if format == "json":
        content = ledger.export_json()
    elif format == "markdown":
        content = ledger.export_markdown()
    else:
        raise ValueError(f"unknown format: {format!r} (use markdown|json)")
    return {
        "format": format,
        "record_count": ledger.count_records(),
        "content": content,
    }


# ─────────────────────────────────────────────────────────────────────────
# Resources
# ─────────────────────────────────────────────────────────────────────────
def corpus_stats(app: facade.AppContext | None = None) -> dict:
    return facade.corpus_stats(app=app)


def paper_outline(paper_id: str, app: facade.AppContext | None = None) -> dict | None:
    rec = facade.get_paper(paper_id, app=app)
    if rec is None:
        return None
    return {"paper_id": rec["paper_id"], "title": rec["title"], "outline": rec["outline"]}


# ─────────────────────────────────────────────────────────────────────────
# Shared shape helpers
# ─────────────────────────────────────────────────────────────────────────
# A DOI: `10.<digits>/<anything>` (optionally prefixed with doi.org/...).
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'\)\]\}<>]+", re.IGNORECASE)
# An arXiv id: `arXiv:NNNN.NNNN[.vN]` or the bare `NNNN.NNNN[.vN]` (20+ years).
_ARXIV_RE = re.compile(
    r"(?:arXiv:\s*)?(\d{4}\.\d{4,5})(?:\.v\d+)?", re.IGNORECASE
)


def _extract_identifier(query: str) -> str | None:
    """Pull an explicit DOI or arXiv id out of a citation string, if present.

    Returns a normalized identifier (a bare DOI, or a bare arXiv id) or None.
    A DOI is preferred when both appear (it is unambiguous).
    """
    q = (query or "").strip()
    if not q:
        return None
    doi = _DOI_RE.search(q)
    if doi:
        return doi.group(0).rstrip(".,;:")
    arx = _ARXIV_RE.search(q)
    if arx:
        return arx.group(1)
    return None


def _resolve_identifier(ident: str, cfg: EvidenceConfig | None = None) -> dict | None:
    """Resolve a DOI or arXiv id to a normalized work dict (fail-soft → None).

    Tries Crossref, then OpenAlex, then Semantic Scholar. Each provider is
    best-effort; the first hit that returns a title wins.
    """
    doi = ident if ident.startswith("10.") else None
    arxiv = None
    if not doi:
        # Only treat it as an arXiv id if it looks like one (digits + dot).
        if re.fullmatch(r"\d{4}\.\d{4,5}", ident):
            arxiv = ident
    mailto = cfg.external.openalex_mailto if cfg else None
    s2_key = cfg.external.s2_api_key if cfg else None

    # 1) Crossref (DOI).
    if doi:
        try:
            w = crossref.get_work(doi, mailto=mailto)
            if w and (w.get("title") or w.get("id")):
                w["source"] = "crossref"
                return w
        except Exception:  # noqa: BLE001 - best-effort
            pass
    # 2) OpenAlex (DOI).
    if doi:
        try:
            w = openalex.get_work(f"doi:{doi}", mailto=mailto)
            if w and (w.get("title") or w.get("id")):
                w["source"] = "openalex"
                return w
        except Exception:  # noqa: BLE001
            pass
    # 3) Semantic Scholar (arXiv id).
    if arxiv:
        try:
            w = semantic_scholar.get_paper(arxiv, api_key=s2_key)
            if w and (w.get("title") or w.get("id")):
                w["source"] = "s2"
                if not w.get("doi"):
                    w["doi"] = f"10.48550/arXiv.{arxiv}"
                return w
        except Exception:  # noqa: BLE001
            pass
    return None


def _work_to_citation(w: dict[str, Any]) -> dict:
    """Normalize an external work record into the plan's citation shape."""
    cited_by = (
        w.get("cited_by_count")
        or w.get("citations")
        or w.get("is-referenced-by-count")
    )
    out = {
        "paper": w.get("title") or w.get("display_name") or "",
        "year": w.get("year"),
        "doi": w.get("doi"),
        "cited_by": cited_by,
        "oa_pdf": w.get("oa_pdf") or w.get("oa_url") or w.get("pdf_url"),
        "abstract": (w.get("abstract") or "")[:500],
        "source": w.get("source", ""),
    }
    # Preserve fast-path provenance (identifier resolution) when present.
    if w.get("resolved_via"):
        out["resolved_via"] = w["resolved_via"]
    if w.get("match"):
        out["match"] = w["match"]
    return out
