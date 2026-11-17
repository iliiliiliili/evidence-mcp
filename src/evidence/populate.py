"""Autonomous `evidence populate` — draft + ground + record design decisions.

Given a code scope (a directory of Python), this module:

  1. **Scans** top-level symbols (functions / classes / module constants)
     using the *same* fingerprinting the ledger uses for drift detection, so a
     freshly-recorded symbol audits as `fresh`.
  2. **Drafts** a design-decision record per symbol with the local reasoning
     model (strict JSON): what was chosen, why, the alternatives considered,
     and what it depends on.
  3. **Grounds** each draft in real evidence via corpus retrieval
     (`search_corpus`) and, when the corpus is empty and external search is
     enabled, the academic APIs (`search_literature`). Only *real* chunk_ids
     and DOIs that came back from a live search are attached — nothing is ever
     invented.
  4. **Records** the decision in the ledger (`Ledger.record`).
  5. **Audits** (drift) and computes **conflicts** (stale / orphaned /
     un-grounded / shared-evidence) for the interactive resolution pass.

The module is Typer-free and dependency-light so it unit-tests against a temp
ledger and a fake Ollama, exactly like the rest of the suite.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import EvidenceConfig
from .ledger.fingerprint import fingerprint_source
from .ledger.store import Ledger
from .models.ollama import Models

logger = logging.getLogger(__name__)

_IGNORE_DIRS = (
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".ruff_cache", "build", "dist", "site-packages",
)


# ── data shapes ──────────────────────────────────────────────────────────────
@dataclass
class SymbolCandidate:
    """One top-level symbol discovered by the scan."""

    name: str            # qualified ``module.Name`` (ledger key)
    module: str
    kind: str            # "function" | "class" | "const"
    docstring: str
    source: str          # the symbol's own source lines
    fingerprint: str


@dataclass
class PopulatedRecord:
    """The outcome of populating one symbol."""

    symbol: str
    decision: str = ""
    rationale: str = ""
    alternatives: list[dict] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    cited: list[dict] = field(default_factory=list)
    grounded: bool = False
    existed_before: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class Conflict:
    """A decision the user should look at (questionable or possibly wrong)."""

    kind: str            # stale_code | stale_evidence | orphaned | no_evidence | shared_evidence
    symbols: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class PopulateReport:
    scope: str
    scanned: int = 0
    recorded: list[PopulatedRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    model: str = ""
    dry_run: bool = False
    external: bool | None = None  # effective external search setting for this run

    def as_text(self) -> str:
        """A dependency-free plain-text summary (used by dry-run + tests)."""
        lines = [
            f"evidence populate — scope: {self.scope}"
            + ("  [dry-run]" if self.dry_run else ""),
            f"  model:      {self.model or '(none)'}",
            f"  scanned:    {self.scanned} symbol(s)",
            f"  recorded:   {len(self.recorded)}",
            f"  skipped:    {len(self.skipped)}",
        ]
        a = self.audit or {}
        if a:
            lines.append(
                f"  audit:      fresh={a.get('fresh', 0)} "
                f"stale_code={a.get('stale_code', 0)} "
                f"stale_evidence={a.get('stale_evidence', 0)} "
                f"orphaned={a.get('orphaned', 0)} "
                f"stale_dependency={a.get('stale_dependency', 0)}"
            )
        lines.append(f"  conflicts:  {len(self.conflicts)}")
        for c in self.conflicts:
            lines.append(f"    - [{c.kind}] {c.detail}")
        return "\n".join(lines)


# ── scanning ─────────────────────────────────────────────────────────────────
def _symbol_lines(node: ast.AST, source_lines: list[str]) -> list[str]:
    """Match ``fingerprint._symbol_lines``: the symbol's own source lines."""
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None) or start
    return source_lines[max(start - 1, 0):end]


def scan_symbols(
    root: str | Path,
    *,
    algo: str = "sha256",
    window: int = 120,
    max_symbols: int = 40,
) -> list[SymbolCandidate]:
    """Enumerate top-level symbols under ``root`` with a matching fingerprint.

    Mirrors ``scan_repo``'s file walk (same ignore dirs, ``.py`` only) but also
    captures each symbol's source, docstring, and kind so a decision can be
    drafted. Never imports the target code. Returns at most ``max_symbols``.
    """
    root = Path(root)
    out: list[SymbolCandidate] = []
    if not root.exists():
        return out
    cap = max_symbols if max_symbols > 0 else None

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix == ".py"
        and not any(part in _IGNORE_DIRS for part in p.parts)
    ]
    for path in files:
        if cap is not None and len(out) >= cap:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue

        rel = path.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts) or path.stem

        for node in tree.body:
            if cap is not None and len(out) >= cap:
                break
            name = kind = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name, kind = node.name, "function"
            elif isinstance(node, ast.ClassDef):
                name, kind = node.name, "class"
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        name, kind = t.id, "const"
                        break
            if not name:
                continue

            qual = f"{module}.{name}" if module else name
            src = "\n".join(_symbol_lines(node, lines))
            fp = fingerprint_source(src, symbol=qual, algo=algo, window=window) or ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = (ast.get_docstring(node) or "").strip()
            else:
                doc = ""
            out.append(
                SymbolCandidate(
                    name=qual,
                    module=module,
                    kind=kind,
                    docstring=doc,
                    source=src,
                    fingerprint=fp,
                )
            )
    return out


# ── service ──────────────────────────────────────────────────────────────────
class PopulateService:
    """Drafts, grounds, and records decisions for a scanned scope."""

    def __init__(
        self,
        cfg: EvidenceConfig,
        *,
        models: Models | None = None,
        ledger: Ledger | None = None,
    ):
        self.cfg = cfg
        self.models = models or Models(cfg)
        if ledger is None:
            from .ledger import get_ledger

            ledger = get_ledger(cfg)
        self.ledger = ledger
        self._app = None  # lazy facade.AppContext for search_corpus

    # -- lazy app (only needed for grounding) ------------------------------
    def _app_or_none(self):
        if self._app is not None:
            return self._app
        try:
            from . import facade

            self._app = facade.get_app(self.cfg)
        except Exception as exc:  # noqa: BLE001 - no index / no langgraph
            logger.warning("populate: corpus app unavailable (%s)", exc)
            self._app = None
        return self._app

    # -- drafting ----------------------------------------------------------
    def _local_papers(self, query: str, limit: int = 3) -> list[dict]:
        """Best-effort local-corpus search: which local papers does this code
        relate to?

        Used to give the drafter the *specific* papers the code implements, so
        it cites those (primary) rather than a famous same-name method (e.g. it
        must not name Kingma & Welling's VAE paper when the code faithfully
        implements a local "VNNs" paper). Returns [] when no index is present
        (client-draft / no-Ollama / no-corpus setups) — the drafter then simply
        has no local context and cites well-known works.
        """
        try:
            return self._run_local_only(query)[:limit]
        except Exception:  # noqa: BLE001 - context is best-effort
            return []

    def _draft(self, sym: SymbolCandidate, *, local_papers: list[dict] | None = None) -> dict:
        """Ask the reasoning model for a strict-JSON decision record.

        Returns a normalized dict, or a dict with ``_error`` when the model is
        unavailable / returned nothing usable — in which case the caller
        should *not* fabricate a record.

        The draft also asks the model for a ``cited`` list — the well-known
        papers / methods this decision relies on — so grounding can resolve
        them to real DOIs instead of falling back to the local corpus alone.

        When ``local_papers`` is supplied, the drafter is told the specific
        local paper(s) the code implements and instructed to cite those as
        primary (over a generic method-origin paper that happens to share the
        method name).
        """
        model = self.cfg.ollama.reasoning
        system = (
            "You are a meticulous software architect documenting WHY code is "
            "written the way it is. Respond with STRICT JSON only — no prose, "
            "no markdown, no code fences, no trailing commas."
        )
        src = "\n".join(sym.source.splitlines()[:40])

        local_block = ""
        if local_papers:
            lines = []
            for p in local_papers:
                title = p.get("paper") or "(untitled)"
                year = f" ({p['year']})" if p.get("year") else ""
                doi = f" DOI={p['doi']}" if p.get("doi") else ""
                lines.append(f"  - {title}{year}{doi}")
            local_block = (
                "\nLocal corpus papers most relevant to THIS code (the code\n"
                "almost certainly implements or extends one of these):\n"
                + "\n".join(lines)
                + "\n"
            )

        user = (
            f"Record the design decision behind this symbol.\n\n"
            f"Symbol: {sym.name}\n"
            f"Kind: {sym.kind}\n"
            f"Docstring: {sym.docstring or '(none)'}\n"
            f"Source (truncated):\n{src}\n"
            f"{local_block}\n"
            "Return JSON with exactly these keys:\n"
            '{\n'
            '  "decision": "one sentence: what was chosen",\n'
            '  "rationale": "1-3 sentences: why, in terms of requirements it serves",\n'
            '  "alternatives": [{"name": "rejected option", "why_rejected": "why"}],\n'
            '  "depends_on": ["symbols or frameworks this builds on"],\n'
            '  "cited": [\n'
            '    {"name": "paper title or short name of the defining work",'
            ' "authors": "First Author et al." , "doi": "10.xxxx/... (optional)"}\n'
            '  ]\n'
            "}\n"
            "Rules:\n"
            "- Infer the decision from the code; never invent facts.\n"
            "- `cited` lists the papers, theorems, or well-known methods this\n"
            "  decision relies on. List them MOST-SPECIFIC-FIRST: the paper the\n"
            "  code actually implements goes first. If a 'Local corpus papers'\n"
            "  list is given and the code matches one of them, THAT paper is the\n"
            "  primary citation — do NOT substitute a more famous paper that only\n"
            "  shares the method name (e.g. do not cite the VAE paper when the\n"
            "  code implements a specific local 'VNNs' paper; the VAE paper may\n"
            "  still be listed SECOND as the origin of the reparameterization\n"
            "  trick). Name works even if not in the corpus, so grounding can\n"
            "  look them up. Leave `doi` out if you do not know it exactly —\n"
            "  never fabricate a DOI.\n"
            "- Keep `depends_on` to symbols in this codebase or well-known\n"
            "  frameworks."
        )
        raw = self.models.complete_json(
            user, model=model, system=system, temperature=0.0, fallback=None
        )
        if not isinstance(raw, dict) or not str(raw.get("decision", "")).strip():
            return {"_error": "model returned no usable decision"}

        def _as_list(v) -> list:
            return v if isinstance(v, list) else []

        cited_raw = _as_list(raw.get("cited"))
        cited = [
            {
                "name": str(c.get("name") or "").strip(),
                "authors": str(c.get("authors") or "").strip(),
                "doi": str(c.get("doi") or "").strip(),
            }
            for c in cited_raw
            if isinstance(c, dict) and (c.get("name") or c.get("doi"))
        ]

        return {
            "decision": str(raw["decision"]).strip(),
            "rationale": str(raw.get("rationale", "")).strip(),
            "alternatives": [a for a in _as_list(raw.get("alternatives")) if isinstance(a, dict)],
            "depends_on": [str(x) for x in _as_list(raw.get("depends_on")) if str(x).strip()],
            "cited": cited,
        }

    # -- grounding ---------------------------------------------------------
    def _ground(
        self,
        symbol: str,
        decision: str,
        *,
        external: bool | None = None,
        cited: list[dict] | None = None,
        local_papers: list[dict] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Attach ONLY real evidence found by a live search (or none).

        Strategy (order of precedence for the final list, deduped by ref_key):

        1. **cited resolution** — each entry the model emitted in ``cited``
           is looked up via the literature APIs (name+authors → DOI);
           entries that already carry a DOI are used verbatim.
        2. **local corpus** — offline, preferred when available.
        3. **external literature** — the generic ``<symbol> <decision>``
           query, used as a supplement.

        Steps 2 and 3 run in **parallel** when ``external`` is enabled, so
        the local corpus no longer shadows external references. Only DOIs and
        chunk_ids that came back from a live search are attached — nothing is
        ever invented.

        ``local_papers``, when supplied, is a pre-fetched local-corpus result
        (e.g. the one the drafter already used) reused here to avoid a second
        identical search.
        """
        evidence: list[dict] = []
        seen_keys: set[str] = set()
        notes: list[str] = []
        works_by_doi: dict[str, dict] = {}
        query = f"{symbol.split('.')[-1]} {decision}".strip()

        external_enabled = self.cfg.external.enabled if external is None else external

        def _add(ref_type: str, ref_key: str, source: str, title: str = "") -> None:
            if not ref_key or ref_key in seen_keys:
                return
            seen_keys.add(ref_key)
            evidence.append(
                {"ref_type": ref_type, "ref_key": ref_key,
                 "source": source, "title": title}
            )

        def _remember_work(doi: str | None, w: dict | None) -> None:
            """Remember the raw provider work for a DOI, so we can save it
            (pdf_url / abstract / authors) after the evidence list is built."""
            if not doi or not w:
                return
            d = doi.strip().lower()
            if d.startswith("https://doi.org/"):
                d = d[len("https://doi.org/"):]
            if d and d not in works_by_doi:
                works_by_doi[d] = w

        # 1) Cited resolution (model knowledge → DOI).
        if cited:
            for c in cited:
                doi = (c.get("doi") or "").strip()
                if doi:
                    _add("doi", doi, "cited", c.get("name", ""))
                    _remember_work(doi, {"doi": doi, "title": c.get("name", ""),
                                          "authors": c.get("authors", "")})
                    continue
                name = c.get("name", "")
                if not name:
                    continue
                q = f"{name} {c.get('authors', '')}".strip()
                if external_enabled:
                    works = self._run_external_only(q)[:1]
                    if works and works[0].get("doi"):
                        w = works[0]
                        _add("doi", w["doi"], w.get("source", "doi"),
                            w.get("title", name))
                        _remember_work(w["doi"], w)
                        continue
                # No live match (or external off): keep as a citation-only
                # ref so the reviewer can see what the model wanted to cite.
                _add("citation", name, "unresolved", c.get("authors", ""))

        # 2+3) Local corpus + external literature (in parallel when external).
        # Each search is a small, patchable method so tests can stub them and
        # production stays side-effect-free in the failure paths.
        #
        # Local: a pre-fetched `local_papers` (from the drafter) is reused when
        # given, so we never run the same local search twice.
        # External: always run when enabled, independent of local.
        local_results: list[dict] = []
        ext_results: list[dict] = []

        def _local() -> list[dict]:
            return local_papers if local_papers is not None else self._run_local_only(query)

        def _ext() -> list[dict]:
            return self._run_external_only(query)

        if external_enabled:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as ex:
                f_local = ex.submit(_local)
                f_ext = ex.submit(_ext)
                local_results = f_local.result() or []
                ext_results = f_ext.result() or []
        else:
            local_results = _local()

        for c in local_results[:3]:
            cid = c.get("chunk_id")
            if cid:
                _add("chunk", cid, "local", c.get("paper", ""))
            doi = c.get("doi")
            if doi:
                _add("doi", doi, c.get("source", "local"),
                     c.get("paper", ""))

        for w in ext_results[:3]:
            doi = w.get("doi")
            if doi:
                _add("doi", doi, w.get("source", "doi"), w.get("title", ""))
                _remember_work(doi, w)

        if not evidence:
            notes.append("no supporting evidence found (un-grounded decision)")

        # Save online references into the local corpus (flag-aware).
        # Best-effort: any failure is logged and never blocks grounding.
        self._save_online_refs(works_by_doi)
        return evidence, notes

    def _save_online_refs(self, works_by_doi: dict[str, dict]) -> None:
        """Persist each external DOI we grounded to, into the local corpus.

        Controlled by ``cfg.external.save_online_to_corpus`` (default True) —
        when True and a PDF URL is available the full document is downloaded
        and indexed, so the reference resolves to a readable
        ``Paper · Section`` offline. When False, only the title+section (and
        whatever metadata the provider gave us) is stored.
        """
        if not works_by_doi:
            return
        app = self._app_or_none()
        if app is None:
            return
        try:
            from .external.online_corpus import save_online_ref
        except Exception as exc:  # noqa: BLE001
            logger.warning("online_corpus module unavailable: %s", exc)
            return
        saved = 0
        for doi, w in works_by_doi.items():
            try:
                report = save_online_ref(
                    w, cfg=self.cfg, models=self.models, store=app.store
                )
                if report.get("saved"):
                    saved += 1
                if report.get("mode") in ("full", "meta"):
                    logger.info(
                        "saved online ref %s (%s, %d chunks)",
                        doi, report["mode"], report.get("chunks", 0),
                    )
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning("save_online_ref(%s) failed: %s", doi, exc)
        if saved:
            logger.info("populated %d online reference(s) into the corpus", saved)

    def _run_local_only(self, query: str) -> list[dict]:
        """Local corpus search with no external fallback (used when external off)."""
        app = self._app_or_none()
        if app is None:
            return []
        try:
            from . import facade

            return facade.search_corpus(query, limit=5, app=app)
        except Exception as exc:  # noqa: BLE001
            # Keep the note so the caller can see it was attempted.
            logger.warning("corpus search failed: %s", exc)
            return []

    def _run_external_only(self, query: str) -> list[dict]:
        """External literature search (OpenAlex/S2/Crossref).

        Patchable: tests stub this to return canned results without touching
        the network. Fail-soft: any provider error is logged and returns [].
        """
        try:
            from .mcp.tools import search_literature

            return search_literature(query, limit=5, cfg=self.cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("literature search failed: %s", exc)
            return []

    # -- populate one / many ----------------------------------------------
    def populate_symbol(
        self,
        sym: SymbolCandidate,
        *,
        force: bool = False,
        dry_run: bool = False,
        external: bool | None = None,
    ) -> PopulatedRecord:
        existed = self.ledger.lookup(sym.name) is not None
        rec = PopulatedRecord(symbol=sym.name, existed_before=existed)

        # Which local papers does this code relate to? Give the drafter the
        # *specific* ones (so it cites them as primary, not a famous same-name
        # method) and reuse the result in grounding to avoid a second search.
        lp_query = f"{sym.name.split('.')[-1]} {sym.docstring or ''}".strip()
        local_papers = self._local_papers(lp_query)

        draft = self._draft(sym, local_papers=local_papers)
        if "_error" in draft:
            rec.notes.append(draft["_error"])
            return rec

        rec.decision = draft["decision"]
        rec.rationale = draft["rationale"]
        rec.alternatives = draft["alternatives"]
        rec.depends_on = draft["depends_on"]
        cited = draft.get("cited") or []
        rec.cited = cited
        effective_external = (
            self.cfg.external.enabled if external is None else external
        )
        rec.evidence, gnotes = self._ground(
            sym.name, rec.decision,
            external=effective_external, cited=cited, local_papers=local_papers,
        )
        rec.notes.extend(gnotes)
        rec.grounded = bool(rec.evidence)
        # Loud warning when a decision is local-only and external was off:
        # this is exactly the MCD/BBB failure mode — the method has a
        # defining paper we did not pull in.
        sources = {e.get("source", "") for e in rec.evidence}
        if rec.evidence and sources <= {"local"} and not effective_external:
            rec.notes.append(
                "⚠ local-only evidence — re-run with `--external` to pull "
                "in the defining paper(s)"
            )

        if not dry_run:
            self.ledger.record(
                sym.name,
                rec.decision,
                evidence=rec.evidence,
                alternatives=rec.alternatives,
                rationale=rec.rationale,
                code_fingerprint=sym.fingerprint,
                depends_on=rec.depends_on,
                cited=cited,
            )
        return rec

    def run(
        self,
        root: str | Path,
        *,
        max_symbols: int = 8,
        force: bool = False,
        dry_run: bool = False,
        external: bool | None = None,
    ) -> PopulateReport:
        root = Path(root).expanduser().resolve()
        algo = self.cfg.ledger.fingerprint_algo
        window = self.cfg.ledger.fingerprint_window
        effective_external = (
            self.cfg.external.enabled if external is None else external
        )
        report = PopulateReport(
            scope=str(root),
            model=self.cfg.ollama.reasoning,
            dry_run=dry_run,
            external=effective_external,
        )

        symbols = scan_symbols(
            root, algo=algo, window=window, max_symbols=max(1, max_symbols)
        )
        report.scanned = len(symbols)

        for sym in symbols:
            existing = self.ledger.lookup(sym.name)
            if existing is not None and not force and not dry_run:
                report.skipped.append(sym.name)
                continue
            report.recorded.append(
                self.populate_symbol(sym, force=force, dry_run=dry_run, external=external)
            )

        # Drift audit over the whole scope (reflects the ledger as it stands).
        app = self._app_or_none()
        store = app.store if app is not None else None
        try:
            report.audit = self.ledger.audit(
                str(root), store=store, algo=algo, window=window
            )
        except Exception as exc:  # noqa: BLE001
            report.audit = {"error": str(exc)}

        report.conflicts = self._find_conflicts(report)
        return report

    # -- conflict detection -------------------------------------------------
    def _find_conflicts(self, report: PopulateReport) -> list[Conflict]:
        conflicts: list[Conflict] = []

        for item in (report.audit.get("needs_reanalysis") or []):
            conflicts.append(
                Conflict(kind=item["status"], symbols=[item["symbol"]],
                         detail=f"{item['symbol']}: {item['status']}")
            )

        for pr in report.recorded:
            if not pr.evidence:
                conflicts.append(
                    Conflict(kind="no_evidence", symbols=[pr.symbol],
                             detail=f"{pr.symbol}: no supporting evidence found")
                )

        # Shared-evidence collisions: >=2 decisions lean on the same chunk/DOI.
        by_ref: dict[str, set[str]] = {}
        try:
            for rec in self.ledger.list_records():
                for ref in rec.get("evidence", []):
                    rk = ref.get("ref_key")
                    if rk:
                        by_ref.setdefault(rk, set()).add(rec["symbol"])
        except Exception:  # noqa: BLE001
            pass
        for rk, syms in by_ref.items():
            if len(syms) >= 2:
                conflicts.append(
                    Conflict(
                        kind="shared_evidence",
                        symbols=sorted(syms),
                        detail=f"{rk} cited by {len(syms)} decisions",
                    )
                )
        return conflicts

    # -- interactive write-back (ledger only, never source) -----------------
    def re_ground(self, symbol: str, *, external: bool | None = None) -> PopulatedRecord:
        """Re-run grounding for an existing record and update its evidence.

        ``external`` overrides ``cfg.external.enabled`` for this call only —
        handy for ``evidence populate --external`` to fix up a record that
        was first grounded with external off.
        """
        rec = self.ledger.lookup(symbol)
        if rec is None:
            raise ValueError(f"no record for {symbol!r}")
        cited = rec.get("cited") or []
        evidence, notes = self._ground(
            symbol, rec.get("decision", ""), external=external, cited=cited
        )
        self.ledger.record(
            symbol,
            rec["decision"],
            evidence=evidence,
            alternatives=rec.get("alternatives", []),
            rationale=rec.get("rationale", ""),
            code_fingerprint=rec.get("code_fingerprint", ""),
            depends_on=rec.get("depends_on", []),
            cited=cited,
        )
        return PopulatedRecord(
            symbol=symbol,
            decision=rec["decision"],
            evidence=evidence,
            grounded=bool(evidence),
            notes=notes,
        )


# ── reporting helpers (Typer-free) ───────────────────────────────────────────
def draft_payload_preview(record: PopulatedRecord) -> str:
    """Compact one-block preview of a drafted record (for dry-run output)."""
    alt = ", ".join(a.get("name", "?") for a in record.alternatives[:3])
    ev = ", ".join(e.get("ref_key", "?") for e in record.evidence[:3])
    return json.dumps(
        {
            "symbol": record.symbol,
            "decision": record.decision,
            "rationale": record.rationale,
            "alternatives": alt,
            "depends_on": record.depends_on,
            "evidence": ev,
            "grounded": record.grounded,
            "notes": record.notes,
        },
        indent=2,
    )
