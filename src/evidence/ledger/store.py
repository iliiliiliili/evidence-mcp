"""Evidence ledger — cache-first decision records for code symbols (new plan).

This is the "cache-first layer" from the plan: before any agent runs retrieval
or the heavy pipeline, it checks whether a *justified decision already exists*
for the symbol it is about to touch.

Records
-------
A **record** captures, for one symbol (a function/class/API, addressed by its
dotted name or bare name):

  - ``decision``     — the design decision made (e.g. "chose Adam w/ cosine").
  - ``rationale``    — *why*, grounded in evidence.
  - ``alternatives`` — what was considered and rejected.
  - ``evidence``     — the supporting corpus chunk IDs and/or external DOIs that
                       justify the decision (the provenance link back to the
                       paper corpus / literature).
  - ``status``       — current staleness verdict (see audit).
  - ``code_fingerprint`` — a stable hash of the symbol's source, so we can tell
                       later whether the code has drifted since the record.

This module is **pure and dependency-free** (stdlib SQLite + the local
``fingerprint`` helper) so it is trivial to unit-test and cheap to call
(``lookup`` is a single indexed SELECT).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fingerprint import scan_repo

# Record staleness states (plan: audit_evidence + impact_analysis).
FRESH = "fresh"
STALE_CODE = "stale_code"
STALE_EVIDENCE = "stale_evidence"
STALE_DEPENDENCY = "stale_dependency"  # a depended-upon decision was invalidated
ORPHANED = "orphaned"


@dataclass
class EvidenceRef:
    """One piece of evidence attached to a record.

    Either a corpus ``chunk_id`` (local) or an external ``doi`` (literature).
    ``source`` is "local" | "openalex" | "s2" | "crossref" | "doi".
    """

    ref_type: str  # "chunk" | "doi"
    ref_key: str  # the chunk_id or the DOI string
    source: str = "local"
    title: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvidenceRef:
        ref_type = d.get("ref_type") or d.get("type") or (
            "chunk" if d.get("chunk_id") else "doi"
        )
        ref_key = d.get("ref_key") or d.get("chunk_id") or d.get("doi") or ""
        return cls(
            ref_type=ref_type,
            ref_key=str(ref_key),
            source=d.get("source", "local"),
            title=d.get("title", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {"ref_type": self.ref_type, "ref_key": self.ref_key, "source": self.source}
        if self.title:
            d["title"] = self.title
        if self.ref_type == "chunk":
            d["chunk_id"] = self.ref_key
        elif self.ref_type == "doi":
            d["doi"] = self.ref_key
        # `citation` refs carry a display name as `ref_key`; no canonical id.
        return d


@dataclass
class Record:
    symbol: str
    decision: str
    rationale: str = ""
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    status: str = FRESH
    code_fingerprint: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status,
            "code_fingerprint": self.code_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Ledger:
    """SQLite-backed evidence ledger with the four public operations.

    ``lookup`` / ``record`` / ``audit`` / ``impact`` map 1:1 onto the MCP tools
    ``lookup_evidence`` / ``record_evidence`` / ``audit_evidence`` /
    ``impact_analysis``.
    """

    def __init__(self, path: str | Path, *, ledger_md_path: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Optional: keep a human-readable artifact in sync on every write. Set
        # by the app factories (open_ledger/get_ledger); left None for the bare
        # store (unit tests) so its pure, side-effect-free contract is kept.
        self.ledger_md_path: Path | None = (
            Path(ledger_md_path) if ledger_md_path is not None else None
        )
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                symbol           TEXT PRIMARY KEY,
                decision         TEXT NOT NULL,
                rationale        TEXT NOT NULL DEFAULT '',
                alternatives     TEXT NOT NULL DEFAULT '[]',   -- JSON list
                cited            TEXT NOT NULL DEFAULT '[]',   -- JSON list of model-cited works
                status           TEXT NOT NULL DEFAULT 'fresh',
                code_fingerprint TEXT NOT NULL DEFAULT '',
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_refs (
                record_id TEXT NOT NULL,
                ref_type  TEXT NOT NULL,   -- 'chunk' | 'doi' | 'citation'
                ref_key   TEXT NOT NULL,
                source    TEXT NOT NULL DEFAULT 'local',
                title     TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (record_id, ref_type, ref_key),
                FOREIGN KEY (record_id) REFERENCES records(symbol) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_refs_key ON evidence_refs(ref_key);
            CREATE TABLE IF NOT EXISTS depends_on (
                dependent TEXT NOT NULL,   -- the record that depends on another
                parent    TEXT NOT NULL,   -- the decision it depends on
                PRIMARY KEY (dependent, parent),
                FOREIGN KEY (dependent) REFERENCES records(symbol) ON DELETE CASCADE
                -- `parent` intentionally has NO FK: a decision may declare a
                -- dependency on a record that is not (yet) recorded.
            );
            """
        )
        # Forward-compatible migration: add `cited` to pre-existing DBs that
        # were created before that column was part of the schema.
        cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(records)")
        }
        if "cited" not in cols:
            self._conn.execute(
                "ALTER TABLE records ADD COLUMN cited TEXT NOT NULL DEFAULT '[]'"
            )
        self._conn.commit()

    # ── public: lookup ────────────────────────────────────────────────────
    def lookup(self, symbol: str) -> dict[str, Any] | None:
        """Return the stored record for `symbol`, or None if none exists.

        This is the *cache-first* read: a hit means the agent already knows why
        the symbol is written the way it is, and can skip retrieval entirely.
        """
        row = self._conn.execute(
            "SELECT * FROM records WHERE symbol=?", (symbol,)
        ).fetchone()
        if row is None:
            return None
        rec = self._record_from_row(row)
        rec["evidence"] = self._refs_for(symbol)
        rec["depends_on"] = self._deps_for(symbol)
        return rec

    # ── dependency helpers ────────────────────────────────────────────────
    def _deps_for(self, symbol: str) -> list[str]:
        """Decisions that `symbol` DEPENDS ON (its parents)."""
        rows = self._conn.execute(
            "SELECT parent FROM depends_on WHERE dependent=? ORDER BY parent",
            (symbol,),
        ).fetchall()
        return [r["parent"] for r in rows]

    def _dependents_of(self, symbol: str) -> list[str]:
        """Decisions that depend ON `symbol` (its children)."""
        rows = self._conn.execute(
            "SELECT dependent FROM depends_on WHERE parent=? ORDER BY dependent",
            (symbol,),
        ).fetchall()
        return [r["dependent"] for r in rows]

    def _all_transitive_parents(self, symbol: str, *, max_depth: int = 10) -> list[str]:
        """All decisions that `symbol` (transitively) depends on, via BFS upward."""
        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(symbol, 0)]
        out: list[str] = []
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for parent in self._deps_for(node):
                if parent in seen or parent == symbol:
                    continue
                seen.add(parent)
                out.append(parent)
                queue.append((parent, depth + 1))
        return out

    def transitive_dependents(self, symbol: str, *, max_depth: int = 10) -> list[str]:
        """All decisions that (transitively) depend on `symbol`, via BFS.

        This is the propagation map for "if I change `symbol`, which derived
        decisions must be revisited?" — e.g. changing "use Python" reaches
        "use PyTorch" → "model.train" even though they cite different evidence.
        Cycle-safe (visited set) and depth-capped.
        """
        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(symbol, 0)]
        out: list[str] = []
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for child in self._dependents_of(node):
                if child in seen or child == symbol:
                    continue
                seen.add(child)
                out.append(child)
                queue.append((child, depth + 1))
        return out

    # ── public: record ────────────────────────────────────────────────────
    def record(
        self,
        symbol: str,
        decision: str,
        evidence: list[dict[str, Any]] | None = None,
        alternatives: list[dict[str, Any]] | None = None,
        *,
        rationale: str = "",
        code_fingerprint: str = "",
        depends_on: list[str] | None = None,
        cited: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Insert or update the decision record for `symbol`.

        Upsert semantics: the *latest* justified decision wins; the evidence
        references and the `depends_on` edges are replaced wholesale (a record
        is a snapshot of the reasoning at the time of the change).

        `depends_on` names OTHER decisions this one is derived from (e.g.
        "use PyTorch" depends on "use Python"). These edges let `impact` /
        `audit` propagate a root decision's change transitively through every
        decision that built on it.
        """
        now = _now_iso()
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not decision or not decision.strip():
            raise ValueError("decision must be a non-empty string")

        exists = self._conn.execute(
            "SELECT 1 FROM records WHERE symbol=?", (symbol,)
        ).fetchone()
        cited_json = json.dumps(cited or [])
        if exists:
            self._conn.execute(
                "UPDATE records SET decision=?, rationale=?, alternatives=?, "
                "cited=?, status=?, code_fingerprint=?, updated_at=? WHERE symbol=?",
                (
                    decision,
                    rationale,
                    json.dumps(alternatives or []),
                    cited_json,
                    FRESH,
                    code_fingerprint,
                    now,
                    symbol,
                ),
            )
        else:
            self._conn.execute(
                "INSERT INTO records (symbol, decision, rationale, alternatives, "
                "cited, status, code_fingerprint, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    decision,
                    rationale,
                    json.dumps(alternatives or []),
                    cited_json,
                    FRESH,
                    code_fingerprint,
                    now,
                    now,
                ),
            )
        self._conn.execute("DELETE FROM evidence_refs WHERE record_id=?", (symbol,))
        for ref in evidence or []:
            self._upsert_ref(symbol, EvidenceRef.from_dict(ref))

        # Dependency edges (replace wholesale, same upsert semantics).
        self._conn.execute("DELETE FROM depends_on WHERE dependent=?", (symbol,))
        for parent in depends_on or []:
            parent = parent.strip()
            if parent:
                self._conn.execute(
                    "INSERT OR IGNORE INTO depends_on (dependent, parent) VALUES (?, ?)",
                    (symbol, parent),
                )
        self._conn.commit()
        self._maybe_refresh_md()  # keep ledger.md in sync (no-op if unset)
        return self.lookup(symbol)

    def _maybe_refresh_md(self) -> None:
        """Best-effort: rewrite the human-readable `ledger.md` if one is set.

        Called after every mutation so the committed review artifact never goes
        stale. Never raises — a failure to render (e.g. a closed connection in
        a test) is swallowed; the ledger itself is already committed.
        """
        if self.ledger_md_path is None:
            return
        try:
            self.ledger_md_path.parent.mkdir(parents=True, exist_ok=True)
            self.ledger_md_path.write_text(self.export_markdown(), encoding="utf-8")
        except Exception:  # noqa: BLE001 - artifact refresh is best-effort
            pass

    # ── public: audit ─────────────────────────────────────────────────────
    def audit(
        self,
        scope: str = "repo",
        *,
        store: Any | None = None,
        algo: str = "sha256",
        window: int = 120,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Scan code fingerprints and flag drifted records.

        Only *invalidated* records are re-checked (the plan's "re-checks only
        invalidated records"). Returns a summary plus the list of records that
        need re-analysis, each with its new ``status``:

          - ``orphaned``       — the symbol no longer exists in the code.
          - ``stale_code``     — the symbol exists but its source drifted.
          - ``stale_evidence`` — a cited corpus chunk is no longer in the index
                                (best-effort when a `store` is provided).
          - ``stale_dependency`` — a decision this record (transitively) depends
                                on was itself invalidated (e.g. the root
                                "use Python" decision changed).
          - ``fresh``          — unchanged.

        `scope` is a repo path to scan (or the literal ``"repo"`` which is
        treated as the current working directory). Cost: hashing only — no LLM.
        """
        root = Path(scope).expanduser() if scope != "repo" else Path.cwd()
        if not root.exists():
            root = Path.cwd()

        # Index every symbol's current fingerprint (best-effort, no imports).
        try:
            fingerprints = scan_repo(root, algo=algo, window=window)
        except Exception:  # noqa: BLE001 - scan is best-effort
            fingerprints = {}

        rows = self._conn.execute("SELECT * FROM records").fetchall()
        fresh: list[str] = []
        stale_code: list[str] = []
        stale_evidence: list[str] = []
        stale_dependency: list[str] = []
        orphaned: list[str] = []
        requeue: list[dict[str, str]] = []
        rows_and_symbols = [r["symbol"] for r in rows]
        requeued_set = set()

        for row in rows:
            symbol = row["symbol"]
            cur_fp = fingerprints.get(symbol) or fingerprints.get(
                symbol.split(".")[-1]
            )
            status = FRESH
            if cur_fp is None:
                status = ORPHANED
            elif row["code_fingerprint"] and cur_fp != row["code_fingerprint"]:
                status = STALE_CODE

            # Best-effort evidence staleness: a cited chunk that is no longer
            # indexed is a signal the corpus the decision relied on changed.
            if status == FRESH and store is not None:
                for ref in self._refs_for(symbol):
                    if ref.ref_type == "chunk" and ref.ref_key:
                        try:
                            if store.get_chunk(ref.ref_key) is None:
                                status = STALE_EVIDENCE
                                break
                        except Exception:  # noqa: BLE001 - store may be closed
                            pass

            # -- refresh mode: silently accept code drift -----------------
            # In auto-refresh (e.g. a git post-checkout hook), a record whose
            # *code* changed is re-stamped to the new fingerprint and marked
            # fresh — the user accepted "the code moved on" by running refresh.
            # Orphans (symbol deleted) can't be re-stamped, so they stay flagged
            # for a human; stale_evidence still needs re-grounding.
            if refresh and status == STALE_CODE and cur_fp is not None:
                self._conn.execute(
                    "UPDATE records SET code_fingerprint=?, status=?, updated_at=? "
                    "WHERE symbol=?",
                    (cur_fp, FRESH, _now_iso(), symbol),
                )
                fresh.append(symbol)
                continue

            if status != FRESH:
                self._conn.execute(
                    "UPDATE records SET status=?, updated_at=? WHERE symbol=?",
                    (status, _now_iso(), symbol),
                )
                requeue.append({"symbol": symbol, "status": status})
                requeued_set.add(symbol)

            if status == STALE_CODE:
                stale_code.append(symbol)
            elif status == STALE_EVIDENCE:
                stale_evidence.append(symbol)
            elif status == ORPHANED:
                orphaned.append(symbol)
            else:
                fresh.append(symbol)

        # ── Dependency propagation ─────────────────────────────────────────
        # A root decision that was invalidated (stale_code / stale_evidence /
        # orphaned) taints every decision that (transitively) depends on it:
        # the derived decisions are now suspect even if their own code and
        # evidence look fine. Mark them `stale_dependency` and re-queue.
        for sym in rows_and_symbols:
            if sym in requeued_set:
                continue
            # does this fresh record depend (transitively) on any invalidated
            # decision?
            parents = self._all_transitive_parents(sym)
            if any(p in requeued_set for p in parents):
                self._conn.execute(
                    "UPDATE records SET status=?, updated_at=? WHERE symbol=?",
                    (STALE_DEPENDENCY, _now_iso(), sym),
                )
                requeue.append({"symbol": sym, "status": STALE_DEPENDENCY})
                stale_dependency.append(sym)
                if sym in fresh:
                    fresh.remove(sym)
        self._conn.commit()

        return {
            "scope": str(root),
            "total": len(rows),
            "fresh": len(fresh),
            "stale_code": len(stale_code),
            "stale_evidence": len(stale_evidence),
            "stale_dependency": len(stale_dependency),
            "orphaned": len(orphaned),
            "needs_reanalysis": requeue,
        }

    # ── public: impact ────────────────────────────────────────────────────
    def impact(self, symbol_or_chunk: str) -> dict[str, Any]:
        """List every OTHER decision that would be affected by a change to `key`.

        Two independent signals are combined (deduped, each tagged with `via`):

          - **shared_evidence** — decisions citing the same chunk/DOI (the
            original, evidence-based propagation).
          - **depends_on**     — decisions that *declare* a dependency on `key`
            (transitively). This is the "root decision changed" map: change
            "use Python" and "use PyTorch" → "model.train" all surface even
            though they cite different evidence.

        Given a symbol → both signals apply (that record's evidence + its
        dependents). Given a bare chunk_id/DOI → only shared_evidence applies.
        """
        key = symbol_or_chunk.strip()
        if not key:
            return {"query": symbol_or_chunk, "shared_refs": [], "affected": []}

        # Resolve the set of ref_keys to match. If `key` is a symbol, use its
        # record's refs; otherwise treat it as a ref_key directly.
        record = self.lookup(key)
        if record is not None:
            ref_keys = [r["ref_key"] for r in record["evidence"] if r.get("ref_key")]
        else:
            ref_keys = [key]

        by_symbol: dict[str, dict[str, Any]] = {}

        # (1) Shared-evidence propagation.
        if ref_keys:
            placeholders = ",".join("?" * len(ref_keys))
            rows = self._conn.execute(
                f"""
                SELECT r.symbol, r.decision, r.status
                FROM records r
                JOIN evidence_refs e ON e.record_id = r.symbol
                WHERE e.ref_key IN ({placeholders})
                ORDER BY r.updated_at DESC
                """,
                ref_keys,
            ).fetchall()
            for r in rows:
                if r["symbol"] == key:
                    continue
                by_symbol.setdefault(
                    r["symbol"],
                    {"symbol": r["symbol"], "decision": r["decision"],
                     "status": r["status"], "via": ["shared_evidence"]},
                )
        else:
            rows = []

        # (2) Dependency propagation (only meaningful when `key` is a symbol).
        if record is not None:
            for dep in self.transitive_dependents(key):
                if dep in by_symbol:
                    by_symbol[dep]["via"].append("depends_on")
                    continue
                dep_rec = self.lookup(dep) or {}
                by_symbol.setdefault(
                    dep,
                    {
                        "symbol": dep,
                        "decision": dep_rec.get("decision", ""),
                        "status": dep_rec.get("status", ""),
                        "via": ["depends_on"],
                    },
                )

        # Shared refs (evidence that several decisions lean on).
        shared = []
        if ref_keys:
            placeholders = ",".join("?" * len(ref_keys))
            shared = self._conn.execute(
                f"""
                SELECT ref_key, ref_type, source, COUNT(*) AS n
                FROM evidence_refs
                WHERE ref_key IN ({placeholders})
                GROUP BY ref_key
                ORDER BY n DESC
                """,
                ref_keys,
            ).fetchall()

        affected = list(by_symbol.values())

        return {
            "query": key,
            "shared_refs": [
                {"ref_key": s["ref_key"], "ref_type": s["ref_type"],
                 "source": s["source"], "used_by": s["n"]}
                for s in shared
            ],
            "affected": affected,
            "affected_count": len(affected),
        }

    # ── helpers ───────────────────────────────────────────────────────────
    def _upsert_ref(self, symbol: str, ref: EvidenceRef) -> None:
        if not ref.ref_key:
            return
        self._conn.execute(
            "INSERT INTO evidence_refs (record_id, ref_type, ref_key, source, title) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(record_id, ref_type, ref_key) DO UPDATE SET "
            "source=excluded.source, title=excluded.title",
            (symbol, ref.ref_type, ref.ref_key, ref.source, ref.title),
        )

    def _refs_for(self, symbol: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT ref_type, ref_key, source, title FROM evidence_refs "
            "WHERE record_id=? ORDER BY ref_key",
            (symbol,),
        ).fetchall()
        return [
            EvidenceRef(
                ref_type=r["ref_type"], ref_key=r["ref_key"],
                source=r["source"], title=r["title"],
            ).to_dict()
            for r in rows
        ]

    def _record_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out: dict[str, Any] = {
            "symbol": row["symbol"],
            "decision": row["decision"],
            "rationale": row["rationale"],
            "alternatives": json.loads(row["alternatives"] or "[]"),
            "status": row["status"],
            "code_fingerprint": row["code_fingerprint"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        # `cited` is forward-optional: older DBs may not have the column yet.
        try:
            out["cited"] = json.loads(row["cited"] or "[]")
        except (KeyError, IndexError):
            out["cited"] = []
        return out

    def list_records(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM records ORDER BY updated_at DESC"
        ).fetchall()
        out = []
        for r in rows:
            d = self._record_from_row(r)
            d["evidence"] = self._refs_for(r["symbol"])
            d["depends_on"] = self._deps_for(r["symbol"])
            out.append(d)
        return out

    def count_records(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) AS n FROM records"
        ).fetchone()["n"]

    # ── public: export (read-only projections for git review) ────────────
    def export_json(self) -> str:
        """Render the whole ledger as pretty JSON (a readable, diff-able
        snapshot of every decision + its evidence links + depends_on edges).

        This is the machine-readable projection a consumer repo can commit so
        `git diff` shows exactly which evidence links changed between commits.
        """
        records = self.list_records()
        payload = {
            "generated_at": _now_iso(),
            "record_count": len(records),
            "records": records,
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def export_markdown(self, resolver=None, group_fn=None) -> str:
        """Render the whole ledger as a human-readable Markdown document.

        Each record shows its decision, rationale, the supporting evidence
        (as human-readable document names / sections), rejected alternatives,
        and the `depends_on` chain so a reviewer can see the full reasoning
        without opening SQLite. Records are ordered by symbol for stable,
        low-churn diffs.

        `resolver` (optional) maps an evidence ref dict → a human label (e.g.
        "Paper Title · Section (p.4)"). When omitted (e.g. the automatic
        on-write refresh, which has no corpus), the stored human `title` is
        used and the raw machine key is never printed — machine keys live only
        in the JSON export and the database.

        `group_fn` (optional) maps a record's evidence list → a list of
        Markdown lines (indented for the bullet list). When provided it
        replaces the flat per-ref rendering, e.g. to group sections under
        their parent paper. It is only consulted for records that have
        evidence; records with none still render `- **Evidence:** _none_`.
        """
        records = sorted(self.list_records(), key=lambda r: r["symbol"])
        lines: list[str] = []
        lines.append("# Evidence Ledger")
        lines.append("")
        lines.append(
            f"_{len(records)} decision record(s), generated "
            f"{_now_iso()}._"
        )
        lines.append("")
        if not records:
            lines.append("_No decision records yet._")
            lines.append("")
            return "\n".join(lines)

        # Status summary.
        by_status: dict[str, int] = {}
        for r in records:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items()))
        lines.append(f"**Status:** {summary}")
        lines.append("")

        for r in records:
            lines.append(f"## `{r['symbol']}`")
            lines.append("")
            lines.append(f"- **Decision:** {r['decision']}")
            lines.append(f"- **Status:** {r['status']}")
            if r.get("rationale"):
                lines.append(f"- **Rationale:** {r['rationale']}")

            cited = r.get("cited") or []
            if cited:
                lines.append("- **Cited works (model-proposed):**")
                for c in cited:
                    if not isinstance(c, dict):
                        continue
                    name = c.get("name") or c.get("doi") or "(unnamed)"
                    authors = f" — {c['authors']}" if c.get("authors") else ""
                    doi = f" [`{c['doi']}`]" if c.get("doi") else " [unresolved]"
                    lines.append(f"  - {name}{authors}{doi}")

            depends = r.get("depends_on") or []
            if depends:
                dep_txt = ", ".join(f"`{d}`" for d in depends)
                lines.append(f"- **Depends on:** {dep_txt}")

            ev = r.get("evidence") or []
            if ev:
                lines.append("- **Evidence:**")
                if group_fn is not None:
                    for ln in group_fn(ev):
                        lines.append(ln)
                else:
                    for e in ev:
                        if resolver is not None:
                            lines.append(f"  - {resolver(e)}")
                        else:
                            # No corpus resolver available (e.g. the automatic
                            # on-write refresh): show the human title when known,
                            # a readable DOI/citation otherwise. Never the raw
                            # chunk machine-key — that belongs only in JSON / DB.
                            if e.get("title"):
                                lines.append(f"  - {e['title']}")
                            elif e.get("ref_type") in ("doi", "citation"):
                                lines.append(f"  - {e['ref_key']}")
                            else:
                                lines.append("  - (local evidence)")
            else:
                lines.append("- **Evidence:** _none_")

            alts = r.get("alternatives") or []
            if alts:
                lines.append("- **Alternatives:**")
                for a in alts:
                    if isinstance(a, dict):
                        name = (
                            a.get("name")
                            or a.get("option")
                            or a.get("description")
                            or a.get("decision")
                            or "(unnamed alternative)"
                        )
                        why = (
                            a.get("why_rejected")
                            or a.get("rejected")
                            or a.get("reason")
                            or ""
                        )
                        desc = f"{name} — {why}" if why else str(name)
                    else:
                        desc = str(a)
                    lines.append(f"  - {desc}")

            fp = r.get("code_fingerprint") or ""
            if fp:
                lines.append(f"- **Code fingerprint:** `{fp}`")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def close(self) -> None:
        self._conn.close()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
