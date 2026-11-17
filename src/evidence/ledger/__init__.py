"""Evidence ledger — cache-first decision records for code symbols.

Public surface:
    Ledger                 — the SQLite-backed store (lookup/record/audit/impact).
    EvidenceRef, Record    — record value types.
    open_ledger            — convenience factory from a EvidenceConfig.
    FRESH/STALE_CODE/STALE_EVIDENCE/ORPHANED — status constants.
"""

from __future__ import annotations

from pathlib import Path

from ..config import EvidenceConfig
from .fingerprint import fingerprint_source, scan_repo, symbol_fingerprint
from .store import (
    FRESH,
    ORPHANED,
    STALE_CODE,
    STALE_DEPENDENCY,
    STALE_EVIDENCE,
    EvidenceRef,
    Ledger,
    Record,
)

__all__ = [
    "Ledger",
    "EvidenceRef",
    "Record",
    "open_ledger",
    "get_ledger",
    "set_ledger",
    "FRESH",
    "STALE_CODE",
    "STALE_DEPENDENCY",
    "STALE_EVIDENCE",
    "ORPHANED",
    "fingerprint_source",
    "symbol_fingerprint",
    "scan_repo",
]


def open_ledger(cfg: EvidenceConfig | None = None) -> Ledger:
    """Open (or create) the ledger at the config's resolved path.

    The ledger also keeps the human-readable `ledger.md` (next to the SQLite
    file) in sync on every write — cheap, best-effort, and makes the committed
    review artifact always current without a separate export step.
    """
    cfg = cfg or EvidenceConfig.load()
    return Ledger(
        cfg.resolved_ledger_path,
        ledger_md_path=cfg.resolved_ledger_path.parent / "ledger.md",
    )


_LEDGER: Ledger | None = None


def get_ledger(cfg: EvidenceConfig | None = None) -> Ledger:
    """Process-wide cached ledger (reuses one SQLite connection).

    Prefer this over `open_ledger` in long-lived processes (the MCP server) so
    we don't open a new connection per call. Pass `cfg` to override.
    """
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = open_ledger(cfg)
    elif cfg is not None and _LEDGER.path != Path(cfg.resolved_ledger_path):
        _LEDGER.close()
        _LEDGER = open_ledger(cfg)
    return _LEDGER


def set_ledger(ledger: Ledger | None) -> None:
    """Override the cached ledger (e.g. with a temp path in tests)."""
    global _LEDGER
    if _LEDGER is not None and _LEDGER is not ledger:
        try:
            _LEDGER.close()
        except Exception:  # noqa: BLE001
            pass
    _LEDGER = ledger
