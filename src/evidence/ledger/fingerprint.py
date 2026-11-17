"""Code fingerprinting for evidence-ledger drift detection (new plan).

The ledger records a design decision for a symbol (function/class/API). To know
later whether the code behind that decision has drifted (`stale_code`) or the
symbol is gone (`orphaned`), we hash a *stable window* of the symbol's source.

Design notes:
  - A whole-file hash is too brittle: any unrelated edit in the file would
    invalidate the record. Instead we hash the symbol's own source lines
    (bounded by `window`) plus its qualified name, so refactors *of that
    symbol* invalidate it but sibling edits do not.
  - `scan_repo` is best-effort and dependency-free (stdlib `ast`): it indexes
    every top-level `def`/`class`/module-level assignment in each `.py` file,
    keyed as `module.Symbol`, so `audit_evidence` can compare fingerprints
    without importing (and executing) the target code.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_ALGOS = {"sha256", "sha1", "md5"}


def _h(data: bytes, algo: str) -> str:
    name = algo if algo in _ALGOS else "sha256"
    return hashlib.new(name, data).hexdigest()


def fingerprint_source(
    source: str,
    *,
    symbol: str = "",
    algo: str = "sha256",
    window: int = 120,
) -> str:
    """Hash a stable prefix of `source` (plus the symbol name).

    Only the first `window` lines are hashed, so a record stays valid while the
    symbol's signature/top-of-body is unchanged, and invalidates if those drift.
    """
    lines = source.splitlines()
    if window > 0:
        lines = lines[:window]
    body = "\n".join(lines)
    seed = f"{symbol}\x00{body}".encode("utf-8", "replace")
    return _h(seed, algo)


def _symbol_lines(node: ast.AST, source_lines: list[str]) -> list[str]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None) or start
    return source_lines[max(start - 1, 0) : end]


def symbol_fingerprint(
    source: str,
    symbol: str,
    *,
    algo: str = "sha256",
    window: int = 120,
) -> str | None:
    """Fingerprint one named symbol found in `source` (a whole file's text).

    Locates the top-level `def`/`class`/assignment whose name (or dotted tail)
    matches `symbol` and hashes its lines. Returns `None` when not found.
    """
    lines = source.splitlines()
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return None

    target = symbol.split(".")[-1]
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    name = t.id
                    break
        if name == target:
            return fingerprint_source(
                "\n".join(_symbol_lines(node, lines)),
                symbol=symbol,
                algo=algo,
                window=window,
            )
    return None


def scan_repo(
    root: str | Path,
    *,
    algo: str = "sha256",
    window: int = 120,
    suffixes: tuple[str, ...] = (".py",),
    ignore: tuple[str, ...] = (
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".mypy_cache", ".ruff_cache", "build", "dist", "site-packages",
    ),
    max_files: int = 400,
) -> dict[str, str]:
    """Best-effort map of ``module.Symbol`` → fingerprint across a repo.

    Walks `.py` files under `root` (skipping `ignore` dirs), parses each, and
    records a fingerprint for every top-level definition. Never imports the
    target code. Raises nothing on a bad file — it is simply skipped.
    """
    root = Path(root)
    out: dict[str, str] = {}
    if not root.exists():
        return out

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix in suffixes
        and not any(part in ignore for part in p.parts)
    ][:max_files]

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Module dotted name relative to root (best-effort, extension-less).
        rel = path.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts) or path.stem
        fp = symbol_fingerprint_module(text, algo=algo, window=window, module=module)
        out.update(fp)
    return out


def symbol_fingerprint_module(
    source: str,
    *,
    module: str,
    algo: str = "sha256",
    window: int = 120,
) -> dict[str, str]:
    """Fingerprint every top-level symbol in one file, keyed as ``module.Name``."""
    lines = source.splitlines()
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return {}
    out: dict[str, str] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    name = t.id
                    break
        if not name:
            continue
        qualified = f"{module}.{name}" if module else name
        fp = fingerprint_source(
            "\n".join(_symbol_lines(node, lines)),
            symbol=qualified,
            algo=algo,
            window=window,
        )
        if fp:
            out[qualified] = fp
            out[name] = fp  # also index by bare name for easy lookup
    return out
