"""Evidence command-line interface.

Commands:
  evidence ingest <corpus>   Extract + chunk + index a corpus directory.
  evidence ask "question"    Run the full evidence-gated RAG pipeline.
  evidence status            Show corpus / index / config summary.
  evidence config            Print the resolved configuration.
  evidence populate [scope]  Draft, ground, and record design decisions
                             for a code scope; resolve conflicts interactively.
  evidence ledger export     Color-coded terminal view; -f md|json writes a file.
  evidence ledger impact     Show which decisions are affected by a symbol/chunk/DOI.
  evidence ledger symbols    List symbols that have a decision record (+status/evidence).
  evidence review <paper>    Structured scientific review (summary/strengths/
                             weaknesses/incorrect/undefined terms + references).
  evidence references <p>    A paper's bibliography as clean, untruncated entries.
  evidence chunk <chunk_id>  Read one chunk's full, untruncated text + context.
  evidence sections <p>      A paper's section outline (hierarchical navigation).
  evidence section <p> <sec> Read one section as a single untruncated unit.
  evidence build-links [p]   Build the chunk-graph (deterministic + LLM links).
  evidence links <chunk|p>   Show the chunk-graph around a chunk or a paper.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import EvidenceConfig
from .indexes.store import Store
from .ingestion.pipeline import ingest_corpus
from .models.ollama import Models

app = typer.Typer(
    name="evidence",
    help="Local-first, evidence-gated agentic RAG for technical literature.",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _stdin_is_tty() -> bool:
    """True when stdin is an interactive terminal (safe to prompt)."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:  # noqa: BLE001
        return False


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, show_path=False, rich_tracebacks=True)],
    )


def _load_config(path: str | None) -> EvidenceConfig:
    return EvidenceConfig.load(path)


def _build_app(cfg: EvidenceConfig):
    from .graph.builder import Evidence

    return Evidence(cfg)


# ── init (one-command setup) ────────────────────────────────────────────────
CONFIG_TEMPLATE = """\
# Evidence project config — created by `evidence init`.
# Edit freely; every field has a working default. Env overrides use the
# `EVIDENCE_<SECTION>_<FIELD>` convention (e.g. `EVIDENCE_OLLAMA_REASONING=...`).

# ── Local models (Ollama) ────────────────────────────────────────────────
ollama:
  host: http://localhost:11434
  reasoning: qwen2.5:9b
  light: qwen2.5:4b
  embed: bge-m3
  request_timeout: 300
  think: false              # set true to enable chain-of-thought on thinking models
  auto_pull: true           # pull missing models from the registry before first use
  keep_alive: 1h            # keep models loaded in VRAM (0 = unload, 168h = resident)

# ── Retrieval / chunking ─────────────────────────────────────────────────
retrieval:
  candidates: 50
  keep_after_rerank: 15
  use_bm25: true
  use_dense: true
chunking:
  max_tokens: 800
  overlap_tokens: 100

# ── Evidence sufficiency gate ────────────────────────────────────────────
sufficiency:
  threshold: 70
  min_evidence_items: 3

# ── External / graph / visual (all optional, all disabled by default) ───
external:
  enabled: false
  # When grounding pulls in an online reference (a DOI from OpenAlex/S2/
  # Crossref), save the FULL document into the corpus if an OA PDF is
  # available; otherwise keep just the title + section/DOI.
  save_online_to_corpus: true
graph_expansion:
  enabled: false
visual:
  enabled: false

# ── Evidence ledger (cache-first decision records) ───────────────────────
ledger:
  enabled: true
  path: null            # defaults to <data_dir>/ledger.sqlite

# ── MCP server ───────────────────────────────────────────────────────────
mcp:
  allow_ingest: false   # gate the mutating `ingest_paper` tool
  # Set true when a capable MCP client model (VS Code Copilot, etc.) is driving
  # the conversation: `answer_research` then returns an evidence bundle for the
  # client to synthesise instead of running Evidence's local generation models.
  prefer_client_generation: false

# ── Storage ──────────────────────────────────────────────────────────────
data_dir: ./.evidence    # index + ledger live here
index_dir: null
sqlite_db: store.sqlite   # hybrid-retrieval chunk store (the ledger lives in ledger_db)
corpus_dir: ./corpus    # documents to search (see `evidence ingest`)
"""

MCP_JSON_TEMPLATE = """{
  \"$schema\": \"https://raw.githubusercontent.com/microsoft/vscode/main/extensions/mcp/types/mcpConfig.schema.json\",
  \"servers\": {
    \"__EVIDENCE_NAME__\": {
      \"type\": \"stdio\",
      \"command\": \"__EVIDENCE_MCP__\",
      \"args\": [\"--config\", \"__EVIDENCE_CONFIG__\"],
      \"env\": {
        \"EVIDENCE_MCP_ALLOW_INGEST\": \"0\"
      }
    }
  }
}
"""


def _evidence_mcp_path() -> str:
    """Absolute path to the `evidence-mcp` executable, if discoverable.

    VS Code spawns stdio MCP servers with a PATH that typically does NOT
    include the venv, so the bare command name will fail. We resolve the
    absolute path of the executable in the current interpreter's venv (or
    fall back to a `which` lookup, and finally to the bare name as a last
    resort).
    """
    import shutil
    import sys

    # 1) Same venv as the running interpreter (the most common case).
    #    Note: do NOT .resolve() — venv `python` is a symlink that chains out
    #    to the system interpreter, and resolving it would point at /usr/bin.
    exe = Path(sys.executable).parent / "evidence-mcp"
    if exe.exists():
        return str(exe)
    resolved = Path(sys.executable).resolve().parent / "evidence-mcp"
    if resolved.exists():
        return str(resolved)
    # 2) On PATH (system install, pipx, etc.).
    found = shutil.which("evidence-mcp")
    if found:
        return found
    # 3) Last resort — let the client try.
    return "evidence-mcp"

GITIGNORE_APPENDIX = """\
# Evidence project local state (created by `evidence init`)
# We ignore the *specific* state files (not the whole `.evidence/` dir) so the
# generated `ledger.md` / `ledger.json` review artifacts remain committable.
.evidence/ledger.sqlite
.evidence/index/
corpus/
"""


@app.command()
def init(
    corpus: Path | None = typer.Argument(
        None,
        help="Directory of documents to index. Defaults to `./corpus` if not given.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't prompt (use defaults)."),
    skip_ingest: bool = typer.Option(
        False, "--skip-ingest", help="Skip the immediate ingest step."
    ),
    use_ollama: bool | None = typer.Option(
        None,
        "--ollama/--no-ollama",
        help="Force on/off the interactive Ollama setup (default: ask).",
    ),
    auto_update: bool | None = typer.Option(
        None,
        "--auto-update/--no-auto-update",
        help="Force on/off the git auto-update hook (default: ask, Yes).",
    ),
    external: bool | None = typer.Option(
        None,
        "--external/--no-external",
        help=(
            "Force on/off external literature search (OpenAlex/S2/Crossref) "
            "for `populate` grounding (default: ask, Yes)."
        ),
    ),
    auto_pull: bool | None = typer.Option(
        True,
        "--auto-pull/--no-auto-pull",
        help=(
            "Pull Ollama models from the registry if the server doesn't have "
            "them yet (default: Yes)."
        ),
    ),
):
    """Set up an Evidence project in the current directory.

    Creates `.evidence/config.yaml`, appends the state files (sqlite index +
    `corpus/`) to `.gitignore` — but not the generated `ledger.md`, which stays
    committable — writes `.vscode/mcp.json` + `.mcp.json` (MCP client config),
    optionally configures local models (Ollama), and optionally ingests an
    existing corpus.

    After this you can:
      1. Drop PDFs / LaTeX / .md / .txt files into `corpus/`
      2. `evidence ingest`        (indexes the corpus)
      3. `evidence ask "..."`     (ask a grounded, cited question)
    """
    _setup_logging(False)
    root = Path.cwd()
    evidence_dir = root / ".evidence"
    config_path = evidence_dir / "config.yaml"
    corpus_dir = (corpus or root / "corpus").expanduser()

    if config_path.exists() and not yes:
        err_console.print(
            f"[yellow].evidence/config.yaml already exists:[/yellow] {config_path}\n"
            f"Pass `--yes` to overwrite."
        )
        raise typer.Exit(code=1)

    # 0. Optional Ollama configuration (interactive unless --yes).
    ollama_cfg: dict | None = None
    if not yes:
        if use_ollama is not False and typer.confirm(
            "Configure local models (Ollama) now?"
        ):
            ollama_cfg = _configure_ollama_interactive()
    elif use_ollama is True:
        ollama_cfg = _probe_ollama_default()

    # 0b. External literature search (default: Yes). Lets `populate` pull in
    #     the defining papers for methods the local corpus does not cover
    #     (MCD, BBB, Adam, ...). Only prompts on a real TTY so CI runs do not
    #     block on stdin; they just honor the "default yes".
    if external is None:
        if yes:
            external = True
        elif _stdin_is_tty():
            external = typer.confirm(
                "Enable external literature search (OpenAlex/S2/Crossref) "
                "for grounding design decisions?",
                default=True,
            )
        else:
            external = True

    # 1. .evidence/config.yaml (bake in the chosen Ollama host/model, if any).
    evidence_dir.mkdir(parents=True, exist_ok=True)
    config_text = CONFIG_TEMPLATE
    # External search: default-on now (the "local corpus shadows external" bug
    # made it look broken; see PLAN.MD). We bake the user's answer in.
    config_text = config_text.replace(
        "external:\n  enabled: false",
        f"external:\n  enabled: {str(external).lower()}",
    )
    # Auto-pull of missing Ollama models (default on). Bake the choice in.
    config_text = config_text.replace(
        "auto_pull: true",
        f"auto_pull: {str(bool(auto_pull)).lower()}",
    )
    if ollama_cfg:
        host = ollama_cfg["host"]
        reasoning = ollama_cfg.get("model", "")
        embed = ollama_cfg.get("embed", "")
        config_text = config_text.replace("host: http://localhost:11434", f"host: {host}")
        if reasoning:
            config_text = (
                config_text
                .replace("reasoning: qwen2.5:9b", f"reasoning: {reasoning}")
                .replace("light: qwen2.5:4b", f"light: {reasoning}")
            )
        if embed:
            config_text = config_text.replace("embed: bge-m3", f"embed: {embed}")
    config_path.write_text(config_text, encoding="utf-8")

    # 2. corpus dir
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 3. MCP client config. VS Code (Copilot Chat) reads `.vscode/mcp.json`,
    #    while many other MCP clients read the root `.mcp.json` — so we write
    #    both with identical content. A unique-per-project server name keeps a
    #    *shared* gateway (e.g. VS Code's McpGatewayService) from collapsing
    #    two projects that both name their server "evidence" into one process.
    server_name = f"evidence-{root.resolve().name}"
    mcp_text = (
        MCP_JSON_TEMPLATE
        .replace("__EVIDENCE_NAME__", server_name)
        .replace("__EVIDENCE_MCP__", _evidence_mcp_path())
        .replace("__EVIDENCE_CONFIG__", str(config_path))
    )
    mcp_targets: list[tuple[Path, Path | None]] = [
        (root / ".vscode" / "mcp.json", root / ".vscode"),
        (root / ".mcp.json", None),
    ]
    for mcp_json, parent in mcp_targets:
        if mcp_json.exists():
            console.print(f"[dim]kept existing[/dim] [cyan]{mcp_json}[/cyan]")
            continue
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        mcp_json.write_text(mcp_text, encoding="utf-8")

    # 4. .gitignore (append, idempotent). Ignore the *specific* state files —
    #    not the whole `.evidence/` dir — so `ledger export`'s generated
    #    `ledger.md` stays committable.
    gitignore = root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    index_rel = ""
    try:
        index_dir = Path(EvidenceConfig.load(str(config_path)).resolved_index_dir)
        if index_dir.is_relative_to(root):
            index_rel = index_dir.relative_to(root).as_posix() + "/"
    except Exception:  # noqa: BLE001 - best-effort; falls back to the default
        index_rel = ""
    if "created by `evidence init`" not in existing:
        text = GITIGNORE_APPENDIX
        if index_rel and index_rel not in text:
            text = text.replace(".evidence/index/", f"{index_rel}")
        with gitignore.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(text)

    console.print(f"[green]created[/green] [cyan]{config_path}[/cyan]")
    console.print(f"[green]created[/green] [cyan]{corpus_dir}/[/cyan] (your documents live here)")
    vscode_mcp = root / ".vscode" / "mcp.json"
    console.print(f"[green]created[/green] [cyan]{vscode_mcp}[/cyan] (VS Code MCP config)")
    console.print(f"[green]updated[/green] [cyan]{gitignore}[/cyan]")
    console.print("")

    # 4b. Auto-update hook (default: Yes). Keeps the ledger's fingerprints in
    #     sync on every checkout — fast (hashing only, no models) and local.
    #     Only *prompt* on a real TTY so non-interactive / CI runs never block
    #     on stdin; they just honor the "default yes".
    if auto_update is None:
        if yes:
            auto_update = True
        elif _stdin_is_tty():
            auto_update = typer.confirm(
                "Auto-update the evidence ledger on every git checkout?",
                default=True,
            )
        else:
            auto_update = True
    if auto_update:
        if _install_ledger_hook(root):
            console.print(
                "[green]✓[/green] auto-update enabled "
                "[dim](post-checkout hook — refreshes fingerprints + ledger.md)[/dim]"
            )
        else:
            err_console.print(
                "[yellow]skipped auto-update: not a git repo "
                "(run `evidence ledger hook install` later)[/yellow]"
            )
    else:
        console.print(
            "[dim]auto-update off (enable later with `evidence ledger hook install`)[/dim]"
        )
    console.print("")

    # 5. Optional immediate ingest (only if the corpus already has files).
    existing_files = _corpus_has_files(corpus_dir)
    if existing_files:
        if not skip_ingest and (yes or typer.confirm("Ingest the corpus now?")):
            _do_ingest(corpus_dir)
    else:
        console.print(
            f"[dim]corpus dir is empty — drop .pdf/.tex/.md/.txt files into "
            f"{corpus_dir}/ then run `evidence ingest`[/dim]"
        )

    console.print("")
    console.print("[bold]Next steps[/bold]")
    console.print(f"  1. Add documents to [cyan]{corpus_dir}/[/cyan]")
    console.print("  2. [bold]evidence ingest[/bold]          # index them")
    console.print('  3. [bold]evidence ask "..."[/bold]     # grounded, cited answer')
    console.print("  4. Restart VS Code MCP to pick up the server.")
    console.print("")
    console.print(
        "[dim]Tip: local models improve quality. Install Ollama and pull:[/dim]\n"
        "  [dim]ollama pull bge-m3 && ollama pull qwen2.5:4b && ollama pull qwen2.5:9b[/dim]\n"
        "[dim]No Ollama? Use BM25-only retrieval (works as-is), or set mcp.[/dim]\n"
        "[dim]prefer_client_generation: true and let your client model write.[/dim]"
    )


def _corpus_has_files(corpus_dir: Path) -> bool:
    supported = {".pdf", ".tex", ".ltx", ".latex", ".html", ".htm",
                 ".txt", ".md", ".markdown"}
    return any(p.is_file() and p.suffix.lower() in supported for p in corpus_dir.rglob("*"))


def _do_ingest(corpus_dir: Path) -> None:
    """Run the ingest pipeline, loading config via auto-discovery."""
    cfg = _load_config(None)
    models = Models(cfg)
    store = Store(cfg.resolved_store_path)
    try:
        stats = ingest_corpus(corpus_dir, cfg=cfg, models=models, store=store)
    finally:
        store.close()
    table = Table(title="Ingestion summary")
    table.add_column("papers", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("failed", justify="right")
    table.add_row(str(stats.papers), str(stats.chunks), str(len(stats.failed)))
    console.print(table)


# ── Ollama setup (interactive, best-effort) ─────────────────────────────────
def _ollama_list_models(host: str, timeout: float = 5.0) -> list[str] | None:
    """Return the list of model names available on `host`, or None on error.

    Uses the `ollama` client if importable, otherwise a raw HTTP GET so the
    probe works even if the python client is missing.
    """
    try:
        import ollama

        client = ollama.Client(host=host, timeout=timeout)
        return [m["name"] for m in client.list_models()["models"]]
    except Exception:  # noqa: BLE001 - client missing / server down
        pass
    try:
        import httpx

        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001 - no httpx or server unreachable
        return None


def _configure_ollama_interactive() -> dict | None:
    """Walk the user through Ollama config.

    Returns {host, model, embed} or None. None means "leave the template
    defaults" (user skipped, or the server was unreachable and they chose to
    skip). The caller bakes the values into the config when present.
    """
    console.print("\n[bold]Ollama setup[/bold]")

    # 1. Host — default localhost:11434, or ask.
    host = typer.prompt(
        "Ollama host",
        default="http://localhost:11434",
        show_default=False,
    ).strip() or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    console.print(f"  connecting to [cyan]{host}[/cyan] …")

    # 2. Check connectivity (retry host on failure) and list models.
    models = _ensure_reachable(host)
    if models is None:
        return None  # user skipped — keep template defaults

    # 3. Pick the reasoning model (used for planner/writer/critic, etc.).
    model = _pick_model(models, "reasoning model (drives planning, drafting, critique)")

    # 4. Pick the embedding model (drives dense retrieval). Default to bge-m3
    #    when present, else the reasoning model.
    embed_default = "bge-m3" if any(m.lower().startswith("bge-m3") for m in models) else model
    embed = _pick_model(
        models,
        "embedding model (dense retrieval)",
        default=_match_index(models, embed_default) or 1,
    )

    console.print(
        f"[green]✓[/green] Ollama [cyan]{host}[/cyan] · "
        f"reasoning [cyan]{model}[/cyan] · embed [cyan]{embed}[/cyan]\n"
    )
    return {"host": host, "model": model, "embed": embed}


def _ensure_reachable(host: str) -> list[str] | None:
    """List models on `host`, offering to retry with a different host on failure.

    Returns the model list on success, or None if the user chose to skip.
    """
    while True:
        models = _ollama_list_models(host)
        if models is not None:
            console.print(f"[green]✓[/green] Connected — {len(models)} model(s) available.")
            return models
        err_console.print(
            f"[red]✗ Could not reach Ollama at {host}.[/red]\n"
            "  - Is the Ollama server running?     [dim]ollama serve[/dim]\n"
            "  - Does a model exist?               [dim]ollama list[/dim]\n"
            "  - Or start one:                     [dim]ollama pull qwen2.5:7b[/dim]"
        )
        if not typer.confirm("Try a different host?", default=True):
            err_console.print(
                "[yellow]skipping local-model setup (using config defaults)[/yellow]\n"
            )
            return None
        new = typer.prompt("Ollama host", default="http://localhost:11434").strip()
        if not new.startswith(("http://", "https://")):
            new = "http://" + new
        host = new


def _match_index(models: list[str], needle: str) -> int | None:
    """1-based index of the model whose name starts with `needle`, else None."""
    s = sorted(models)
    for i, name in enumerate(s, start=1):
        if name.lower().startswith(needle.lower()):
            return i
    return None


def _pick_model(models: list[str], what: str, *, default: int = 1) -> str:
    """Show the model list and let the user choose one (by number or name)."""
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("model")
    for i, name in enumerate(sorted(models), start=1):
        table.add_row(str(i), name)
    console.print(table)
    choice = typer.prompt(
        f"Which {what}? (number, or type a name)", type=int, default=default, show_default=False
    )
    if 1 <= choice <= len(models):
        return sorted(models)[choice - 1]
    return typer.prompt("Exact model name").strip() or sorted(models)[0]


def _probe_ollama_default() -> dict | None:
    """Non-interactive (--yes --ollama): defaults + sensible model choices."""
    host = "http://localhost:11434"
    models = _ollama_list_models(host)
    if not models:
        console.print(f"[yellow]Ollama at {host} unavailable; using config defaults.[/yellow]")
        return None
    model = sorted(models)[0]
    embed = next((m for m in sorted(models) if m.lower().startswith("bge-m3")), model)
    return {"host": host, "model": model, "embed": embed}


# ── ingest ────────────────────────────────────────────────────────────────
@app.command()
def ingest(
    corpus: Path | None = typer.Argument(
        None, help="Directory of documents (defaults to `corpus_dir` from config, or `./corpus`)."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Extract, chunk, and index a corpus directory (recursive)."""
    _setup_logging(verbose)
    cfg = _load_config(config)
    corpus = (corpus or cfg.corpus_dir or Path.cwd() / "corpus").expanduser()
    if not corpus.exists():
        err_console.print(f"[red]corpus not found:[/red] {corpus}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Ingesting[/bold] [cyan]{corpus}[/cyan] → "
                  f"[cyan]{cfg.resolved_index_dir}[/cyan]")
    models = Models(cfg)
    store = Store(cfg.resolved_store_path)
    try:
        stats = ingest_corpus(corpus, cfg=cfg, models=models, store=store)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]ingestion failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Ingestion summary")
    table.add_column("papers", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("failed", justify="right")
    table.add_row(str(stats.papers), str(stats.chunks), str(len(stats.failed)))
    console.print(table)
    if stats.failed:
        for f in stats.failed:
            err_console.print(f"  [yellow]skipped:[/yellow] {f}")
    console.print("[green]done.[/green]")


# ── ask ───────────────────────────────────────────────────────────────────
@app.command()
def ask(
    question: str = typer.Argument(..., help="The research question to answer."),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    stream: bool = typer.Option(False, "--stream", help="Print state updates as they happen."),
    timings: bool = typer.Option(
        False,
        "--timings",
        help="Report the wall-clock time spent in each pipeline stage after the answer.",
    ),
    group: list[str] = typer.Option(
        [],
        "--group", "-g",
        help=(
            "Restrict evidence to paper group(s) (the folder name the paper "
            "was ingested from). Repeatable. When set, the pipeline only uses "
            "chunks from papers in those folders — useful for comparative "
            "questions like 'what does group A conclude, and how does group B "
            "change it?'."
        ),
    ),
    no_citations: bool = typer.Option(
        False,
        "--no-citations",
        help=(
            "Skip the verified citation table that is normally printed after "
            "the answer (each [C<n>] resolved to its paper, section, a short "
            "quote, and the filename / DOI)."
        ),
    ),
):
    """Run the full pipeline and print the cited, QC'd answer.

    \b
    Options:
      --group, -g     Restrict to one or more paper groups (folder names).
                      Repeatable. Papers in the same folder form a group;
                      external works belong to the "external" group.
      --no-citations  Skip the citation table (printed by default).

    After the answer, a citation table maps every [C<n>] marker to the
    paper it came from, the section within that paper, a short quote from
    the supporting chunk, and the filename / DOI.
    """
    _setup_logging(verbose)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    app_ = _build_app(cfg)
    console.print(f"[bold]Question:[/bold] {question}")
    if group:
        console.print(f"[dim]groups: {', '.join(group)}[/dim]")
    console.print()
    result: dict = {}
    # Per-stage wall time (seconds) and call count, both accumulated per node
    # name (a node may run more than once across retry/regeneration loops).
    # The call count is what reveals critic/revise cycles: a `draft` that hit
    # 3 times means 3 writer passes, not one slow call. Filled when --timings.
    stage_times: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    stage_order: list[str] = []
    t_start = time.perf_counter()
    t_prev = t_start
    try:
        # Indeterminate spinner + live stage label + elapsed time. A percent
        # bar is impossible: the graph has retry/regeneration loops, so the
        # step count is unknown up front. Instead we stream node-by-node and
        # label the spinner with the most recently completed stage — an honest
        # "where are we" that updates as each stage finishes. The spinner runs
        # on the error console so it never interleaves with the answer output.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("starting pipeline…", total=None)
            for update in app_.ask(
                question,
                stream=True,
                allowed_groups=group or None,
            ):
                for node, payload in update.items():
                    if stream and isinstance(payload, dict) and payload:
                        console.print(f"[dim]{node}:[/dim] " + _brief(payload))
                    if timings:
                        t_now = time.perf_counter()
                        dur = t_now - t_prev
                        t_prev = t_now
                        if node not in stage_times:
                            stage_order.append(node)
                        stage_times[node] = stage_times.get(node, 0.0) + dur
                        stage_counts[node] = stage_counts.get(node, 0) + 1
                    progress.update(task_id, description=f"{node} ✓ · next stage…")
                    if isinstance(payload, dict):
                        result.update(payload)
    except KeyboardInterrupt:  # intentional Ctrl-C exit path
        err_console.print("[red]interrupted[/red]")
        raise typer.Exit(code=130) from None

    # Pass the store so `_print_answer` can resolve each [C<n>] to its
    # paper/section/quote. Honor --no-citations by passing store=None.
    store_for_citations = None if no_citations else getattr(app_, "store", None)
    _print_answer(result, store=store_for_citations)
    if timings:
        _print_timings(stage_order, stage_times, stage_counts)


def _brief(payload: dict) -> str:
    keys = {"sufficiency_decision", "intent", "retries", "critic_verdict", "qc_report"}
    bits = [f"{k}={payload[k]}" for k in keys if k in payload]
    if len(payload) > 5:
        bits.append(f"…({len(payload)} fields)")
    return " ".join(bits) or f"{len(payload)} fields"


def _print_timings(
    order: list[str],
    stage_times: dict[str, float],
    stage_counts: dict[str, int] | None = None,
) -> None:
    """Print a per-stage wall-clock breakdown, slowest stage first.

    `stage_counts` (call count per node, when available) lets the reader tell
    one genuinely slow call apart from a stage that ran several times (critic
    / revise / retry loops). When present, a `calls` and `avg` column are shown.
    """
    counts = stage_counts or {}
    total = sum(stage_times.values())
    if not stage_times:
        return
    table = Table(title="Stage timings (wall clock, slowest first)")
    table.add_column("stage", style="cyan")
    table.add_column("time", justify="right")
    table.add_column("share", justify="right")
    if counts:
        table.add_column("calls", justify="right")
        table.add_column("avg/call", justify="right")
    # Stable order: preserve execution order for equal times, but sort by time
    # descending so the bottleneck jumps out.
    for node in sorted(order, key=lambda n: (-stage_times[n], order.index(n))):
        secs = stage_times[node]
        pct = 100.0 * secs / total if total else 0.0
        # Flag anything over ~20% of the budget as the likely bottleneck.
        style = "[bold yellow]" if pct >= 20 else ""
        cells = [
            f"{style}{node}",
            f"{style}{secs:,.2f}s[/]" if style else f"{secs:,.2f}s",
            f"{style}{pct:,.1f}%[/]" if style else f"{pct:,.1f}%",
        ]
        if counts:
            n = max(counts.get(node, 1), 1)
            # Flag multi-run stages in red so critic/revise loops are obvious.
            call_style = "[bold red]" if n > 1 else ""
            cells.append(f"{call_style}{n}[/]" if call_style else str(n))
            cells.append(f"{secs / n:,.2f}s")
        table.add_row(*cells)
    total_row = ["[bold]total[/bold]", f"[bold]{total:,.2f}s[/bold]", "[bold]100%[/bold]"]
    if counts:
        total_row += [str(sum(counts.values())), ""]
    table.add_row(*total_row)
    console.print(table)


def _print_answer(result: dict, *, store=None) -> None:
    from rich.markdown import Markdown

    intent = result.get("intent")
    suff = result.get("sufficiency") or {}
    decision = result.get("sufficiency_decision")
    n_cite = len(result.get("citations", []))
    n_drop = len(result.get("citations_dropped", []))
    qc = result.get("qc_report") or {}

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("intent", str(intent))
    info.add_row("sufficiency", f"{suff.get('total', 0):.1f}/100")
    info.add_row("gate decision", str(decision))
    info.add_row("citations", f"{n_cite} verified / {n_drop} dropped")
    info.add_row("QC", "PASS" if qc.get("passed") else "FAIL (see answer caveats)")
    console.print(info)
    console.print()

    answer = result.get("answer") or "(no answer produced)"
    console.print(Markdown(answer))

    # Default-on citation table: every verified [C<n>] marker resolved to
    # its paper, section, a short quote, and the filename / DOI. Pass
    # `store=None` (e.g. from a test harness) to skip the resolution.
    if store is not None:
        _print_citation_table(result, store)


def _print_citation_table(result: dict, store) -> None:
    """Render the verified citation list as a human-readable table.

    Each row is `[C<n>] <paper title> — <section>  "<quote>"  <filename · DOI>`
    (the exact format requested: marker, paper title, section title, short
    quote, and the source identifier). Dropped citations — those the
    citation-integrity stage could not resolve to a real chunk — are listed
    below the table as a dimmed note so the reader knows which inline
    markers had no grounded source.
    """
    verified = result.get("citations") or []
    dropped = result.get("citations_dropped") or []

    if not verified and not dropped:
        return

    if verified:
        from pathlib import Path

        table = Table(
            title="Citations",
            title_style="bold",
            border_style="dim",
            show_lines=False,
            pad_edge=False,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Paper", ratio=2, overflow="fold")
        table.add_column("Section", ratio=1, overflow="fold")
        table.add_column("Quote", ratio=4, overflow="fold", style="italic")
        table.add_column("Source", ratio=2, overflow="fold", style="dim")

        # Stable order: by marker id (C1, C2, …). `id` may be int or str.
        def _key(c: dict) -> int:
            cid = c.get("id")
            try:
                return int(cid)
            except (TypeError, ValueError):
                return 999999

        from .graph.nodes.evidence_util import (
            _derived_paper_title,
            _is_weak_title_stem,
        )

        for c in sorted(verified, key=_key):
            cid = c.get("id")
            chunk_id = str(c.get("chunk_id", "")).strip()
            paper = None
            chunk = None
            if chunk_id:
                chunk = store.get_chunk(chunk_id)
                if chunk is not None:
                    paper = store.get_paper(chunk.paper_id)
            # Prefer a real title. When the stored title is just a filename
            # stem (e.g. `11299`), derive one from the paper's first chunk.
            if paper and _is_weak_title_stem(paper.title):
                derived = _derived_paper_title(paper, store)
                title = derived or paper.title
            else:
                title = (paper.title if paper and paper.title else (c.get("paper_id") or chunk_id))
            section = (chunk.heading if chunk else "") or "(no section)"
            quote = (chunk.text if chunk else (c.get("claim") or "")) or "—"
            quote = " ".join(str(quote).split())
            if len(quote) > 220:
                quote = quote[:220] + "…"
            src_parts: list[str] = []
            if paper and paper.path:
                src_parts.append(Path(paper.path).name)
            if paper and paper.doi:
                src_parts.append(paper.doi)
            src = " · ".join(src_parts) if src_parts else "—"
            table.add_row(f"C{cid}", title, section, f"“{quote}”", src)
        console.print(table)

    if dropped:
        console.print(
            f"\n[dim]{len(dropped)} citation(s) could not be verified and were dropped:[/dim]"
        )
        for d in dropped:
            cid = d.get("id", "?")
            reason = d.get("reason", "unresolved")
            console.print(f"  [dim]- C{cid}: {reason}[/dim]")
        console.print()


# ── papers (list the corpus tree as paper groups) ─────────────────────────
@app.command()
def papers(
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    group: str | None = typer.Option(
        None,
        "--group", "-g",
        help="Only show papers in this group (folder name).",
    ),
):
    """List the indexed corpus as a tree of paper groups.

    A paper's *group* is the name of its immediate parent folder — the same
    grouping `evidence ask --group` filters on. This is the fast, deterministic
    way to answer "what's in the corpus?" (no retrieval, no LLM): it reads the
    store directly.

    \b
    Examples:
      evidence papers                 # all groups + papers
      evidence papers -g finance      # only the 'finance' group
    """
    cfg = _load_config(config)
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {cfg.resolved_index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .graph.nodes.evidence_util import _best_title, group_from_path

    store = Store(cfg.resolved_store_path)
    try:
        rows = []
        for p in store.list_papers():
            grp = group_from_path(p.path) or "(root)"
            if group and grp != group:
                continue
            rows.append((grp, p))

        # Group rows by group, preserving store order within each group.
        by_group: dict[str, list] = {}
        for grp, p in rows:
            by_group.setdefault(grp, []).append(p)

        if not rows:
            console.print(
                "[yellow]No papers[/yellow]"
                + (f" in group [bold]{group}[/bold]." if group else ".")
            )
            return

        n_groups = len(by_group)
        n_papers = sum(len(v) for v in by_group.values())
        for grp, plist in sorted(by_group.items(), key=lambda kv: kv[0].lower()):
            table = Table(
                title=f"group: {grp}  ·  {len(plist)} paper(s)",
                title_style="bold cyan",
                border_style="dim",
                show_lines=False,
                pad_edge=False,
            )
            table.add_column("title", overflow="fold")
            table.add_column("path", style="dim", overflow="fold")
            table.add_column("chunks", justify="right")
            table.add_column("year", justify="right")
            table.add_column("doi", style="dim", overflow="fold")
            for p in plist:
                table.add_row(
                    _best_title(p, store) or p.title or p.path,
                    p.path,
                    str(store.count_chunks_for_paper(p.paper_id)),
                    str(p.year) if p.year else "",
                    p.doi or "",
                )
            console.print(table)

        console.print(
            f"\n[dim]{n_papers} paper(s) across {n_groups} group(s). "
            f"Use `evidence ask --group <name>` to restrict a question to a group.[/dim]"
        )
    finally:
        store.close()


# ── build-knowledge-graph ─────────────────────────────────────────────────
@app.command("build-knowledge-graph")
def build_knowledge_graph(
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    load_neo4j: bool = typer.Option(
        False, "--load-neo4j", help="Also load into a running Neo4j instance."
    ),
    author_online: bool = typer.Option(
        False, "--author-online", help="Resolve author IDs via OpenAlex (network)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build the offline knowledge graph (entities, citations, authors) over the index."""
    _setup_logging(verbose)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    console.print("[bold]Building knowledge graph[/bold] over "
                  f"[cyan]{index_dir}[/cyan]" + (" + Neo4j" if load_neo4j else ""))
    from .kg import build_kg

    store = Store(cfg.resolved_store_path)
    models = Models(cfg)
    try:
        report = build_kg(
            store,
            models=models,
            cfg=cfg,
            load_neo4j=load_neo4j,
            author_online=author_online,
        )
        summary = report.summary()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]knowledge-graph build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        store.close()

    table = Table(title="Knowledge-graph summary")
    table.add_column("metric", justify="left")
    table.add_column("value", justify="right")
    table.add_row("papers processed", str(summary["papers_processed"]))
    table.add_row("entity-extracted papers", str(len(summary["entities"])))
    table.add_row("papers with authors", str(len(summary["authors"])))
    table.add_row("citation links", str(summary["citations"]))
    table.add_row("co-citation pairs", str(summary["co_citation_pairs"]))
    neo = summary.get("neo4j")
    neo_text = "— (skipped)" if neo is None else f"{neo['papers']} papers / {neo['cites']} cites"
    table.add_row("neo4j", neo_text)
    if summary["skipped"]:
        table.add_row("skipped", f"{len(summary['skipped'])}")
    console.print(table)
    if summary["skipped"]:
        for s in summary["skipped"][:20]:
            err_console.print(f"  [yellow]skipped:[/yellow] {s}")
    console.print("[green]done.[/green]")


# ── eval ──────────────────────────────────────────────────────────────────
@app.command()
def eval(
    golden: Path = typer.Argument(
        ..., help="Path to a golden set (JSON array or JSONL)."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the full JSON report here."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the pipeline over a golden set and report retrieval + gate + faithfulness metrics."""
    _setup_logging(verbose)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)
    golden = golden.expanduser()
    if not golden.exists():
        err_console.print(f"[red]golden set not found:[/red] {golden}")
        raise typer.Exit(code=1)

    from .eval import run_eval

    console.print(f"[bold]Evaluating[/bold] over [cyan]{golden}[/cyan]")
    try:
        report = run_eval(cfg, str(golden))
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    agg = report["aggregate"]
    table = Table(title=f"Evaluation — {report['n_items']} item(s)")
    table.add_column("metric", justify="left")
    table.add_column("value", justify="right")
    for k in agg["recall"]:
        table.add_row(f"recall@{k}", f"{agg['recall'][k]:.3f}")
    table.add_row("MRR", f"{agg['mrr']:.3f}")
    for k in agg["ndcg"]:
        table.add_row(f"nDCG@{k}", f"{agg['ndcg'][k]:.3f}")
    table.add_row("gate accuracy", f"{agg['gate']['accuracy']:.3f}")
    table.add_row("citation faithfulness", f"{agg['citation_faithfulness']:.3f}")
    table.add_row("answer rate", f"{agg['answer_rate']:.3f}")
    console.print(table)

    if output:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[green]report written:[/green] {output}")


# ── review (structured scientific review of one paper) ──────────────────────
@app.command()
def review(
    paper: str = typer.Argument(
        ...,
        help="Paper id (exact store id) or a title substring to match.",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    external: bool | None = typer.Option(
        None,
        "--external/--no-external",
        help=(
            "Enable/disable external reference lookup (OpenAlex/S2/Crossref). "
            "Default follows `cfg.external.enabled`."
        ),
    ),
    resolve_limit: int = typer.Option(
        25, "--resolve-limit", help="Max references to check externally (high first)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the review as Markdown to this path."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit the full structured report as JSON."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Produce a structured scientific (peer) review of ONE paper.

    The review follows a fixed five-part structure:

      1. **Summary**        — problem, contribution, method, headline result.
      2. **Strengths**      — the paper's merits.
      3. **Weaknesses**     — its shortcomings.
      4. **Incorrect?**     — candidate factual errors / contradictions /
                            misused methods (each with a confidence + chunk).
      5. **Undefined terms** — terms/acronyms used without a definition.

    and a **References** table — the paper's reference list, each row with a
    relevance (high/medium/low) and role, plus whether the work was found in
    the external literature sources (when `--external` is on).

    \b
    Examples:
      evidence review "Attention-based Sequence Models"
      evidence review p1 --external            # also check references online
      evidence review p1 -o review.md          # write a Markdown report

    Read-only: it never modifies the corpus or the ledger.
    """
    _setup_logging(verbose)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .review import ReviewService, resolve_paper

    store = Store(cfg.resolved_store_path)
    try:
        resolved = resolve_paper(store, paper)
        if resolved is None:
            err_console.print(f"[red]paper not found:[/red] {paper}")
            console.print("[dim]Run `evidence papers` to list indexed papers.[/dim]")
            raise typer.Exit(code=1)
        pid, p = resolved
        console.print(
            f"[bold]Reviewing[/bold] [cyan]{p.title or p.path}[/cyan]"
            + (f" [dim]({pid})[/dim]" if pid != paper else "")
            + f"  [dim]model [cyan]{cfg.ollama.reasoning}[/cyan][/]"
            + (
                "  [dim]external references [bold]on[/bold][/]"
                if (cfg.external.enabled if external is None else external)
                else "  [dim]external references off[/dim]"
            )
        )
        report = ReviewService(cfg, store=store).review(
            pid, external=external, resolve_limit=resolve_limit
        )
    finally:
        store.close()

    if json_out:
        print(json.dumps(report.as_dict(), indent=2))
        return

    _print_review(report)

    if output:
        output.write_text(report.to_markdown(), encoding="utf-8")
        console.print(f"[green]review written:[/green] {output}")


# ── references (a paper's bibliography, structured + untruncated) ────────────
@app.command("references")
def references(
    paper: str = typer.Argument(
        ..., help="Paper id (exact) or a title substring."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Print a paper's reference list as clean, untruncated entries.

    Reads the References/Bibliography section (or the final chunks), joins it
    in document order, and splits it into per-entry strings (numbering kept).
    This is the structured, non-truncated way to read a bibliography — far more
    reliable than pulling fragments from `search_corpus`. Read-only, no model.

    \b
    Examples:
      evidence references "Attention-based Sequence Models"
      evidence references p1
    """
    _setup_logging(False)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import _resolve_paper
    from .indexes.store import Store
    from .review import references_text, split_references

    store = Store(cfg.resolved_store_path)
    try:
        p = _resolve_paper(store, paper)
        if p is None:
            err_console.print(f"[red]paper not found:[/red] {paper}")
            raise typer.Exit(code=1)
        chunks = store.chunks_for_paper(p.paper_id)
        entries = split_references(references_text(chunks))
        if not entries:
            console.print("[yellow]no reference section found in this paper's chunks[/yellow]")
            raise typer.Exit(code=0)
        console.print(
            Panel(
                f"[bold]{p.title or p.path}[/bold]",
                title=f"References · {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}",
                border_style="cyan", expand=False,
            )
        )
        for e in entries:
            console.print(f"[cyan]•[/cyan] {e}")
    finally:
        store.close()


# ── chunk (read one chunk in full) ──────────────────────────────────────────
@app.command()
def chunk(
    chunk_id: str = typer.Argument(..., help="A chunk id, e.g. `p1::c7` or `online__doi__...#c3`."),
    context: bool = typer.Option(
        True, "--context/--no-context", help="Show neighbouring chunks for local context."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Print ONE chunk's FULL, untruncated text + provenance (+ neighbours).

    The "give me chunk X" command: read an exact passage (a table row, a
    definition, a reference entry) that `search_corpus` only returned truncated.
    Read-only, no model.

    \b
    Examples:
      evidence chunk p1::c7
      evidence chunk p1::c7 --no-context
    """
    _setup_logging(False)
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {index_dir}.[/yellow]\n"
            f"Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import get_chunk

    rec = get_chunk(chunk_id)
    if rec is None:
        err_console.print(f"[red]chunk not found:[/red] {chunk_id}")
        raise typer.Exit(code=1)

    meta_bits = [rec.get("paper") or rec["paper_id"]]
    if rec.get("section"):
        meta_bits.append(rec["section"])
    if rec.get("page"):
        meta_bits.append(f"p.{rec['page']}")
    if rec.get("doi"):
        meta_bits.append(f"DOI {rec['doi']}")
    console.print(Panel(
        " · ".join(meta_bits),
        title=f"chunk {chunk_id} · {rec.get('tokens', '?')} tokens",
        border_style="cyan", expand=False,
    ))
    console.print(rec["text"])
    if context and rec.get("context"):
        console.print("\n[dim]── neighbouring chunks ──[/dim]")
        for c in rec["context"]:
            head = c.get("section") or "(no section)"
            if c.get("page"):
                head += f" · p.{c['page']}"
            console.print(f"[dim]{c['chunk_id']}[/dim] [dim]({head})[/dim]")
            console.print(f"  [dim]{c['text'][:200]}{'…' if len(c['text']) > 200 else ''}[/dim]")


# ── sections (hierarchical outline) ────────────────────────────────────────
@app.command()
def sections(
    paper: str = typer.Argument(
        ..., help="Paper id (exact) or a title substring."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Print a paper's sections as a coarse-to-fine navigation outline.

    Each section lists its heading, page, chunk count, token count, and the
    chunk ids that make it up. Follow up with `evidence section <paper> <sec>`
    to read a section in full. Read-only, no model.

    \b
    Examples:
      evidence sections p1
      evidence sections "FinLLM-KG"
    """
    _setup_logging(False)
    cfg = _load_config(config)
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {cfg.resolved_index_dir}[/yellow]\n"
            "Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import _resolve_paper, get_sections
    from .indexes.store import Store

    store = Store(cfg.resolved_store_path)
    try:
        p = _resolve_paper(store, paper)
        if p is None:
            err_console.print(f"[red]paper not found:[/red] {paper}")
            raise typer.Exit(code=1)
        secs = get_sections(p.paper_id, app=None)
        if not secs:
            err_console.print("[yellow]no sections found for this paper[/yellow]")
            raise typer.Exit(code=0)
        body = Table(title=f"{p.title or p.path} — {len(secs)} sections")
        body.add_column("#", justify="right", style="dim")
        body.add_column("section", style="cyan")
        body.add_column("page", justify="right")
        body.add_column("chunks", justify="right")
        body.add_column("tokens", justify="right")
        body.add_column("section_id")
        for i, s in enumerate(secs, start=1):
            body.add_row(
                str(i),
                s.get("heading") or "(untitled)",
                str(s.get("page") if s.get("page") is not None else "—"),
                str(s.get("n_chunks", 0)),
                str(s.get("tokens", 0)),
                f"[dim]{s['section_id']}[/dim]",
            )
        console.print(Panel(
            f"read one with: [bold]evidence section {p.paper_id} <section_id>[/bold]",
            border_style="cyan", expand=False,
        ))
        console.print(body)
    finally:
        store.close()


@app.command()
def section(
    paper: str = typer.Argument(..., help="Paper id (exact) or a title substring."),
    section_id: str = typer.Argument(
        ..., help="Section id (from `evidence sections`) or a heading substring."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Print ONE section as a single untruncated unit (all its chunks stitched).

    The hierarchical counterpart of `evidence chunk` — read a whole section
    (Method, Results, References, …) cleanly, in document order. Read-only.

    \b
    Examples:
      evidence section p1 sec2
      evidence section "FinLLM-KG" Method
    """
    _setup_logging(False)
    cfg = _load_config(config)
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {cfg.resolved_index_dir}[/yellow]\n"
            "Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import get_section

    rec = get_section(paper, section_id)
    if rec is None:
        err_console.print(f"[red]section not found:[/red] {paper} / {section_id}")
        raise typer.Exit(code=1)

    bits = [rec.get("paper") or rec["paper_id"]]
    if rec.get("page") is not None:
        bits.append(f"p.{rec['page']}")
    bits.append(f"{rec.get('n_chunks', '?')} chunks")
    console.print(Panel(
        " · ".join(bits),
        title=f"{rec.get('heading') or rec['section_id']} · {rec.get('tokens', '?')} tokens",
        border_style="cyan", expand=False,
    ))
    console.print(rec["text"])
    console.print(f"\n[dim]chunks: {', '.join(rec.get('chunk_ids', []))}[/dim]")


# ── chunk graph (links) ─────────────────────────────────────────────────────
@app.command("build-links")
def build_links(
    paper: str | None = typer.Argument(
        None, help="Scope to one paper (exact id or title substring). Omit for the whole corpus."
    ),
    reset: bool = typer.Option(
        True, "--reset/--no-reset", help="Wipe existing links first (scoped to `paper` if given)."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """(Re)build the chunk-graph: deterministic + (if available) LLM links.

    Always emits deterministic links (adjacent 'continuation', same-section
    'see_also'); adds semantic LLM links (see_also/depends_on/cites/contrasts/
    example) when a local reasoning model is reachable. Fail-soft — a model
    error just yields fewer links, never a crash.

    \b
    Examples:
      evidence build-links
      evidence build-links p1
      evidence build-links p1 --no-reset
    """
    _setup_logging(False)
    cfg = _load_config(config)
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {cfg.resolved_index_dir}[/yellow]\n"
            "Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import _resolve_paper
    from .indexes.store import Store
    from .links import build_chunk_links

    store = Store(cfg.resolved_store_path)
    try:
        paper_id = paper
        if paper_id is not None:
            p = _resolve_paper(store, paper_id)
            paper_id = p.paper_id if p else paper_id
        from .models.ollama import Models

        models = Models(cfg)
        report = build_chunk_links(
            store, models=models, cfg=cfg, paper_id=paper_id, reset=reset
        )
        s = report.summary()
        console.print(Panel(
            f"[green]✓[/green] processed [bold]{s['papers_processed']}[/bold] paper(s)\n"
            f"  links emitted: [bold]{s['links_emitted']}[/bold] "
            f"(deterministic {s['deterministic_links']}, LLM {s['llm_links']})\n"
            f"  total links in store: [bold]{store.count_chunk_links()}[/bold]",
            title="chunk-graph build",
            border_style="green" if s["papers_processed"] else "yellow", expand=False,
        ))
        if s.get("skipped"):
            console.print(f"[yellow]skipped:[/yellow] {', '.join(s['skipped'])}")
    finally:
        store.close()


@app.command()
def links(
    target: str = typer.Argument(
        ...,
        help="A chunk id (e.g. `p1::c3`) or a paper id / title substring.",
    ),
    direction: str = typer.Option(
        "both", "--direction", "-d", help="out | in | both (for chunk targets)."
    ),
    min_confidence: float = typer.Option(
        0.0, "--min-confidence", help="Drop links below this confidence."
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Show the chunk-graph around a chunk (its links + neighbours) or a paper.

    Auto-detects the target kind: if `target` is a chunk id it shows that
    chunk's neighbours (with relation + reason + preview); if it resolves to a
    paper it lists every link sourced from that paper, confidence-ordered,
    with short previews of both endpoints. Read-only.

    \b
    Examples:
      evidence links p1::c3
      evidence links p1::c3 --direction out
      evidence links p1
      evidence links "FinLLM-KG" --min-confidence 0.6
    """
    _setup_logging(False)
    cfg = _load_config(config)
    if not cfg.resolved_store_path.exists():
        err_console.print(
            f"[yellow]No index found at {cfg.resolved_index_dir}[/yellow]\n"
            "Run [bold]evidence ingest[/bold] first."
        )
        raise typer.Exit(code=1)

    from .facade import _resolve_paper
    from .indexes.store import Store
    from .links import linked_chunks, links_for_paper

    store = Store(cfg.resolved_store_path)
    try:
        # Chunk target?
        if store.get_chunk(target) is not None:
            res = linked_chunks(
                target,
                store=store,
                direction=direction,
                min_confidence=min_confidence,
            )
            if not res["neighbours"]:
                console.print(
                    "[yellow]no links found for this chunk yet — run "
                    "[bold]evidence build-links[/bold] first.[/yellow]"
                )
                raise typer.Exit(code=0)
            console.print(
                Panel(
                    f"chunk [bold]{target}[/bold] · {res['n_neighbours']} neighbour(s)",
                    title=f"links ({direction})", border_style="cyan", expand=False,
                )
            )
            for n in res["neighbours"]:
                rels = ", ".join(sorted({lk["relation"] for lk in n["links"]}))
                head = [n.get("paper_id") or "?", n.get("heading") or "(no heading)"]
                if n.get("page") is not None:
                    head.append(f"p.{n['page']}")
                head_s = " · ".join(head)
                console.print(
                    f"[cyan]→ {n['chunk_id']}[/cyan] [dim]({head_s})[/dim] "
                    f"[green]({rels})[/green]"
                )
                for lk in n["links"]:
                    if lk.get("reason"):
                        conf = lk["confidence"]
                        console.print(
                            f"    [dim]{lk['relation']} ({conf:.2f}): {lk['reason']}[/dim]"
                        )
                if n.get("preview"):
                    prev = n["preview"]
                    console.print(
                        f"    [dim]{prev[:160]}{'…' if len(prev) > 160 else ''}[/dim]"
                    )
            raise typer.Exit(code=0)

        # Paper target.
        p = _resolve_paper(store, target)
        if p is None:
            err_console.print(f"[red]not a known chunk or paper:[/red] {target}")
            raise typer.Exit(code=1)
        res = links_for_paper(p.paper_id, store=store, min_confidence=min_confidence)
        if not res["links"]:
            console.print(
                "[yellow]no links found for this paper yet — run "
                "[bold]evidence build-links[/bold] first.[/yellow]"
            )
            raise typer.Exit(code=0)
        console.print(Panel(
            f"[bold]{p.title or p.path}[/bold] · {res['n_links']} link(s)",
            title="chunk-graph", border_style="cyan", expand=False,
        ))
        for ln in res["links"]:
            src = (ln.get("src_preview") or "")[:60].replace("\n", " ")
            dst = (ln.get("dst_preview") or "")[:60].replace("\n", " ")
            cross = (
                ""
                if ln.get("dst_paper_id") == p.paper_id
                else f" → {ln['dst_paper_id']}"
            )
            console.print(
                f"[green]{ln['relation']}[/green] [{ln['confidence']:.2f}] "
                f"[dim]{src}[/dim] ⤳ [dim]{dst}[/dim]{cross}"
            )
            if ln.get("reason"):
                console.print(f"    [dim]{ln['reason']}[/dim]")
    finally:
        store.close()


def _print_review(report) -> None:
    """Render a ReviewReport to the terminal (readable, no markdown fences)."""
    from rich.markdown import Markdown

    # Header meta line.
    meta_bits: list[str] = []
    if report.authors:
        a = report.authors
        meta_bits.append(a[0] + (" et al." if len(a) > 1 else ""))
    if report.year:
        meta_bits.append(str(report.year))
    if report.doi:
        meta_bits.append(f"DOI {report.doi}")
    meta = " · ".join(meta_bits)
    console.print(Panel(
        f"[bold]{report.title or report.paper_id}[/bold]"
        + (f"\n[dim]{meta}[/dim]" if meta else "")
        + f"\n[dim]{report.n_chunks} indexed section(s)[/dim]",
        title="Scientific Review",
        border_style="cyan",
        expand=False,
    ))

    # 1. summary
    console.print("\n[bold cyan]1 · Summary[/bold cyan]")
    console.print(Markdown(report.summary or "_(no summary)_"))

    # 2. strengths
    console.print("\n[bold green]2 · Strengths[/bold green]")
    for s in report.strengths or ["_(none recorded)_"]:
        console.print(f"  [green]+[/green] {s}")

    # 3. weaknesses
    console.print("\n[bold yellow]3 · Weaknesses[/bold yellow]")
    for w in report.weaknesses or ["_(none recorded)_"]:
        console.print(f"  [yellow]−[/yellow] {w}")

    # 4. incorrect statements
    console.print("\n[bold red]4 · Possibly incorrect statements[/bold red]")
    if report.incorrect_statements:
        conf_style = {"high": "red", "medium": "yellow", "low": "dim"}
        for s in report.incorrect_statements:
            conf = str(s.get("confidence") or "").lower()
            style = conf_style.get(conf, "white")
            stmt = s.get("statement", "")
            problem = s.get("problem", "")
            loc = s.get("chunk_id", "") or s.get("context", "") or "—"
            console.print(f"  [{style}]{conf or '?':>6}[/]")
            console.print(f"      [bold]{stmt}[/bold]")
            if problem:
                console.print(f"      [dim]{problem}[/dim]")
            console.print(f"      [dim]{loc}[/dim]")
        console.print(
            "  [dim]Model-flagged candidates — verify each against the paper.[/dim]"
        )
    else:
        console.print("  [dim]None flagged.[/dim]")

    # 5. undefined terms
    console.print("\n[bold cyan]5 · Undefined terms[/bold cyan]")
    if report.undefined_terms:
        for t in report.undefined_terms:
            term = t.get("term", "")
            ctx = t.get("context", "") or ""
            sug = t.get("suggestion", "") or ""
            line = f"  [bold]{term}[/bold]"
            if ctx:
                line += f"  [dim]{ctx}[/dim]"
            console.print(line)
            if sug:
                console.print(f"      [dim]→ {sug}[/dim]")
    else:
        console.print("  [dim]None identified.[/dim]")

    # 6. references
    console.print("\n[bold cyan]6 · References[/bold cyan]")
    if report.references:
        table = Table(pad_edge=False, show_lines=False)
        table.add_column("#", justify="right", width=4, style="dim")
        table.add_column("Reference", ratio=3, overflow="fold")
        table.add_column("Year", width=5, justify="right")
        table.add_column("Relevance", width=9)
        table.add_column("Role", ratio=1, overflow="fold")
        table.add_column("Resolved", ratio=2, overflow="fold", style="dim")
        rel_style = {"high": "green", "medium": "yellow", "low": "dim"}
        for r in report.references:
            ref = r.get("title") or r.get("authors") or "—"
            if r.get("authors") and r["authors"] != r.get("title"):
                ref += f" — {r['authors']}"
            rel = r.get("relevance", "")
            resolved = r.get("resolved") or {}
            if resolved.get("found") is True:
                res = "[green]found[/green]"
                if resolved.get("citations") is not None:
                    res += f" ({resolved['citations']} cites)"
            elif resolved.get("found") is False:
                res = "[red]not found[/red]"
            else:
                res = "[dim]—[/dim]"
            table.add_row(
                str(r.get("index", "")),
                ref + (f"\n[dim]{r['doi']}[/dim]" if r.get("doi") else ""),
                str(r.get("year") or ""),
                f"[{rel_style.get(rel, 'white')}]{rel}[/]",
                r.get("role", ""),
                res,
            )
        console.print(table)
    else:
        console.print("  [dim]No references identified.[/dim]")

    if report.notes:
        console.print("\n[dim]── notes ──[/dim]")
        for n in report.notes:
            console.print(f"[dim]{n}[/dim]")
    console.print(
        "\n[dim]Tip: write this review to a file → "
        "[bold]evidence review <paper> -o review.md[/bold][/dim]"
    )


# ── status ────────────────────────────────────────────────────────────────
@app.command()
def status(
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Show corpus / index / model summary."""
    cfg = _load_config(config)
    index_dir = cfg.resolved_index_dir
    store_path = cfg.resolved_store_path

    console.print(f"[bold]index dir:[/bold] {index_dir}")
    table = Table()
    table.add_column("item")
    table.add_column("value", justify="right")
    if store_path.exists():
        store = Store(store_path)
        table.add_row("papers indexed", str(store.count_papers()))
        table.add_row("chunks indexed", str(store.count_chunks()))
        table.add_row("embedding dim", str(store.kv_get("index_dim", "?")))
        store.close()
    else:
        table.add_row("store", "(not built — run `evidence ingest`)")
    table.add_row(
        "dense index (FAISS)",
        "present" if (index_dir / "chunks.faiss").exists() else "absent",
    )
    table.add_row(
        "sparse index (BM25)",
        "present" if (index_dir / "bm25.pkl").exists() else "absent",
    )
    console.print(table)
    console.print(f"\n[bold]models:[/bold] reasoning={cfg.ollama.reasoning}  "
                  f"light={cfg.ollama.light}  embed={cfg.ollama.embed}")
    console.print(f"[bold]retrieval:[/bold] candidates={cfg.retrieval.candidates}  "
                  f"keep_after_rerank={cfg.retrieval.keep_after_rerank}  "
                  f"rrf_k={cfg.retrieval.rrf_k}")
    console.print(f"[bold]sufficiency:[/bold] threshold={cfg.sufficiency.threshold}  "
                  f"min_items={cfg.sufficiency.min_evidence_items}")


# ── config ────────────────────────────────────────────────────────────────
@app.command()
def config(
    path: str | None = typer.Argument(None, help="Optional YAML config file to load."),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty JSON output."),
):
    """Print the resolved configuration as JSON."""
    cfg = _load_config(path)
    indent = 2 if pretty else None
    console.print_json(cfg.model_dump_json(indent=indent))


# ── symbols (fast symbol inventory) ─────────────────────────────────────────
@app.command()
def symbols(
    scope: Path = typer.Argument(
        Path("."),
        help="Directory of Python code to scan (defaults to CWD).",
    ),
    kind: str = typer.Option(
        None, "--kind", "-k",
        help="Filter by kind: function | class | const.",
    ),
    prefix: str = typer.Option(
        None, "--prefix", "-p", help="Keep only symbols whose name starts with this.",
    ),
    limit: int = typer.Option(
        0, "--limit", "-n", help="Max symbols to print (0 = all).",
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON object per line (symbol, kind, module).",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """List the top-level symbols in a code scope (fast, no imports).

    Uses the same AST walk the ledger and `populate` use, so the printed names
    are exactly the keys `record_evidence` / `ledger impact` expect. Stdlib
    only — it parses, never imports.
    """
    _setup_logging(False)
    cfg = _load_config(config)
    scope = scope.expanduser().resolve()
    if not scope.exists():
        err_console.print(f"[red]scope not found:[/red] {scope}")
        raise typer.Exit(code=1)

    from .populate import scan_symbols

    algo = cfg.ledger.fingerprint_algo
    window = cfg.ledger.fingerprint_window
    syms = scan_symbols(scope, algo=algo, window=window, max_symbols=0)

    if kind:
        syms = [s for s in syms if s.kind == kind]
    if prefix:
        syms = [s for s in syms if s.name.split(".")[-1].startswith(prefix)]
    if limit > 0:
        syms = syms[:limit]

    if json_out:
        for s in syms:
            print(json.dumps({"symbol": s.name, "kind": s.kind, "module": s.module}))
    else:
        if not syms:
            console.print("[dim]no symbols found[/dim]")
            return
        for s in syms:
            console.print(f"{s.name}  [dim]({s.kind})[/dim]")
        console.print(f"\n[dim]{len(syms)} symbol(s)[/dim]")


# ── populate (autonomous decision-record population) ─────────────────────────
@app.command()
def populate(
    scope: Path = typer.Argument(
        Path("."),
        help="Directory of Python code to scan for design decisions.",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    limit: int = typer.Option(8, "--limit", "-n", help="Max symbols to populate."),
    force: bool = typer.Option(
        False, "--force", help="Re-draft and overwrite records that already exist."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Draft + ground, print the report, but record nothing."
    ),
    external: bool | None = typer.Option(
        None,
        "--external/--no-external",
        help=(
            "Enable/disable external literature search (OpenAlex/S2/Crossref). "
            "Default follows `cfg.external.enabled`."
        ),
    ),
    no_interactive: bool = typer.Option(
        False, "--no-interactive", help="Skip the one-by-one conflict prompts."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Autonomously draft, ground, and record design decisions for a code scope.

    Scans the scope for top-level symbols, uses the local reasoning model to
    draft a decision record per symbol, grounds each in real evidence
    (corpus + external literature), records it in the ledger, then runs a
    drift audit and interactively walks any questionable/conflicting decisions
    with you. Never edits source files.
    """
    _setup_logging(verbose)
    cfg = _load_config(config)
    scope = scope.expanduser().resolve()
    if not scope.exists():
        err_console.print(f"[red]scope not found:[/red] {scope}")
        raise typer.Exit(code=1)
    if not scope.is_dir():
        err_console.print(
            f"[yellow]'{scope}' is not a directory.[/yellow] Point populate at a folder "
            f"of Python code (e.g. [cyan]evidence populate src[/cyan])."
        )
        raise typer.Exit(code=2)

    from .populate import PopulateService, draft_payload_preview

    console.print(
        f"[bold]Populating evidence[/bold] in [cyan]{scope}[/cyan]"
        f"  (model [cyan]{cfg.ollama.reasoning}[/cyan]"
        + (", [bold]dry-run[/bold]" if dry_run else "")
        + (", [bold]forced[/bold]" if force else "")
        + ")"
    )
    svc = PopulateService(cfg)
    report = svc.run(
        scope,
        max_symbols=limit,
        force=force,
        dry_run=dry_run,
        external=external,
    )

    # Per-record detail.
    for pr in report.recorded:
        tag = "[cyan]updated[/cyan]" if pr.existed_before else "[green]new[/green]"
        grounded = (
            "[green]grounded[/green]" if pr.grounded else "[yellow]un-grounded[/yellow]"
        )
        console.print(f"  {tag} [bold]{pr.symbol}[/bold]  [{grounded}]")
        console.print(f"    decision: {pr.decision}")
        if pr.evidence:
            refs = ", ".join(e["ref_key"] for e in pr.evidence[:4])
            console.print(f"    evidence: {refs}")
        for note in pr.notes:
            console.print(f"    [dim]{note}[/dim]")
    for name in report.skipped:
        console.print(f"  [dim]skipped (exists): {name}[/dim]")

    if dry_run:
        console.print("\n[bold]Dry-run previews[/bold] (not written):")
        for pr in report.recorded:
            console.print(draft_payload_preview(pr))
        return

    # Audit summary.
    a = report.audit or {}
    table = Table(title="Ledger audit (after populate)")
    table.add_column("verdict", justify="left")
    table.add_column("count", justify="right")
    for key in ("fresh", "stale_code", "stale_evidence", "stale_dependency", "orphaned"):
        table.add_row(key, str(a.get(key, 0)))
    table.add_row("needs re-analysis", str(len(a.get("needs_reanalysis") or [])))
    console.print(table)

    if not report.conflicts:
        console.print("[green]No conflicts detected — every decision is grounded "
                      "and fresh.[/green]")
        _suggest_export(console)
        return

    _print_conflicts(report.conflicts)
    if no_interactive:
        err_console.print(
            "[yellow]--no-interactive: skipping conflict resolution.[/yellow] "
            "Run again without it to resolve one-by-one."
        )
        return
    _resolve_conflicts_interactive(svc, report, external=external)
    _suggest_export(console)


def _suggest_export(console_) -> None:
    console_.print(
        "\n[dim]Tip: view the ledger (color) → [bold]evidence ledger export[/bold][/dim]\n"
        "[dim]The review file is kept current automatically at "
        ".evidence/ledger.md (committable); or write a copy with "
        "[bold]evidence ledger export -f markdown -o docs/ledger.md[/bold][/dim]"
    )


def _print_conflicts(conflicts) -> None:
    table = Table(title="Conflicts to review")
    table.add_column("kind")
    table.add_column("symbol(s)")
    table.add_column("detail")
    for c in conflicts:
        table.add_row(c.kind, ", ".join(c.symbols), c.detail)
    console.print(table)


_KIND_STYLE = {
    "shared_evidence": "yellow",
    "no_evidence": "red",
    "stale_code": "magenta",
    "stale_evidence": "orange3",
    "stale_dependency": "red",
    "orphaned": "red",
}


def _corpus_store(svc):
    """Best-effort access to the local corpus store (for readable chunk refs)."""
    app = svc._app_or_none()
    return app.store if app is not None else None


def _shared_map(ledger) -> dict[str, list[str]]:
    """Map each evidence ref_key → sorted list of symbols whose records cite it."""
    by_ref: dict[str, set[str]] = {}
    try:
        for rec in ledger.list_records():
            for ref in rec.get("evidence", []):
                rk = ref.get("ref_key")
                if rk:
                    by_ref.setdefault(rk, set()).add(rec["symbol"])
    except Exception:  # noqa: BLE001 - ledger unavailable; best-effort
        pass
    return {k: sorted(v) for k, v in by_ref.items()}


def _doi_title_map(store) -> dict[str, str]:
    """Map lowercased DOI → paper title from the local corpus.

    Fills in paper names for evidence refs that were recorded as a bare DOI
    (no title), so the reviewer sees the work's name, not just the identifier.
    Best-effort: returns {} when there's no store.
    """
    out: dict[str, str] = {}
    if store is None:
        return out
    try:
        for paper in store.list_papers():
            if paper.doi:
                out[paper.doi.strip().lower()] = paper.title
    except Exception:  # noqa: BLE001 - best-effort
        pass
    return out


def _corpus_store_from_cfg(cfg):
    """Open the corpus store for resolving refs to human names, or None."""
    try:
        from .indexes.store import Store

        return Store(cfg.resolved_store_path)
    except Exception:  # noqa: BLE001 - no corpus / no index yet
        return None


def _make_doi_resolver(cfg, store):
    """A lazy, cached ``doi -> {"title","year"}`` resolver for human labels.

    Consults the local corpus first (no network), then Crossref/OpenAlex, and
    caches the result in the store's kv table so a given DOI is resolved once.
    Returns None when the feature isn't usable (no store, or module missing).
    """
    if store is None:
        return None
    try:
        from .external.online_corpus import resolve_doi_meta
    except Exception:  # noqa: BLE001
        return None

    def _resolve(doi: str) -> dict:
        try:
            return resolve_doi_meta(doi, cfg=cfg, store=store)
        except Exception:  # noqa: BLE001 - best-effort
            return {}

    return _resolve


def _evidence_label(e: dict, store, doi_titles: dict[str, str], doi_resolver=None) -> str:
    """Human-readable, PLAIN-TEXT label for one evidence ref (no markup).

    Local chunks → `Paper Title · Section (p.N)`; DOIs → `Paper Name (DOI: …)`;
    model-cited works → their name. Never returns a bare chunk machine-key when
    a corpus is available — that key belongs only in the JSON export / DB.

    ``doi_resolver``, when given, is consulted lazily for DOIs whose title we
    don't already have (from the ref itself or the corpus). The resolver
    caches its result (kv store), so the same DOI is resolved once and then
    served from cache.
    """
    rtype = e.get("ref_type", "chunk")
    key = e.get("ref_key", "")
    title = (e.get("title") or "").strip()

    if rtype == "chunk":
        if store is not None:
            rec = store.get_chunk(key)
            if rec is not None:
                paper = store.get_paper(rec.paper_id)
                ptitle = paper.title if paper else "(untitled paper)"
                sec = f" · {rec.heading}" if rec.heading else ""
                page = f" (p.{rec.page})" if rec.page else ""
                return f"{ptitle}{sec}{page}"
        return title or "(local evidence)"
    if rtype == "doi":
        k = key.strip().lower()
        name = title or doi_titles.get(k, "")
        if not name and doi_resolver is not None:
            try:
                r = doi_resolver(key)
                if r and r.get("title"):
                    name = r["title"]
            except Exception:  # noqa: BLE001 — resolver is best-effort
                pass
        return f"{name} (DOI: {key})" if name else f"DOI: {key}"
    # citation / unknown
    return f"{key} — {title}" if title else (key or "(evidence)")


def _evidence_paper_key(e: dict, store, doi_titles: dict[str, str]) -> str:
    """Grouping key for one ref: the paper title (when known) so that
    multiple chunks/sections from the same paper land in one group."""
    rtype = e.get("ref_type", "chunk")
    key = e.get("ref_key", "")
    title = (e.get("title") or "").strip()

    if rtype == "chunk":
        if store is not None:
            rec = store.get_chunk(key)
            if rec is not None:
                paper = store.get_paper(rec.paper_id)
                if paper is not None and paper.title:
                    return paper.title
                # Paper not in the store (or untitled): group by the paper_id
                # so all chunks from the same paper still land in one group.
                if rec.paper_id:
                    return rec.paper_id
        # No store / chunk not found: fall back to the title or the chunk key.
        return title or key
    if rtype == "doi":
        k = key.strip().lower()
        name = title or doi_titles.get(k, "")
        # The DOI itself is a stable, unambiguous key; the "DOI: …" prefix is
        # a display convention, not a grouping discriminator.
        return name or key
    return title or key or "(evidence)"


def _evidence_section_label(e: dict, store) -> str:
    """Per-ref label *within* its paper group (i.e. the section, not the paper).

    - chunk   → heading + page, or "(no section)" when the chunk has none
    - doi     → "(whole document)"
    - other   → the ref's own title, or "(evidence)"
    """
    rtype = e.get("ref_type", "chunk")
    if rtype == "chunk":
        if store is not None:
            rec = store.get_chunk(e.get("ref_key", ""))
            if rec is not None:
                sec = rec.heading or "(no section)"
                page = f" (p.{rec.page})" if rec.page else ""
                return f"{sec}{page}"
        return (e.get("title") or "").strip() or "(local evidence)"
    if rtype == "doi":
        return "(whole document)"
    return (e.get("title") or "").strip() or "(evidence)"


def _group_evidence(
    ev: list[dict], store, doi_titles: dict[str, str]
) -> list[tuple[str, list[dict]]]:
    """Group evidence refs by paper (stable first-seen order).

    Returns ``[(paper_key, [refs...]), ...]``. A paper with a single ref is
    still its own group — the caller decides whether to collapse it.
    """
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for e in ev:
        k = _evidence_paper_key(e, store, doi_titles)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(e)
    return [(k, buckets[k]) for k in order]


# ── merge chunk + DOI groups for the same paper ─────────────────────────

def _group_paper_ids(paper_key: str, refs: list[dict], store) -> set[str]:
    """All paper identifiers a group claims (paper_ids + DOIs), so that
    chunk and DOI groups for the same underlying paper can be merged.

    - chunk ref → its paper_id, plus the paper's DOI (if in the store).
    - DOI ref   → its normalized DOI, plus the paper_id of any paper in the
                  store saved under that DOI (e.g. via ``save_online_ref``).
    """
    ids: set[str] = set()
    for r in refs:
        rtype = r.get("ref_type", "chunk")
        rk = r.get("ref_key", "")
        if rtype == "chunk" and store is not None:
            rec = store.get_chunk(rk)
            if rec is not None:
                if rec.paper_id:
                    ids.add(rec.paper_id)
                paper = store.get_paper(rec.paper_id)
                if paper is not None and paper.doi:
                    ids.add(paper.doi.strip().lower())
        elif rtype == "doi" and rk:
            from .external.online_corpus import _normalize_doi

            n = _normalize_doi(rk)
            if n:
                ids.add(n)
                if store is not None:
                    p = store.get_paper(f"online__doi__{n}")
                    if p is not None:
                        ids.add(p.paper_id)
    return ids


def _pick_best_header(keys: list[str]) -> str:
    """Pick the most informative header from candidate keys.

    Preference: a paper title (non-DOI-looking key) over a bare DOI.
    Among same-kind candidates the longest wins.
    """
    if not keys:
        return ""
    import re

    doi_re = re.compile(r"^\d+\.\d+/.+$", re.IGNORECASE)
    non_doi = [k for k in keys if k and not doi_re.match(k)]
    pool = non_doi if non_doi else keys
    return max(pool, key=len)


def _merge_evidence_groups(
    groups: list[tuple[str, list[dict]]], store, doi_titles: dict[str, str]
) -> list[tuple[str, list[dict]]]:
    """Merge groups that share a paper (via paper_id or DOI).

    Union-find over group indices; two groups merge when their claimed
    paper-identifier sets intersect. Preserves first-seen order.
    """
    if len(groups) <= 1:
        return groups

    n = len(groups)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids_per_group = [_group_paper_ids(k, r, store) for k, r in groups]
    for i in range(n):
        for j in range(i + 1, n):
            if ids_per_group[i] & ids_per_group[j]:
                union(i, j)

    components: dict[int, list[tuple[str, list[dict]]]] = {}
    for i, (k, r) in enumerate(groups):
        components.setdefault(find(i), []).append((k, r))

    ordered_roots: list[int] = []
    seen_roots: set[int] = set()
    for i in range(n):
        root = find(i)
        if root not in seen_roots:
            ordered_roots.append(root)
            seen_roots.add(root)

    merged: list[tuple[str, list[dict]]] = []
    for root in ordered_roots:
        bucket = components[root]
        best_key = _pick_best_header([k for k, _ in bucket])
        all_refs: list[dict] = []
        seen_keys: set[str] = set()
        for _, refs in bucket:
            for ref in refs:
                rk = ref.get("ref_key", "")
                if rk and rk not in seen_keys:
                    all_refs.append(ref)
                    seen_keys.add(rk)
        merged.append((best_key, all_refs))
    return merged


# ── sort sections in document order ──────────────────────────────────────

def _sort_sections(refs: list[dict], store) -> list[dict]:
    """Sort refs within a group: chunks by ordinal (document order), then
    DOIs, then other refs. Stable within a class."""

    def _key(r: dict):
        rtype = r.get("ref_type", "chunk")
        rk = r.get("ref_key", "")
        if rtype == "chunk" and store is not None:
            rec = store.get_chunk(rk)
            if rec is not None:
                return (0, rec.ordinal or 0, rk)
        if rtype == "doi":
            return (1, 0, rk)
        return (2, 0, rk)

    return sorted(refs, key=_key)


# ── paper header with year / author ──────────────────────────────────────

def _paper_index(store) -> tuple[dict[str, object], dict[str, object]]:
    """Precompute title→paper and doi→paper maps (lowercased) for O(1) lookups."""
    by_title: dict[str, object] = {}
    by_doi: dict[str, object] = {}
    if store is None:
        return by_title, by_doi
    for p in store.list_papers():
        if p.title:
            by_title[p.title.lower()] = p
        if p.doi:
            by_doi[p.doi.strip().lower()] = p
    return by_title, by_doi


def _paper_header(paper_key: str, by_title, by_doi) -> str:
    """Enhance a paper_key with year + first author (when in the store).

    Returns the original key if no paper matches, or if there's nothing to add.
    """
    p = by_title.get(paper_key.lower()) or by_doi.get(paper_key.strip().lower())
    if p is None:
        return paper_key
    suffix = ""
    if p.year:
        suffix += f" ({p.year})"
    if p.authors:
        if len(p.authors) == 1:
            suffix += f", {p.authors[0]}"
        else:
            suffix += f", {p.authors[0]} et al."
    return f"{paper_key}{suffix}" if suffix else paper_key


# ── rendering helpers ────────────────────────────────────────────────────

_INLINE_SECTION_MAX = 3
_CAP_SECTIONS = 5


def _render_group_markdown(
    paper_key: str,
    refs: list[dict],
    store,
    doi_titles: dict[str, str],
    doi_resolver=None,
    by_title=None,
    by_doi=None,
) -> list[str]:
    """Markdown lines for one paper group.

    - 1 ref   → single bullet (full label).
    - 2-3 refs → single bullet with inline sections:
        ``**Paper (2021, Smith et al.)** (Intro, Methods)``.
    - >3 refs → bold header + one bullet per section, capped at 5 with
      ``…+N more`` for the tail.
    """
    refs = _sort_sections(refs, store)
    header = _paper_header(paper_key, by_title or {}, by_doi or {})

    if len(refs) == 1:
        label = _evidence_label(
            refs[0], store, doi_titles, doi_resolver=doi_resolver
        )
        return [f"  - {label}"]

    if len(refs) <= _INLINE_SECTION_MAX:
        sections = ", ".join(_evidence_section_label(e, store) for e in refs)
        return [f"  - **{header}** ({sections})"]

    out = [f"  - **{header}**"]
    shown = refs[:_CAP_SECTIONS]
    for e in shown:
        out.append(f"    - {_evidence_section_label(e, store)}")
    remaining = len(refs) - len(shown)
    if remaining > 0:
        out.append(f"    - …+{remaining} more")
    return out


def _markdown_evidence_group(
    ev: list[dict], store, doi_titles: dict[str, str], doi_resolver=None
) -> list[str]:
    """Markdown lines for a record's evidence, grouped by paper.

    Single-ref papers render as one bullet.  Multi-ref papers render compactly
    (inline ≤3 sections, else header + bullets capped at 5).  Chunk and DOI
    groups for the same paper are merged.  Suitable for ``group_fn`` in
    ``Ledger.export_markdown``.
    """
    by_title, by_doi = _paper_index(store)
    out: list[str] = []
    groups = _merge_evidence_groups(
        _group_evidence(ev, store, doi_titles), store, doi_titles
    )
    for paper_key, refs in groups:
        out.extend(
            _render_group_markdown(
                paper_key, refs, store, doi_titles,
                doi_resolver=doi_resolver,
                by_title=by_title, by_doi=by_doi,
            )
        )
    return out


def _render_evidence_groups_console(
    ev: list[dict], console, store, doi_titles: dict[str, str], doi_resolver=None
) -> None:
    """Render a record's evidence grouped by paper, to the terminal.

    Single-ref groups → one coloured bullet.  Multi-ref groups → a bold paper
    header (with year/author when known), then dim section bullets.  Compact
    inline for ≤3 sections.  Chunk and DOI groups for the same paper are merged.
    """
    by_title, by_doi = _paper_index(store)
    groups = _merge_evidence_groups(
        _group_evidence(ev, store, doi_titles), store, doi_titles
    )
    for paper_key, refs in groups:
        refs = _sort_sections(refs, store)
        header = _paper_header(paper_key, by_title, by_doi)

        if len(refs) == 1:
            e = refs[0]
            rtype = e.get("ref_type", "chunk")
            key = e.get("ref_key", "")
            label = _evidence_label(e, store, doi_titles, doi_resolver=doi_resolver)
            if rtype == "chunk":
                console.print(
                    f"    [green]●[/green] [bold]{label}[/bold]  [dim]{key}[/dim]"
                )
            elif rtype == "doi":
                console.print(f"    [magenta]◉[/magenta] {label}")
            else:
                console.print(f"    [blue]◆[/blue] {label}")
            continue

        if len(refs) <= _INLINE_SECTION_MAX:
            sections = ", ".join(_evidence_section_label(e, store) for e in refs)
            icon = (
                "[green]●[/green]"
                if refs[0].get("ref_type", "chunk") == "chunk"
                else "[magenta]◉[/magenta]"
            )
            console.print(
                f"    {icon} [bold]{header}[/bold] [dim]({sections})[/dim]"
            )
            continue

        icon = (
            "[green]●[/green]"
            if refs[0].get("ref_type", "chunk") == "chunk"
            else "[magenta]◉[/magenta]"
        )
        console.print(f"    {icon} [bold]{header}[/bold]")
        shown = refs[:_CAP_SECTIONS]
        for e in shown:
            console.print(f"        [dim]· {_evidence_section_label(e, store)}[/dim]")
        remaining = len(refs) - len(shown)
        if remaining > 0:
            console.print(f"        [dim]· …+{remaining} more[/dim]")


def _evidence_line(e: dict, store, doi_titles: dict[str, str], doi_resolver=None) -> str:
    """Colour-coded terminal line for one evidence ref (conflict view).

    Wraps the shared plain-text label from `_evidence_label` with an icon and
    keeps the raw ref_key in dim text (this is the working/review view, so the
    underlying key is shown alongside the human name).
    """
    rtype = e.get("ref_type", "chunk")
    key = e.get("ref_key", "")
    label = _evidence_label(e, store, doi_titles, doi_resolver=doi_resolver)

    if rtype == "chunk":
        return f"    [green]●[/green] [bold]{label}[/bold]  [dim]{key}[/dim]"
    if rtype == "doi":
        return f"    [magenta]◉[/magenta] {label}"
    return f"    [blue]◆[/blue] {label}"


def _print_conflict_block(
    idx, total, sym, kind, svc, store, shared, doi_titles, doi_resolver=None
) -> bool:
    """Render one conflict. Returns False if there's nothing to resolve."""
    style = _KIND_STYLE.get(kind, "white")
    console.print(f"\n[bold]{style}{'─' * 64}[/]")
    console.print(
        f"[bold white]{idx}/{total}[/bold white]  [bold cyan]{sym}[/bold cyan]   "
        f"[on {style} white]{kind}[/]"
    )
    rec = svc.ledger.lookup(sym)
    if rec is None:
        console.print("  [dim]no record found — nothing to resolve.[/]")
        return False

    console.print(f"  [bold green]DECISION  [/bold green]  {rec.get('decision', '')}")
    if rec.get("rationale"):
        console.print(f"  [bold blue]RATIONALE [/bold blue]  {rec['rationale']}")

    ev = rec.get("evidence") or []
    if ev:
        groups = _merge_evidence_groups(
            _group_evidence(ev, store, doi_titles), store, doi_titles
        )
        n_papers = len(groups)
        if n_papers == 1:
            count = f"({len(ev)} ref)"
        else:
            count = f"({len(ev)} refs · {n_papers} papers)"
        console.print(f"  [bold yellow]EVIDENCE  [/bold yellow]  {count}:")
        _render_evidence_groups_console(
            ev, console, store, doi_titles, doi_resolver=doi_resolver
        )
    else:
        console.print("  [bold red]EVIDENCE  [/bold red]  [red](none — un-grounded)[/]")

    # For shared evidence, make explicit who else leans on each shared ref.
    others: set[str] = set()
    n_shared = 0
    for e in ev:
        rk = e.get("ref_key")
        if not rk:
            continue
        who = shared.get(rk) or []
        if len(who) >= 2:
            n_shared += 1
            others.update(s for s in who if s != sym)
    if n_shared:
        console.print(
            f"  [bold red]SHARED    [/bold red]  {n_shared} ref(s) also used by:"
        )
        for o in sorted(others):
            console.print(f"      [cyan]· {o}[/cyan]")
        console.print(
            "  [dim]→ confirm each decision still needs this evidence, or "
            "re-ground one to point at its own source.[/dim]"
        )
    return True


def _resolve_conflicts_interactive(svc, report, *, external: bool | None = None) -> None:
    """Review every conflict first, then re-ground the chosen ones.

    Two phases so nothing is mutated until you've seen the whole picture:
      1. REVIEW  — render each decision colour-coded with human-readable
                   evidence and an explicit "shared by" view (no changes).
      2. RESOLVE — pick which to re-ground (number(s), `all`, or `none`),
                   then re-ground those. Only the ledger is mutated
                   (via `re_ground`); source files are never touched.
    """
    # De-dupe to the actionable set, in a stable order.
    seen: set[str] = set()
    ordered = []
    for c in report.conflicts:
        for sym in c.symbols:
            if sym in seen:
                continue
            seen.add(sym)
            ordered.append((sym, c.kind, c.detail))

    store = _corpus_store(svc)
    shared = _shared_map(svc.ledger)
    doi_titles = _doi_title_map(store)
    doi_resolver = _make_doi_resolver(getattr(svc, "cfg", None), store)

    # ── Phase 1: REVIEW (read-only) ────────────────────────────────────────
    console.print(f"\n[bold cyan]Conflicts to review[/bold cyan] — {len(ordered)} item(s).")
    console.print("[dim]Read them all first; nothing changes until the Resolve step.[/dim]\n")
    for i, (sym, kind, _detail) in enumerate(ordered, start=1):
        _print_conflict_block(
            i, len(ordered), sym, kind, svc, store, shared, doi_titles,
            doi_resolver=doi_resolver,
        )

    if not ordered:
        console.print("\n[dim]Nothing to resolve.[/dim]\n")
        return

    # ── Phase 2: RESOLVE (mutates the ledger only for chosen items) ────────
    console.print("\n[bold]Resolve[/bold] — which to re-ground? "
                  "[dim]number(s) e.g. 1 3, 'all', or 'none'[/dim]")
    choice = typer.prompt(
        "  Re-ground", default="all",
        show_default=False,
    ).strip().lower()

    if choice in ("none", "n", "no", ""):
        console.print("  [dim]Nothing re-grounded — ledger left unchanged.[/dim]\n")
        return

    if choice in ("all", "a", "yes", "y"):
        selected = list(range(1, len(ordered) + 1))
    else:
        selected = []
        for tok in choice.replace(",", " ").split():
            try:
                selected.append(int(tok))
            except ValueError:
                err_console.print(f"  [yellow]ignoring '{tok}' — expected a number.[/]")
        selected = [n for n in selected if 1 <= n <= len(ordered)]
        if not selected:
            console.print("  [dim]No valid selection — ledger left unchanged.[/dim]\n")
            return

    for n in selected:
        sym = ordered[n - 1][0]
        console.print(f"\n  [bold]{n}. {sym}[/bold]  [dim]re-grounding…[/dim]")
        try:
            new = svc.re_ground(sym, external=external)
            console.print(
                f"      [green]✓[/green] {len(new.evidence)} evidence ref(s)"
            )
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"      [red]re-ground failed:[/red] {exc}")
    console.print()


# ── ledger (evidence-link projection for git review) ────────────────────────
_STATUS_STYLE = {
    "fresh": "green",
    "stale_code": "yellow",
    "stale_evidence": "yellow",
    "stale_dependency": "red",
    "orphaned": "red",
}
_STATUS_ICON = {
    "fresh": "✓",
    "stale_code": "↻",
    "stale_evidence": "⚠",
    "stale_dependency": "↓",
    "orphaned": "✗",
}


def _render_ledger_terminal(
    records: list[dict], console: Console, store=None, doi_titles=None, doi_resolver=None
) -> None:
    """Print a color-coded, human-readable ledger view to the terminal.

    Status is colour-coded (green=fresh, yellow=stale, red=orphaned/dependency);
    evidence is shown as document names + sections (human-readable), not raw
    machine keys; ungrounded records are flagged.
    """
    doi_titles = doi_titles or {}
    if not records:
        console.print(Panel("[dim]No decision records yet.[/dim]", title="Evidence Ledger"))
        return

    # Header with a colour-coded status summary.
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.get("status", "")] = by_status.get(r.get("status", ""), 0) + 1
    bits = []
    for status in ("fresh", "stale_code", "stale_evidence", "stale_dependency", "orphaned"):
        if by_status.get(status):
            icon = _STATUS_ICON.get(status, "•")
            bits.append(f"[{_STATUS_STYLE[status]}]{icon} {by_status[status]} {status}[/]")
    summary = "   ".join(bits)
    console.print(
        f"[bold]Evidence Ledger[/bold]  [dim]({len(records)} record(s))[/dim]   {summary}"
    )
    console.print()

    by_title, by_doi = _paper_index(store)
    for r in records:
        status = r.get("status", "")
        style = _STATUS_STYLE.get(status, "dim")
        icon = _STATUS_ICON.get(status, "•")
        ev = r.get("evidence") or []
        ungrounded = not ev

        # Count: refs + papers (papers computed after merge).
        n_groups = len(
            _merge_evidence_groups(_group_evidence(ev, store, doi_titles), store, doi_titles)
        ) if ev else 0
        if ev and n_groups == 1:
            count = f"({len(ev)} ref)"
        elif ev:
            count = f"({len(ev)} refs · {n_groups} papers)"
        else:
            count = "(0 evidence)"

        lines: list[str] = [
            f"[bold]{icon} [cyan]{r['symbol']}[/cyan]"
            f"  [{style}]{status}[/]"
            f"  [dim]{count}[/]"
            + ("  [yellow]un-grounded[/]" if ungrounded else "")
        ]
        decision = (r.get("decision") or "").strip()
        if decision:
            lines.append(f"    decision   {decision}")
        rationale = (r.get("rationale") or "").strip()
        if rationale:
            lines.append(f"    [dim]rationale  {rationale}[/]")

        cited = r.get("cited") or []
        if cited:
            def _cite_label(c: object) -> str:
                if not isinstance(c, dict):
                    return str(c)
                name = c.get("name") or c.get("doi") or "?"
                authors = f" ({c['authors']})" if c.get("authors") else ""
                doi = f" [`{c['doi']}`]" if c.get("doi") else " [unresolved]"
                return f"{name}{authors}{doi}"

            cite_txt = "  ".join(_cite_label(c) for c in cited[:4])
            lines.append(f"    [dim]cited      {cite_txt}[/]")

        deps = r.get("depends_on") or []
        if deps:
            dep_txt = "  ".join(f"[blue]{d}[/]" for d in deps)
            lines.append(f"    [dim]depends    {dep_txt}[/]")

        ev = r.get("evidence") or []
        if ev:
            groups = _merge_evidence_groups(
                _group_evidence(ev, store, doi_titles), store, doi_titles
            )
            for paper_key, refs in groups:
                refs = _sort_sections(refs, store)
                header = _paper_header(paper_key, by_title, by_doi)
                if len(refs) == 1:
                    label = _evidence_label(
                        refs[0], store, doi_titles, doi_resolver=doi_resolver
                    )
                    lines.append(f"    [magenta]evidence   {label}[/magenta]")
                elif len(refs) <= _INLINE_SECTION_MAX:
                    sections = ", ".join(
                        _evidence_section_label(e, store) for e in refs
                    )
                    lines.append(
                        f"    [magenta]evidence[/magenta]   "
                        f"[bold]{header}[/bold] [dim]({sections})[/dim]"
                    )
                else:
                    lines.append(f"    [magenta]evidence   [bold]{header}[/bold][/magenta]")
                    shown = refs[:_CAP_SECTIONS]
                    for e in shown:
                        sec = _evidence_section_label(e, store)
                        lines.append(f"      [dim]· {sec}[/dim]")
                    remaining = len(refs) - len(shown)
                    if remaining > 0:
                        lines.append(f"      [dim]· …+{remaining} more[/dim]")
        alts = r.get("alternatives") or []
        if alts:
            def _alt_label(a: object) -> str:
                if not isinstance(a, dict):
                    return str(a)
                name = (
                    a.get("name")
                    or a.get("option")
                    or a.get("description")
                    or a.get("decision")
                    or "?"
                )
                why = a.get("why_rejected") or a.get("rejected") or a.get("reason")
                return f"{name} ({why})" if why else str(name)

            alt_txt = "  ".join(_alt_label(a) for a in alts[:4])
            lines.append(f"    [dim]alternatives   {alt_txt}[/]")

        console.print(Panel("\n".join(lines), border_style=style, expand=False,
                            title=None))


ledger_app = typer.Typer(
    name="ledger",
    help="Inspect, export, and maintain the evidence ledger (decision records + links).",
    no_args_is_help=True,
)
app.add_typer(ledger_app, name="ledger")

# Git-hook body installed by `evidence ledger hook install`. Runs in the repo
# root on every checkout; the `python` line is self-locating so the hook works
# regardless of the venv/PATH.
_EVIDENCE_MARK = "# evidence-auto-update"
_LEDGER_HOOK = f"""#!/bin/sh
{_EVIDENCE_MARK}
# Auto-update the evidence ledger on checkout (added by `evidence ledger hook`).
# Silent on success; re-stamps drifted records and regenerates ledger.md.
python -m evidence.cli ledger auto-update >/dev/null 2>&1 || true
"""


@ledger_app.command("auto-update")
def ledger_auto_update(
    scope: Path = typer.Argument(
        Path("."),
        help="Code scope to scan (defaults to CWD).",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    json_out: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
):
    """Refresh the ledger's code fingerprints and re-export ``ledger.md``.

    Fast and local: it *accepts* code drift (re-stamps records whose symbol
    changed and marks them fresh), flags only genuine orphans / stale
    evidence, and writes a fresh ``.evidence/ledger.md``. No models, no
    network. This is what the git post-checkout hook runs.
    """
    _setup_logging(False)
    cfg = _load_config(config)
    scope = scope.expanduser().resolve()
    from .ledger import open_ledger

    lg = open_ledger(cfg)
    try:
        result = lg.audit(
            str(scope),
            algo=cfg.ledger.fingerprint_algo,
            window=cfg.ledger.fingerprint_window,
            refresh=True,
        )
        # Regenerate the review artifact alongside the ledger.
        md = lg.export_markdown()
    finally:
        lg.close()

    md_path = cfg.resolved_ledger_path.parent / "ledger.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    if json_out:
        print(json.dumps({**result, "ledger_md": str(md_path)}))
        return

    console.print(
        f"[green]✓[/green] ledger refreshed — "
        f"{result['total']} record(s): {result['fresh']} fresh, "
        f"{result['orphaned']} orphaned, {result['stale_evidence']} stale_evidence"
    )
    if result["needs_reanalysis"]:
        console.print(
            f"[yellow]⚠ {len(result['needs_reanalysis'])} record(s) need review:[/yellow]"
        )
        for item in result["needs_reanalysis"]:
            console.print(f"    [dim]{item['symbol']}: {item['status']}[/dim]")
    console.print(f"[dim]wrote {md_path}[/dim]")


def _find_git_dir(root: Path) -> Path | None:
    """The `.git` dir for `root`, walking up to the repo top (or None)."""
    for d in [root, *root.parents]:
        g = d / ".git"
        if g.exists():
            return g
    return None


def _install_ledger_hook(root: Path, event: str = "post-checkout") -> bool:
    """Install the auto-update hook (idempotent). Returns True on success."""
    git_dir = _find_git_dir(root)
    if git_dir is None:
        err_console.print("[red]not a git repo[/red] (no `.git/` found)")
        return False

    hook_path = git_dir / "hooks" / event
    hook_path.parent.mkdir(parents=True, exist_ok=True)

    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8")
        if _EVIDENCE_MARK in existing:
            console.print(f"[dim]hook already installed at {hook_path}[/dim]")
            return True
        if not existing.endswith("\n"):
            existing += "\n"
        existing += _LEDGER_HOOK
    else:
        existing = _LEDGER_HOOK
    hook_path.write_text(existing, encoding="utf-8")
    hook_path.chmod(0o755)
    return True


def _uninstall_ledger_hook(root: Path, event: str = "post-checkout") -> bool:
    """Remove the auto-update hook (leaving any other hook content)."""
    git_dir = _find_git_dir(root)
    if git_dir is None:
        err_console.print("[red]not a git repo[/red] (no `.git/` found)")
        return False

    hook_path = git_dir / "hooks" / event
    if not hook_path.exists():
        console.print(f"[dim]no {event} hook to remove[/dim]")
        return True
    existing = hook_path.read_text(encoding="utf-8")
    if _EVIDENCE_MARK not in existing:
        console.print(f"[dim]{event} hook has no evidence block[/dim]")
        return True

    new_lines: list[str] = []
    skipping = False
    for line in existing.splitlines(keepends=True):
        if _EVIDENCE_MARK in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "":
                skipping = False
            continue
        new_lines.append(line)
    if not new_lines:
        hook_path.unlink()
        console.print(f"[green]✓ removed {event} hook[/green]")
    else:
        hook_path.write_text("".join(new_lines), encoding="utf-8")
        hook_path.chmod(0o755)
        console.print(f"[green]✓ removed evidence block from {event} hook[/green]")
    return True


@ledger_app.command("hook")
def ledger_hook(
    action: str = typer.Argument(
        "install", help="install | uninstall"
    ),
    event: str = typer.Option(
        "post-checkout", "--event", "-e",
        help="git hook: post-checkout | post-merge | pre-commit | post-commit",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Install/remove the git hook that auto-updates the ledger on checkout.

    Default installs a ``post-checkout`` hook (re-stamps fingerprints +
    regenerates ``ledger.md`` every time you switch branches). Use
    ``uninstall`` to remove it.
    """
    root = Path.cwd()
    if action == "install":
        if not _install_ledger_hook(root, event):
            raise typer.Exit(code=1)
        console.print(f"[green]✓ installed {event} hook[/green]")
        console.print("[dim]the ledger will now auto-update on every checkout[/dim]")
    elif action == "uninstall":
        if not _uninstall_ledger_hook(root, event):
            raise typer.Exit(code=1)
    else:
        err_console.print(f"[red]unknown action:[/red] {action} (use install|uninstall)")
        raise typer.Exit(code=2)


@ledger_app.command("export")
def ledger_export(
    format: str | None = typer.Option(
        None, "--format", "-f",
        help="Write a file in this format: markdown | json. "
        "Omit to view the ledger in the terminal (default).",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Write the file here (default: `.evidence/ledger.<ext>` next to "
        "the ledger).",
    ),
    stdout: bool = typer.Option(
        False, "--stdout",
        help="Print the raw text (markdown by default) to stdout instead of "
        "a color view.",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """View or export the evidence ledger.

    Default: a color-coded, human-readable view in the terminal (green=fresh,
    yellow=stale, red=orphaned/dependency; un-grounded records are flagged).
    Pass `-f markdown|json` to write a diff-able artifact file, or `--stdout`
    to print the raw text (e.g. for piping).
    """
    from .ledger import open_ledger

    cfg = _load_config(config)
    lg = open_ledger(cfg)
    # Corpus store + DOI→title map for resolving evidence to human-readable
    # document names / sections. Machine keys stay in JSON + the DB.
    store = _corpus_store_from_cfg(cfg)
    doi_titles = _doi_title_map(store)
    doi_resolver = _make_doi_resolver(cfg, store)
    resolver = (
        (lambda e: _evidence_label(e, store, doi_titles, doi_resolver=doi_resolver))
        if store is not None else None
    )
    # Group evidence by paper in the markdown artifact (single-ref papers
    # collapse to one bullet; multi-ref papers render as paper + sections).
    group_fn = (
        (lambda ev: _markdown_evidence_group(
            ev, store, doi_titles, doi_resolver=doi_resolver))
        if store is not None else None
    )

    try:
        records = sorted(lg.list_records(), key=lambda r: r["symbol"])

        # 1) Raw text to stdout (for piping / scripting).
        if stdout:
            text = (
                lg.export_json()
                if format == "json"
                else lg.export_markdown(resolver=resolver, group_fn=group_fn)
            )
            print(text, end="")
            return

        # 2) Default: color-coded terminal view (human-readable evidence).
        if format is None:
            _render_ledger_terminal(
                records, console, store=store, doi_titles=doi_titles,
                doi_resolver=doi_resolver,
            )
            return

        # 3) Write a file.
        if format not in ("markdown", "json"):
            err_console.print(f"[red]unknown format:[/red] {format} (use markdown|json)")
            raise typer.Exit(code=2)
        text = (
            lg.export_json()
            if format == "json"
            else lg.export_markdown(resolver=resolver, group_fn=group_fn)
        )
    finally:
        lg.close()
        if store is not None:
            store.close()

    if output is None:
        suffix = "ledger.json" if format == "json" else "ledger.md"
        output = cfg.resolved_ledger_path.parent / suffix
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    console.print(f"[green]wrote[/green] [cyan]{output}[/cyan] "
                  f"({len(text.splitlines())} lines)")


@ledger_app.command("impact")
def ledger_impact(
    symbol: str = typer.Argument(
        ...,
        help="A symbol (e.g. `mod.func`), a bare chunk_id, or a DOI.",
    ),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """Show which other decisions would be affected by a change to `symbol`.

    Two signals are combined:

    - **shared_evidence** — other decisions citing the same chunk/DOI.
    - **depends_on**      — decisions that declared a dependency on `symbol`.

    Given a bare chunk_id or DOI (no record found), only `shared_evidence`
    applies.
    """
    from .ledger import open_ledger

    cfg = _load_config(config)
    lg = open_ledger(cfg)
    try:
        result = lg.impact(symbol)
    finally:
        lg.close()

    affected = result.get("affected", [])
    shared = result.get("shared_refs", [])

    if not affected and not shared:
        console.print(f"[yellow]No other decisions reference {symbol!r}.[/yellow]")
        return

    if shared:
        table = Table(title=f"Shared evidence for {symbol}")
        table.add_column("ref_key", style="cyan")
        table.add_column("type")
        table.add_column("source")
        table.add_column("used by", justify="right")
        for s in shared:
            table.add_row(
                s["ref_key"], s.get("ref_type", ""),
                s.get("source", ""), str(s.get("used_by", 1)),
            )
        console.print(table)

    if affected:
        table = Table(title="Affected decisions")
        table.add_column("symbol", style="cyan")
        table.add_column("decision")
        table.add_column("status")
        table.add_column("via")
        for a in affected:
            table.add_row(
                a["symbol"],
                a.get("decision", "")[:60],
                a.get("status", ""),
                ", ".join(a.get("via", [])),
            )
        console.print(table)
    else:
        console.print("[dim]No other decisions are directly affected.[/dim]")


@ledger_app.command("symbols")
def ledger_symbols(
    status: str = typer.Option(
        None, "--status", "-s",
        help="Filter by status: fresh | stale_code | stale_evidence | "
        "stale_dependency | orphaned.",
    ),
    ungrounded: bool = typer.Option(
        False, "--ungrounded", help="Only symbols with no supporting evidence.",
    ),
    json_out: bool = typer.Option(False, "--json", help="One JSON object per line."),
    config: str | None = typer.Option(None, "--config", "-c", help="YAML config file."),
):
    """List the symbols that have a decision record in the ledger.

    The reverse of `evidence symbols` (code → names): this is ledger → symbols,
    i.e. which functions/classes/constants already have a recorded design
    decision, their status, and how much evidence backs them.
    """
    from .ledger import open_ledger

    cfg = _load_config(config)
    lg = open_ledger(cfg)
    try:
        records = lg.list_records()
    finally:
        lg.close()

    if status:
        records = [r for r in records if r.get("status") == status]
    if ungrounded:
        records = [r for r in records if not r.get("evidence")]
    records = sorted(records, key=lambda r: r["symbol"])

    if json_out:
        for r in records:
            print(json.dumps({
                "symbol": r["symbol"],
                "status": r.get("status", ""),
                "evidence": len(r.get("evidence") or []),
                "decision": r.get("decision", ""),
            }))
        return

    if not records:
        console.print("[dim]no symbols have a decision record[/dim]")
        return

    table = Table(title=f"Symbols with decision records ({len(records)})")
    table.add_column("symbol", style="cyan")
    table.add_column("status")
    table.add_column("evidence", justify="right")
    table.add_column("decision")
    for r in records:
        ev = r.get("evidence") or []
        table.add_row(
            r["symbol"],
            r.get("status", ""),
            str(len(ev)),
            (r.get("decision", "") or "")[:50],
        )
    console.print(table)


if __name__ == "__main__":
    app()
