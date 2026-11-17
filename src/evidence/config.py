"""Central, swappable configuration for Evidence.

Everything tunable lives here as a pydantic model so it can be overridden by a
YAML file, environment variables, or programmatic construction. The paper's
rubric, model assignments, budgets, and thresholds are all exposed as config so
they can be calibrated without touching code (see PLAN.MD "Key risks").
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Connection + model assignments (see PLAN.MD "Local model assignments")."""

    host: str = Field(default="http://localhost:11434")
    # Heavy reasoning model (Planner/Writer/Critic, sufficiency review).
    reasoning: str = Field(default="qwen2.5:9b")
    # Light model (classify, rewrite, lightweight reasoning).
    light: str = Field(default="qwen2.5:4b")
    # Embedding model for FAISS dense vectors.
    embed: str = Field(default="bge-m3")
    # Optional reranker (sentence-transformers cross-encoder). Falls back to
    # the LLM when not installed / not configured.
    reranker_local: str = Field(default="BAAI/bge-reranker-v2-m3")
    request_timeout: float = Field(default=300.0)
    # Disable the model's "thinking" / chain-of-thought pass. Some local models
    # (e.g. qwen3.5, gemma3, deepseek-r1) burn their entire token budget on
    # hidden reasoning tokens, which with `format=json` yields an EMPTY output
    # and makes the pipeline fall back to deterministic drafts. Ignored by
    # models that don't think (qwen2.5, mistral, llama) and is a no-op there.
    think: bool = Field(default=False)
    # If a configured model is missing from the Ollama server, pull it from the
    # registry before first use. Best-effort: a failed/unavailable list or pull
    # never blocks the call — it just lets the underlying request surface the
    # usual "model not found" error.
    auto_pull: bool = Field(default=True)
    # How long Ollama should keep a loaded model in memory after its last use,
    # so it isn't re-loaded (a costly GPU/VRAM load, big for large models) on the
    # next request. Ollama's default is 5m; "0" unloads immediately, "24h"/
    # "168h" keeps it resident. Use "0" only if you need the VRAM freed fast.
    keep_alive: str = Field(default="1h")


class RetrievalConfig(BaseModel):
    """Hybrid retrieval budgets (PLAN.MD stages 3–4)."""

    candidates: int = Field(default=50, description="FAISS+BM25 candidates pre-fusion")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    keep_after_rerank: int = Field(default=15, description="chunks kept post-rerank")
    use_bm25: bool = Field(default=True)
    use_dense: bool = Field(default=True)
    # Weights for the (normalized) dense vs sparse scores fed into RRF. RRF is
    # rank-based so weights mostly affect tie-breaking; kept for clarity.
    dense_weight: float = Field(default=1.0)
    sparse_weight: float = Field(default=1.0)


class ChunkingConfig(BaseModel):
    """Structure-aware chunking (PLAN.MD offline pipeline, step 2)."""

    max_tokens: int = Field(default=800)
    overlap_tokens: int = Field(default=100)
    min_chunk_tokens: int = Field(default=40)


class SufficiencyConfig(BaseModel):
    """Evidence sufficiency gate — 100-pt rubric (PLAN.MD stage 7).

    Five dimensions, each scored 0–20. Threshold is the pass mark. Make these
    swappable; they are deliberately conservative for small local models.
    """

    dimensions: dict[str, float] = Field(
        default_factory=lambda: {
            "coverage": 20.0,
            "relevance": 20.0,
            "depth": 20.0,
            "recency": 20.0,
            "consistency": 20.0,
        }
    )
    threshold: float = Field(default=70.0)
    # If a chunk's relevance is below this, its score contribution is damped
    # (multiplied by `relevance_damp_factor`). Punishes weak chunks (PLAN.MD stage 7).
    relevance_floor: float = Field(default=0.35)
    relevance_damp_factor: float = Field(default=0.5)
    # Number of evidence items required before the rubric even runs; below this
    # we short-circuit to "insufficient" (cheap rule-based gate).
    min_evidence_items: int = Field(default=3)


class RetryConfig(BaseModel):
    """Agentic retry + external escalation budgets (PLAN.MD stage 8)."""

    max_retries: int = Field(default=2)
    # Drift guard: max cosine distance between original query embedding and a
    # reformulated query embedding. Reject reformulations above this.
    drift_max_distance: float = Field(default=0.6)
    max_query_variants: int = Field(default=3)


class GenerationConfig(BaseModel):
    """Planner/Researcher/Writer/Critic budgets (PLAN.MD stage 12–13)."""

    planner_temp: float = Field(default=0.4)
    researcher_temp: float = Field(default=0.2)
    writer_temp: float = Field(default=0.6)
    critic_temp: float = Field(default=0.2)
    critic_max_cycles: int = Field(default=2)
    critic_pass_threshold: float = Field(default=70.0)
    max_regenerations: int = Field(default=1)


class ExternalConfig(BaseModel):
    """External academic search (PLAN.MD stage 9, Phase 3)."""

    enabled: bool = Field(default=False)
    providers: list[str] = Field(default_factory=lambda: ["openalex", "s2", "crossref"])
    openalex_mailto: str | None = Field(default=None)
    s2_api_key: str | None = Field(default=None)
    s2_rps: int = Field(default=1)
    max_results_per_provider: int = Field(default=10)
    require_approval: bool = Field(default=False)
    # When an online reference (a DOI pulled from OpenAlex/S2/Crossref during
    # grounding) is used, save the FULL document (title + abstract + the cited
    # section/claim text) into the corpus's online part, so it resolves to a
    # readable "Paper · Section" later and is searchable. When False, only the
    # title + section/DOI is stored (no document capture). Best-effort: a
    # download/parse failure never blocks grounding — it just keeps the lighter
    # title+section record.
    save_online_to_corpus: bool = Field(default=True)


class GraphExpansionConfig(BaseModel):
    """Neo4j graph-guided chunk expansion (PLAN.MD stage 5, Phase 2)."""

    enabled: bool = Field(default=False)
    max_additional_chunks: int = Field(default=10)
    max_hops: int = Field(default=1)


class VisualConfig(BaseModel):
    """Visual page retrieval (PLAN.MD stage 6, Phase 4)."""

    enabled: bool = Field(default=False)
    model: str = Field(default="colsmol")  # or "clip" fallback
    top_pages: int = Field(default=5)
    candidates: int = Field(default=100)


class LedgerConfig(BaseModel):
    """Evidence ledger — the cache-first decision record store (new plan).

    Records the design decision behind a code symbol (function/class/API), its
    rationale, alternatives considered, and the corpus/external evidence that
    justifies it. `lookup_evidence` reads this BEFORE retrieval; `record_evidence`
    writes it after a justified change; `audit_evidence` flags records whose code
    or cited evidence has drifted.
    """

    enabled: bool = Field(default=True)
    # SQLite file. Defaults to `data_dir/ledger.sqlite` when unset.
    path: Path | None = Field(default=None)
    # Algorithm used to fingerprint a symbol's source for drift detection.
    fingerprint_algo: str = Field(default="sha256", description="sha256 | sha1 | md5")
    # Number of trailing source lines captured into the fingerprint. A full-file
    # hash is brittle (any unrelated edit invalidates the record), so we hash a
    # stable window around the symbol.
    fingerprint_window: int = Field(default=120)


class McpConfig(BaseModel):
    """MCP server behaviour (new plan: "Server implementation")."""

    # Gate for the state-changing `ingest_paper` tool. Off by default — see the
    # plan's "Read-only by default" principle.
    allow_ingest: bool = Field(default=False)
    # Truncate returned chunk text to this many chars (plan: keep outputs compact).
    max_chunk_chars: int = Field(default=1500)
    # Default result limits.
    default_search_limit: int = Field(default=8)
    default_citation_limit: int = Field(default=10)
    # When True, `answer_research` returns a raw *evidence bundle* (retrieved
    # chunks + sufficiency score + guidance) for the CALLING model to synthesise
    # a cited answer from — instead of running the local Planner/Writer/Critic
    # stack. Use this when the MCP client already has a capable model (e.g. VS
    # Code Copilot) and you don't want to also run Evidence's local reasoning
    # models. When False (default), `answer_research` runs the full local
    # pipeline and returns a finished, cited answer (headless / no-client-LLM
    # use).
    prefer_client_generation: bool = Field(default=False)


class EvidenceConfig(BaseModel):
    """Top-level config. Build via `EvidenceConfig.load()` to read YAML/env."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    sufficiency: SufficiencyConfig = Field(default_factory=SufficiencyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    external: ExternalConfig = Field(default_factory=ExternalConfig)
    graph_expansion: GraphExpansionConfig = Field(
        default_factory=GraphExpansionConfig
    )
    visual: VisualConfig = Field(default_factory=VisualConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

    # Storage locations.
    data_dir: Path = Field(default=Path.home() / ".evidence")
    index_dir: Path | None = Field(default=None, description="Defaults to data_dir/index")
    corpus_dir: Path | None = Field(default=None, description="Directory of PDF/LaTeX")
    sqlite_db: str = Field(default="store.sqlite", description="Store filename in index dir")
    checkpoint_dir: Path | None = Field(default=None)

    @property
    def resolved_index_dir(self) -> Path:
        base = self.index_dir or (self.data_dir / "index")
        return Path(base).expanduser()

    @property
    def resolved_store_path(self) -> Path:
        """The canonical store file location — single source of truth so every
        reader (ingest, ask, status, MCP) agrees on the same file."""
        return self.resolved_index_dir / self.sqlite_db

    @property
    def resolved_ledger_path(self) -> Path:
        base = self.ledger.path or (self.data_dir / "ledger.sqlite")
        return Path(base).expanduser()

    @classmethod
    def load(cls, path: str | Path | None = None) -> EvidenceConfig:
        """Load config from an optional YAML file, then env overrides.

        Env override convention: `EVIDENCE_<SECTION>_<FIELD>` (e.g.
        `EVIDENCE_OLLAMA_REASONING=qwen2.5:7b`).
        """
        from pydantic import ValidationError

        data: dict = {}
        # Allow pointing the MCP server (or any entry) at a config file via the
        # environment, without needing a CLI `--config` flag.
        env_cfg = os.environ.get("EVIDENCE_CONFIG")
        if path is None and env_cfg:
            path = env_cfg
        # Auto-discovery: walk up from CWD looking for `.evidence/config.yaml`
        # (created by `evidence init`). Lets the user run `evidence ask ...`
        # from any subdirectory of the project without `-c`.
        if path is None:
            path = _find_project_config()
        if path is not None:
            import yaml

            p = Path(path)
            if p.exists():
                with p.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                # Rebase relative path values against the *project root* (the
                # config's parent when it lives in `.evidence/`, else its own
                # directory). This makes the project work from any subdirectory
                # and keeps the index/ledger alongside the config — not in the
                # CWD the user happened to run from.
                base = p.parent.parent if p.parent.name == ".evidence" else p.parent
                data = _rebase_paths(data, base=base)

        # Environment overrides (best-effort, only top-level scalars/sections).
        for key, value in os.environ.items():
            if not key.startswith("EVIDENCE_"):
                continue
            parts = key[len("EVIDENCE_"):].lower().split("_", 1)
            if len(parts) == 2 and parts[0] in {
                "ollama",
                "retrieval",
                "chunking",
                "sufficiency",
                "retry",
                "generation",
                "external",
                "graph_expansion",
                "visual",
                "ledger",
                "mcp",
            }:
                section = data.setdefault(parts[0], {})
                field = parts[1]
                # Try to coerce to the type expected by the model.
                section[field] = _coerce(value)

        try:
            return cls(**data)
        except ValidationError as exc:  # pragma: no cover - config error path
            raise SystemExit(
                f"Invalid Evidence config:\n{exc}\n"
                "Check your YAML file / EVIDENCE_* env vars."
            ) from exc


# Top-level config keys that are filesystem paths (rebased when relative).
_PATH_KEYS = ("data_dir", "index_dir", "corpus_dir", "checkpoint_dir")


def _rebase_paths(data: dict, *, base: Path) -> dict:
    """Rebase relative path values in `data` against `base` (the config file's
    directory). Absolute paths and non-path values pass through unchanged.
    Nested `ledger.path` is handled too.
    """
    if not isinstance(data, dict):
        return data
    for key in _PATH_KEYS:
        val = data.get(key)
        if val is None:
            continue
        candidate = Path(str(val))
        if not candidate.is_absolute():
            data[key] = base / candidate
    ledger = data.get("ledger")
    if isinstance(ledger, dict):
        lp = ledger.get("path")
        if lp is not None:
            lpath = Path(str(lp))
            if not lpath.is_absolute():
                ledger["path"] = base / lpath
    return data


def _find_project_config() -> Path | None:
    """Walk up from CWD looking for a `.evidence/config.yaml`.

    Created by `evidence init` in the project root. Lets the user run
    `evidence ask ...` from any subdirectory without passing `-c`.
    Returns the first match, or None.
    """
    try:
        current = Path.cwd()
    except OSError:
        return None
    for d in (current, *current.parents):
        candidate = d / ".evidence" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _coerce(value: str) -> str | int | float | bool:
    """Best-effort coercion of an env string into a native type."""
    lowered = value.lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
