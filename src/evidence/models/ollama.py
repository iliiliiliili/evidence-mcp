"""Ollama-backed model wrappers with graceful fallbacks.

Design goals (PLAN.MD "Local model assignments" + "Key risks"):
  - A single `Models` facade so nodes never import a concrete provider.
  - Structured-output helpers that are robust to small local models:
    * prefer the provider's JSON mode when available,
    * fall back to tolerant JSON extraction from free text,
    * and finally a deterministic rule-based default so the pipeline never
      hard-fails because a 4B model returned prose.
  - Deterministic "fallback" implementations for embed/rerank so Phase 1 runs
    even when Ollama or sentence-transformers are not available (useful for CI
    and for developing nodes in isolation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Protocol

from ..config import EvidenceConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────────────────
class ChatLike(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        json: bool = False,
    ) -> str:
        """Return the raw text completion."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings (one vector per input)."""
        ...

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score per passage (higher = better)."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# JSON extraction (robust to noisy small-model output)
# ─────────────────────────────────────────────────────────────────────────
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from possibly-noisy model output."""
    if not text:
        raise ValueError("empty completion")
    text = text.strip()
    # 1) Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) Strip a code fence.
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3) Grab the outermost {...} or [...].
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from completion: {text[:200]!r}")


# ─────────────────────────────────────────────────────────────────────────
# Ollama provider
# ─────────────────────────────────────────────────────────────────────────
class OllamaModel:
    """Thin wrapper around the `ollama` python client (HTTP, lazy import).

    `host_name` pins a specific named host (for per-role routing). When it is
    None (the default) the active/default host is used, preserving prior
    behaviour for callers that do `OllamaModel(cfg)`.
    """

    def __init__(self, cfg: EvidenceConfig, host_name: str | None = None):
        self.cfg = cfg
        self.host_name = host_name
        self._client = None
        # Models we have already confirmed present (or tried to pull) this
        # process, so we don't re-list the server on every call.
        self._ensured: set[str] = set()
        # Cache of the last successful `list_models` names, so `pull` can
        # re-check after a pull without another round-trip where possible.
        self._known_models: list[str] | None = None

    def _host_url(self) -> str:
        return (
            self.cfg.ollama.host_url(self.host_name)
            if self.host_name
            else self.cfg.ollama.active_host_url()
        )

    def _client_inner(self):
        if self._client is None:
            try:
                import ollama  # local dependency
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "The 'ollama' package is not installed. "
                    "Run `pip install ollama` or set a fallback provider."
                ) from exc
            self._client = ollama.Client(
                host=self._host_url(),
                timeout=self.cfg.ollama.request_timeout,
            )
        return self._client

    def _list_model_names(self) -> list[str] | None:
        """Return the model names on the server, or None if it can't be listed.

        Never raises — a down server or a missing client yields None so callers
        can fall through to the normal request path (which will surface the
        usual "model not found" error if the model truly is absent).
        """
        client = self._client_inner()
        try:
            # Newer clients expose `list_models()`; older ones expose `list()`.
            # Both return a payload with a `models` list. The per-model name key
            # differs (`name` vs `model`) and the entries may be typed objects,
            # so extract defensively.
            payload = client.list_models() if hasattr(client, "list_models") else client.list()
            models = payload["models"] if isinstance(payload, dict) else payload.models
            names: list[str] = []
            for m in models:
                raw = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
                if raw is None:
                    raw = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
                if raw:
                    names.append(str(raw))
            self._known_models = names
            return names
        except Exception as exc:  # noqa: BLE001 - list must never block a call
            logger.debug("auto-pull: could not list Ollama models: %s", exc)
            return None

    @staticmethod
    def _model_matches(wanted: str, available: list[str]) -> bool:
        """True if `wanted` matches an available model name.

        Matches on exact name or on the name-without-digest (`name:tag` or
        `name`). Ollama's `list` returns fully-qualified names (e.g.
        `qwen2.5:9b`), so we compare both the full name and the base name.
        """
        wanted = wanted.strip()
        base = wanted.split(":", 1)[0]
        for name in available:
            if name == wanted:
                return True
            # `wanted` may omit the tag; match if the available base + tag line up.
            if base and (name == base or name.split(":", 1)[0] == base):
                # Only accept the base match when `wanted` had no explicit tag,
                # otherwise the user pinned a specific tag we shouldn't substitute.
                if ":" not in wanted:
                    return True
        return False

    def ensure_model(self, name: str) -> None:
        """Ensure `name` is present on the server, pulling it if configured.

        Best-effort by design: if listing or pulling fails, log and return so
        the caller's request proceeds and surfaces the real error. Controlled
        by `cfg.ollama.auto_pull`.
        """
        if not self.cfg.ollama.auto_pull:
            return
        name = (name or "").strip()
        if not name or name in self._ensured:
            return

        available = self._list_model_names()
        if available is not None:
            if self._model_matches(name, available):
                self._ensured.add(name)
                return
            # This is a notable, side-effecting action (a potentially large
            # download), so it must be visible even at the default WARNING
            # level — hence `logger.warning` for the start line.
            logger.warning("Pulling Ollama model %r (not present on server)…", name)
            try:
                client = self._client_inner()
                # Streaming pull so progress is visible if the caller logs it;
                # we just drain the generator to completion.
                for _chunk in client.pull(model=name, stream=True):
                    pass
                self._ensured.add(name)
                logger.info("auto-pull: pulled %r", name)
                return
            except Exception as exc:  # noqa: BLE001 - pull is best-effort
                logger.warning("auto-pull: pull of %r failed: %s", name, exc)
        # Listing failed (or pull failed): fall through and let the real call
        # raise the familiar "model not found" error.
        self._ensured.add(name)

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        json: bool = False,
    ) -> str:
        client = self._client_inner()
        name = model or self.cfg.ollama.light
        self.ensure_model(name)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        fmt: str | None = "json" if json else None
        # Pass `think` through as an option so thinking models (qwen3.x, gemma3,
        # deepseek-r1) skip the chain-of-thought pass. Models that don't think
        # simply ignore the option.
        options: dict[str, Any] = {"temperature": temperature}
        options["think"] = bool(self.cfg.ollama.think)
        response = client.chat(
            model=name,
            messages=messages,
            options=options,
            format=fmt,
            keep_alive=self.cfg.ollama.keep_alive,
        )
        return response["message"]["content"]

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        client = self._client_inner()
        name = model or self.cfg.ollama.embed
        self.ensure_model(name)
        out: list[list[float]] = []
        # Batch in chunks to avoid huge single requests.
        batch = 32
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            resp = client.embed(model=name, input=chunk, keep_alive=self.cfg.ollama.keep_alive)
            out.extend(resp["embeddings"])
        return out

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """LLM-based rerank fallback (no cross-encoder installed).

        We ask the model to rank passages 1..N and invert the order to get a
        score. This is intentionally simple — Phase 1 quality expects the
        bge-reranker cross-encoder (see `Reranker`) when available.
        """
        if not passages:
            return []
        listing = "\n".join(f"{i+1}. {p[:400]}" for i, p in enumerate(passages))
        prompt = (
            f"Rank the following passages by relevance to the query. "
            f"Respond ONLY with a JSON array of their numbers in best-first order.\n\n"
            f"Query: {query}\n\nPassages:\n{listing}\n\n"
            f"Answer format: [3,1,2]"
        )
        try:
            raw = self.complete(prompt, model=self.cfg.ollama.light, json=True)
            order = extract_json(raw)
            if not isinstance(order, list):
                raise ValueError("expected list")
            # Map rank position -> score (higher earlier).
            scores = [0.0] * len(passages)
            for pos, idx in enumerate(order):
                try:
                    j = int(idx) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= j < len(passages):
                    scores[j] = float(len(passages) - pos)
            return scores
        except Exception as exc:  # noqa: BLE001 - fallback must never raise
            logger.warning("LLM rerank failed (%s); using neutral scores", exc)
            return [0.5] * len(passages)


# ─────────────────────────────────────────────────────────────────────────
# OpenAI-compatible provider (llama-server, vLLM, LM Studio, OpenAI, …)
# ─────────────────────────────────────────────────────────────────────────
class OpenAIModel:
    """Client for any OpenAI-compatible HTTP server.

    Covers llama-server, vLLM, LM Studio, a hosted OpenAI endpoint, or any
    service exposing `/v1/chat/completions` and `/v1/embeddings`. Auth is
    optional (many local llama-server instances run with no key).

    Uses `httpx` (already a core dependency). Like `OllamaModel`, `complete`
    returns raw text and `embed` returns one vector per input; `rerank` reuses
    the same LLM-ranking trick as the Ollama path. `ensure_model` is a no-op:
    OpenAI-style servers do not pull models on demand.
    """

    def __init__(self, cfg: EvidenceConfig, host_name: str):
        self.cfg = cfg
        self.host_name = host_name
        # Base URL: strip a trailing `/v1` if the user included it, then we add
        # it back in the endpoint paths below.
        self.base_url = cfg.ollama.host_url(host_name).rstrip("/")
        self.api_key = cfg.ollama.api_key_for(host_name)
        self.timeout = cfg.ollama.request_timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx  # core dependency

        url = f"{self.base_url}/v1/{path}"
        resp = httpx.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def ensure_model(self, name: str) -> None:
        """No-op: OpenAI-compatible servers manage their own model lifecycle."""
        return None

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        json: bool = False,
    ) -> str:
        name = model or self.cfg.ollama.light
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": name,
            "messages": messages,
            "temperature": temperature,
        }
        if json:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        name = model or self.cfg.ollama.embed
        # Batch to avoid one huge request (mirrors the Ollama path).
        batch = 32
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            data = self._post("embeddings", {"model": name, "input": chunk})
            # Preserve input order via the `index` field (OpenAI returns sorted).
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            out.extend(r["embedding"] for r in rows)
        return out

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """LLM-based rerank (same heuristic as the Ollama fallback)."""
        if not passages:
            return []
        listing = "\n".join(f"{i+1}. {p[:400]}" for i, p in enumerate(passages))
        prompt = (
            "Rank the following passages by relevance to the query. "
            "Respond ONLY with a JSON array of their numbers in best-first order.\n\n"
            f"Query: {query}\n\nPassages:\n{listing}\n\n"
            "Answer format: [3,1,2]"
        )
        try:
            raw = self.complete(prompt, model=self.cfg.ollama.light, json=True)
            order = extract_json(raw)
            if not isinstance(order, list):
                raise ValueError("expected list")
            scores = [0.0] * len(passages)
            for pos, idx in enumerate(order):
                try:
                    j = int(idx) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= j < len(passages):
                    scores[j] = float(len(passages) - pos)
            return scores
        except Exception as exc:  # noqa: BLE001 - fallback must never raise
            logger.warning("OpenAI rerank failed (%s); using neutral scores", exc)
            return [0.5] * len(passages)


# ─────────────────────────────────────────────────────────────────────────
# Deterministic fallbacks (no model required)
# ─────────────────────────────────────────────────────────────────────────
class DeterministicEmbedder:
    """Hash-based embedding so the pipeline runs without Ollama.

    NOT semantically meaningful — it only lets you smoke-test the graph,
    persistence, and node wiring. Do not rely on it for retrieval quality.
    """

    dim = 256

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand the digest to `dim` floats deterministically.
            vec: list[float] = []
            i = 0
            while len(vec) < self.dim:
                byte = h[i % len(h)]
                vec.append((byte / 255.0) * 2 - 1)
                i += 1
            # L2-normalize.
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class DeterministicReranker:
    """TF-overlap heuristic so the rerank node runs without sentence-transformers."""

    def __init__(self):
        try:
            import sentence_transformers  # noqa: F401

            self._has_st = True
        except ImportError:
            self._has_st = False

    def _tokenize(self, s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return [0.0] * len(passages)
        scores = []
        for p in passages:
            p_tokens = set(self._tokenize(p))
            overlap = len(q_tokens & p_tokens) / len(q_tokens)
            scores.append(float(overlap))
        return scores


# ─────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker (optional, best quality)
# ─────────────────────────────────────────────────────────────────────────
class CrossEncoderReranker:
    """bge-reranker-v2-m3 via sentence-transformers (Phase 1 recommended).

    Lazily loads the model; falls back to `DeterministicReranker` on failure
    so a missing heavy dep never blocks the pipeline.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._fallback = DeterministicReranker()

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s …", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cross-encoder unavailable (%s); falling back to heuristic rerank.",
                exc,
            )
            return None

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        if model is None or not passages:
            return self._fallback.rerank(query, passages)
        pairs = [[query, p] for p in passages]
        scores = model.predict(pairs, convert_to_numpy=True)
        return [float(s) for s in scores]


# ─────────────────────────────────────────────────────────────────────────
# Facade
# ─────────────────────────────────────────────────────────────────────────
class EmbedBackend:
    """Uniform `.embed(texts)` over either Ollama or the deterministic hash."""

    def __init__(self, impl):
        self.impl = impl

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.impl.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.impl.embed([text])[0]


class Models:
    """Provider-agnostic facade used by all nodes.

    Routes each call to the right host + provider based on the model's ROLE:
      * `reasoning` / `light` / `embed` map to models in `cfg.ollama.*`;
      * each role resolves to a named host via `host_by_role` (else the default);
      * each host resolves to a provider (`ollama` or `openai`) via `host_types`.

    This lets e.g. embeddings live on one box and chat on another, or use
    llama-server/vLLM (openai) for one role and Ollama for another — while a
    single `active_host` keeps everything on one box by default.
    """

    def __init__(self, cfg: EvidenceConfig):
        self.cfg = cfg
        self._ollama: OllamaModel | None = None
        self._embed: EmbedBackend | None = None
        self._rerank: CrossEncoderReranker | None = None
        self._providers: dict[tuple[str, str | None], Any] = {}

    # -- routing -----------------------------------------------------------
    def _provider(self, role: str | None = None, model: str | None = None) -> Any:
        """Resolve the provider instance for a role (or a model name).

        A provider is an `OllamaModel` or `OpenAIModel` bound to the host that
        serves this role. Instances are cached per (type, host).
        """
        if role is None:
            role = self.cfg.ollama.role_for_model(model)
        host = self.cfg.ollama.host_for_role(role)
        ptype = self.cfg.ollama.host_type(host)
        key = (ptype, host)
        provider = self._providers.get(key)
        if provider is None:
            if ptype == "openai":
                provider = OpenAIModel(self.cfg, host)  # host is non-None here
            else:
                provider = OllamaModel(self.cfg, host)
            self._providers[key] = provider
        return provider

    # -- lazy accessors (kept for backward compatibility) ------------------
    @property
    def ollama(self) -> OllamaModel:
        if self._ollama is None:
            self._ollama = OllamaModel(self.cfg)
        return self._ollama

    @property
    def embedder(self) -> EmbedBackend:
        """Uniform embed backend for the `embed` role's host (hash fallback)."""
        if self._embed is None:
            host = self.cfg.ollama.host_for_role("embed")
            ptype = self.cfg.ollama.host_type(host)
            if ptype == "openai":
                self._embed = EmbedBackend(OpenAIModel(self.cfg, host))
            else:
                try:
                    import ollama  # noqa: F401

                    self._embed = EmbedBackend(OllamaModel(self.cfg, host))
                except ImportError:
                    logger.warning(
                        "Ollama not available; using deterministic hash embeddings. "
                        "Retrieval quality will be poor — install Ollama + pull "
                        f"{self.cfg.ollama.embed} for real embeddings."
                    )
                    self._embed = EmbedBackend(DeterministicEmbedder())
        return self._embed

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._rerank is None:
            self._rerank = CrossEncoderReranker(self.cfg.ollama.reranker_local)
        return self._rerank

    # -- high-level ops with fallbacks -------------------------------------
    # Design rule (PLAN.MD "Key risks"): the pipeline NEVER hard-fails because a
    # small model is unavailable or returned prose. These methods always return
    # a value; callers branch on the fallback (usually None / "") and fall back
    # to deterministic logic.
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        json: bool = False,
        fallback: str | None = None,
    ) -> str:
        """Complete, returning `fallback` (default "") on any provider failure."""
        try:
            provider = self._provider(model=model)
            return provider.complete(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                json=json,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM complete failed: %s", exc)
            return fallback or ""

    def complete_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        fallback: Any = None,
    ) -> Any:
        """Complete + parse JSON. Returns `fallback` (default None) on any failure."""
        try:
            provider = self._provider(model=model)
            raw = provider.complete(
                prompt, model=model, system=system, temperature=temperature, json=True
            )
            return extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JSON completion failed (%s); using fallback", exc)
            return fallback

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        # Try the cross-encoder first; on exception use the LLM rerank fallback.
        try:
            return self.reranker.rerank(query, passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder rerank failed (%s); using LLM fallback", exc)
            try:
                return self._provider(role="light").rerank(query, passages)
            except Exception:  # noqa: BLE001
                return [0.5] * len(passages)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(y * y for y in b) ** 0.5 or 1.0
        return dot / (na * nb)
