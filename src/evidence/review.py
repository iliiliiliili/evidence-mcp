"""Scientific review of a corpus paper — Typer-free core (see ``populate.py``).

``review_paper(paper_id)`` produces a structured peer-review of ONE paper in the
local corpus, following a fixed five-part structure:

  1. **explain the paper**        → ``summary``
  2. **strengths & weaknesses**   → ``strengths`` / ``weaknesses``
  3. **incorrect statements?**    → ``incorrect_statements`` (chunk-grounded)
  4. **undefined terms?**         → ``undefined_terms`` (model + acronym heuristic)
  5. **references**               → ``references`` (each optionally resolved via
                                     the external literature APIs, with a
                                     ``relevance`` and ``role`` per reference)

Design follows the rest of Evidence:

* A local reasoning model drafts the judgement as **strict JSON**
  (``Models.complete_json``). Every model call has a **deterministic fallback**
  so the tool never hard-fails when Ollama is absent — the review degrades to
  the abstract summary plus a model-free acronym/undefined-term scan.
* Reference resolution reuses the existing external providers
  (``search_literature`` → OpenAlex/S2/Crossref) and is **fail-soft**: a network
  error or a disabled provider is recorded on the reference, never raised.
* The module is dependency-light (no Typer, no LangGraph) so it unit-tests
  against a temp ``Store`` and a fake model, exactly like the suite.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import EvidenceConfig
from .indexes.store import Store
from .models.ollama import Models

logger = logging.getLogger(__name__)


# ── data shapes ──────────────────────────────────────────────────────────────
@dataclass
class ReviewReport:
    """The outcome of reviewing one paper."""

    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    n_chunks: int = 0

    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    incorrect_statements: list[dict] = field(default_factory=list)
    undefined_terms: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    model: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "n_chunks": self.n_chunks,
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "incorrect_statements": self.incorrect_statements,
            "undefined_terms": self.undefined_terms,
            "references": self.references,
            "model": self.model,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        title = self.title or self.paper_id
        lines.append(f"# Scientific Review — {title}")
        meta_bits: list[str] = []
        if self.authors:
            a = self.authors
            meta_bits.append(a[0] + (" et al." if len(a) > 1 else ""))
        if self.year:
            meta_bits.append(str(self.year))
        if self.doi:
            meta_bits.append(f"DOI {self.doi}")
        if meta_bits:
            lines.append("")
            lines.append("*" + " · ".join(meta_bits) + "*")

        # 1. summary
        lines += ["", "## 1. Summary", "", self.summary or "_(no summary)_"]

        # 2. strengths
        lines += ["", "## 2. Strengths", ""]
        lines += [f"- {s}" for s in self.strengths] or ["- _(none recorded)_"]

        # 3. weaknesses
        lines += ["", "## 3. Weaknesses", ""]
        lines += [f"- {w}" for w in self.weaknesses] or ["- _(none recorded)_"]

        # 4. incorrect statements
        lines += ["", "## 4. Possibly incorrect statements", ""]
        if self.incorrect_statements:
            lines += [
                "| Statement | Problem | Location | Confidence |",
                "|---|---|---|---|",
            ]
            for s in self.incorrect_statements:
                stmt = _md(s.get("statement", ""))
                prob = _md(s.get("problem", ""))
                loc = _md(s.get("chunk_id", "") or s.get("context", "") or "—")
                conf = s.get("confidence", "")
                lines.append(f"| {stmt} | {prob} | {loc} | {conf} |")
            lines += [
                "",
                "_(These are model-flagged candidates, not verified errors — "
                "check each against the paper.)_",
            ]
        else:
            lines += ["_None flagged._"]

        # 5. undefined terms
        lines += ["", "## 5. Undefined terms", ""]
        if self.undefined_terms:
            lines += [
                "| Term | First used | Suggested definition |",
                "|---|---|---|",
            ]
            for t in self.undefined_terms:
                lines.append(
                    f"| {_md(t.get('term', ''))} | {_md(t.get('context', '') or '—')} "
                    f"| {_md(t.get('suggestion', '') or '—')} |"
                )
        else:
            lines += ["_None identified._"]

        # 6. references
        lines += ["", "## 6. References", ""]
        if self.references:
            lines += [
                "| # | Reference | Year | Relevance | Role | Notes | Resolved |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in self.references:
                ref = r.get("title") or r.get("authors") or "—"
                if r.get("authors") and r["authors"] != r.get("title"):
                    ref += f" — {r['authors']}"
                if r.get("doi"):
                    ref += f" · {r['doi']}"
                year = r.get("year") or "—"
                res = _md(_resolve_label(r.get("resolved")))
                lines.append(
                    f"| {r.get('index', '')} | {_md(ref)} | {year} "
                    f"| {r.get('relevance', '')} | {r.get('role', '')} "
                    f"| {_md(r.get('notes', '') or '—')} | {res} |"
                )
        else:
            lines += ["_No references could be identified._"]

        if self.notes:
            lines += ["", "---", ""]
            lines += [f"_{n}_" for n in self.notes]

        return "\n".join(lines) + "\n"


# ── helpers ─────────────────────────────────────────────────────────────────
def _md(s: str) -> str:
    """Escape a cell for a Markdown table (pipes + newlines)."""
    return str(s or "").replace("|", "\\|").replace("\n", " ").strip()


def _resolve_label(resolved: dict | None) -> str:
    """Human label for a reference's external-resolution outcome."""
    if not resolved:
        return "—"
    if resolved.get("found") is None:
        return "not checked"
    if resolved.get("found") is True:
        bits = []
        if resolved.get("match_year"):
            bits.append(str(resolved["match_year"]))
        if resolved.get("citations") is not None:
            bits.append(f"{resolved['citations']} citations")
        if resolved.get("source"):
            bits.append(resolved["source"])
        return "found" + (f" ({', '.join(bits)})" if bits else "")
    return "not found"


def resolve_paper(store: Store, query: str) -> tuple[str, object] | None:
    """Resolve a user-supplied identifier to a paper in the store.

    Accepts an exact ``paper_id`` first, then a case-insensitive title
    substring (picking the most-chunked match on ties). Returns
    ``(paper_id, PaperRecord)`` or ``None``.
    """
    p = store.get_paper(query)
    if p is not None:
        return p.paper_id, p
    q = (query or "").strip().lower()
    if not q:
        return None
    matches = [
        p for p in store.list_papers() if q in (p.title or "").lower()
    ]
    if not matches:
        # prefix match on the paper_id
        matches = [
            p for p in store.list_papers()
            if (p.paper_id or "").lower().startswith(q)
        ]
    if not matches:
        return None
    matches.sort(
        key=lambda p: store.count_chunks_for_paper(p.paper_id), reverse=True
    )
    return matches[0].paper_id, matches[0]


# ── paper text assembly ──────────────────────────────────────────────────────
_SECTION_LABELS = (
    "reference", "bibliography", "cited work", "works cited", "appendix",
    "acknowledg",
)


def _paper_chunks(store: Store, paper_id: str):
    return store.chunks_for_paper(paper_id)


def _section_outline(chunks) -> list[str]:
    """Deduped, in-order list of section headings (for the model prompt)."""
    out: list[str] = []
    for c in chunks:
        h = (c.heading or "").strip()
        if h and h not in out:
            out.append(h)
    return out


def _paper_text(chunks, *, max_chars: int = 48_000, per_chunk: int = 1500) -> str:
    """Join a paper's chunks into one prompt-ready text (bounded)."""
    parts: list[str] = []
    for c in chunks:
        header = f"[{c.heading or 'section'}" + (f" p.{c.page}" if c.page else "") + "]"
        body = (c.text or "").strip()
        if per_chunk > 0 and len(body) > per_chunk:
            body = body[:per_chunk].rstrip() + " […]"
        parts.append(f"{header}\n{body}")
    text = "\n\n".join(parts)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[...truncated...]"
    return text


# Quick test: does this chunk contain at least one line starting with a
# numbered reference marker (e.g. "[1]", "[12]")?
_REF_MARKER_LINE = re.compile(r"^\s*\[\s*\d+", re.MULTILINE)

# Whole-line boilerplate from PDF extraction (watermarks, confidential banners,
# journal page headers) that is NOT part of any reference. Filtered so the
# assembled reference list is clean.
_REF_BOILERPLATE = [
    re.compile(r"^\s*(for review only)\s*\.?\s*$", re.I),
    re.compile(r"contents must remain confidential", re.I),
    re.compile(r"paper\s*#\s*\d+", re.I),           # "IEEE ICASSP 2027 Paper #11299"
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.I),    # "Page 5 of 5"
]


def _strip_ref_boilerplate(text: str) -> str:
    """Drop whole lines that are pure extraction boilerplate; keep everything
    else (including multi-line references) intact."""
    kept = []
    for line in (text or "").splitlines():
        if any(p.search(line) for p in _REF_BOILERPLATE):
            continue
        kept.append(line)
    return "\n".join(kept)


def references_text(chunks) -> str:
    """The paper's reference list, from its References/Bibliography chunks.

    Strategy (first match wins):
      1. Chunks whose **heading** matches a reference-section label
         ("References", "Bibliography", …).
      2. Chunks whose **text** contains numbered reference markers
         (``[1]``, ``[2]``, …) — catches the common case where PDF extraction
         yields empty headings and the reference block spans multiple chunks.
      3. The **final** chunk (most papers put references last).

    Returns the RAW text (untruncated) so callers can split it into entries.
    All matching chunks are concatenated in ordinal order, so a reference list
    spread across multiple chunks is assembled in full.
    """
    def _finish(text: str) -> str:
        # Drop watermark / confidential-banner lines, then trim any preamble
        # (section text like "COMPLIANCE WITH ETHICAL STANDARDS") that appears
        # before the first reference marker.
        text = _strip_ref_boilerplate(text).strip()
        m = _REF_MARKER_LINE.search(text)
        if m:
            text = text[m.start():]
        return text.strip()

    # 1) Heading-based match.
    ref_chunks = [
        c for c in chunks
        if c.heading and any(k in c.heading.lower() for k in _SECTION_LABELS)
    ]
    if ref_chunks:
        return _finish("\n".join((c.text or "") for c in ref_chunks))

    # 2) Marker-based: any chunk that contains at least one "[N]" line.
    #    This catches the real-world case where headings are empty and the
    #    reference block is split across multiple chunks (e.g. chunk 9 has
    #    [1]–[11], chunk 10 has [11]–[19]).
    ref_chunks = [
        c for c in chunks if _REF_MARKER_LINE.search(c.text or "")
    ]
    if ref_chunks:
        return _finish("\n".join((c.text or "") for c in ref_chunks))

    # 3) Fallback: the final chunk.
    if chunks:
        return _finish(chunks[-1].text or "")

    return ""


# Matches a leading reference marker: [1], [12], 1., (1), [a], [A1], etc.
_REF_MARKER_RE = re.compile(
    r"^\s*(?:\[\s*(\d+[a-z]?)\s*\]|(\d+)[\.\)]\s|\(\s*\d+\s*\))"
)


def split_references(text: str) -> list[str]:
    """Split raw reference-list text into individual entry strings.

    Heuristic but robust to common layouts:
      - numbered:  ``[1] Title ...``, ``1. Title ...``, ``(1) Title ...``
      - a new entry starts at the next marker; everything before it (including
        wrapped lines) belongs to the current entry.

    Returns a list of cleaned entry strings (marker kept, so the index is
    recoverable). Falls back to ``[text]`` when no marker is found.
    """
    text = (text or "").strip()
    if not text:
        return []
    # Normalize common line-break-before-marker artifacts: PDF extraction often
    # splits an entry's last line from the next marker with a stray newline.
    entries: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                current.append("")  # blank line = soft break within an entry
            continue
        m = _REF_MARKER_RE.match(line)
        if m:
            if current:
                entries.append(_clean_entry(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(_clean_entry(current))
    return [e for e in entries if e]


def _clean_entry(parts: list[str]) -> str:
    """Collapse a raw entry's lines into one readable string."""
    joined = " ".join(p for p in parts if p).strip()
    # Collapse runs of whitespace.
    return re.sub(r"\s+", " ", joined).strip()


# ── deterministic undefined-term (acronym) scan ──────────────────────────────
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
# Acronyms a reader is expected to know already (suppress false positives).
_COMMON_ACRONYMS = {
    "API", "GPU", "CPU", "TPU", "RAM", "SSD", "HDD", "HTTP", "HTTPS", "JSON",
    "XML", "CSV", "HTML", "CSS", "PDF", "URL", "URI", "SQL", "SDK", "CLI",
    "IDE", "UTF", "RGB", "ISO", "IEEE", "ACM", "USA", "US", "UK", "EU", "UN",
    "NASA", "NIST", "OS", "IO", "ID", "AND", "NOT", "OR", "FOR", "ALL",
    "THE", "NEW", "NOW", "FIG", "TAB", "EQ", "ALG", "OPT", "MAX", "MIN",
    "STD", "ERR", "NOTE", "REF", "VER", "NUM", "LOG", "INFO",
    # ubiquitous ML/CS terms
    "LLM", "NLP", "ML", "DL", "AI", "ANN", "RNN", "CNN", "LSTM", "GRU",
    "GAN", "VAE", "BERT", "GPT", "BPE", "FFT", "PCA", "SVM", "KNN",
    "MSE", "RMSE", "MAP", "MCMC",
}


def _acronym_defined(acr: str, text: str) -> bool:
    """Heuristic: is `acr` ever expanded/defined anywhere in `text`?"""
    esc = re.escape(acr)
    # "Full Name (ACR)" — acronym alone in parentheses (the most common form).
    if re.search(rf"\(\s*{esc}\s*\)", text):
        return True
    # "Full Name (ACR)" — acronym inside parentheses next to other words.
    m = re.search(rf"\(([^()]*\b{esc}\b[^()]*)\)", text)
    if m:
        inner = m.group(1).replace(acr, "").strip(" :—-,.")
        if inner:
            return True
    # "ACR (full name)"
    if re.search(rf"\b{esc}\b\s*\([A-Za-z]", text):
        return True
    # "ACR stands for / denotes / is short for / defined as / refers to ..."
    if re.search(
        rf"\b{esc}\b\s+(?:stands for|denotes|is defined as|defined as|is short for|"
        rf"short for|refers to|referring to|acronym for)",
        text,
        re.I,
    ):
        return True
    return False


def find_undefined_acronyms(text: str, *, limit: int = 15) -> list[dict]:
    """Model-free scan: acronyms used but never expanded in the text.

    Returns ``[{"term", "context", "suggestion", "source"}]``. A genuinely
    useful, deterministic signal (works with no local model at all). The list
    is capped and de-duplicated; order is first-appearance.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for m in _ACRONYM_RE.finditer(text or ""):
        acr = m.group(0)
        key = acr.lower()
        if key in seen:
            continue
        if acr in _COMMON_ACRONYMS:
            continue
        # Skip if it looks like a section label / header token we injected.
        if key in {"chunk", "section", "page", "truncated"}:
            continue
        if _acronym_defined(acr, text):
            continue
        seen.add(key)
        # Capture a short context window around the first use.
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        ctx = " ".join(text[start:end].split())
        out.append(
            {
                "term": acr,
                "context": ctx,
                "suggestion": f"Define on first use, e.g. \"{acr} (full name).\"",
                "source": "acronym-scan",
            }
        )
        if len(out) >= limit:
            break
    return out


# ── service ──────────────────────────────────────────────────────────────────
class ReviewService:
    """Drafts a structured review of one corpus paper.

    ``store`` is passed in by the caller (CLI / MCP / tests) and is NOT closed
    here — ownership stays with the caller.
    """

    def __init__(
        self,
        cfg: EvidenceConfig,
        *,
        models: Models | None = None,
        store: Store | None = None,
    ):
        self.cfg = cfg
        self.models = models or Models(cfg)
        self.store = store if store is not None else Store(cfg.resolved_store_path)

    # ── model calls (fail-soft) ─────────────────────────────────────────
    def _analyze(self, paper, text: str, outline: list[str]) -> dict | None:
        """Phase A — summary + strengths/weaknesses + incorrect + undefined.

        Returns the parsed JSON dict, or None when the model is unavailable.
        """
        model = self.cfg.ollama.reasoning
        system = (
            "You are a meticulous peer reviewer of a research paper. You are "
            "skeptical, precise, and evidence-driven. Respond with STRICT JSON "
            "only — no prose, no markdown fences, no trailing commas."
        )
        authors = ", ".join(paper.authors) if paper.authors else "(unknown)"
        user = (
            "Review the following paper. Ground every judgement in the provided "
            "text; where a point is tied to a specific passage, include that "
            "passage's chunk_id (it appears in the [section p.N] header).\n\n"
            f"Paper title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Year: {paper.year}\n"
            f"Section outline: {', '.join(outline) or '(none)'}\n\n"
            "Full text (chunked):\n"
            f"{text}\n\n"
            "Return STRICT JSON with exactly these keys:\n"
            "{\n"
            '  "summary": "3-5 sentences: the problem addressed, the main '
            'contribution/claim, the method, and the headline result.",\n'
            '  "strengths": ["..."],\n'
            '  "weaknesses": ["..."],\n'
            '  "incorrect_statements": [\n'
            '    {"statement": "short quote of the questionable sentence",'
            ' "problem": "why it is likely wrong or unsupported",'
            ' "chunk_id": "the chunk where it appears",'
            ' "confidence": "high | medium | low"}\n'
            "  ],\n"
            '  "undefined_terms": [\n'
            '    {"term": "the term/acronym",\n'
            '     "context": "where it first appears (section or short quote)",\n'
            '     "suggestion": "how the paper should define it"}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Be conservative with `incorrect_statements`: flag only genuine "
            "factual errors, internal contradictions, or misuse of a named "
            "method/theorem. If unsure, use confidence \"low\". An empty list "
            "is a valid, correct answer.\n"
            "- `undefined_terms`: terms/acronyms used without any definition "
            "or expansion.\n"
            "- Never invent content that is not in the text.\n"
        )
        raw = self.models.complete_json(
            user, model=model, system=system, temperature=0.1, fallback=None
        )
        return raw if isinstance(raw, dict) else None

    def _references(self, paper, ref_text: str) -> list[dict]:
        """Phase B — structure the reference list with relevance + role.

        Returns a (possibly empty) normalized list. Empty when the model is
        unavailable or there is no reference text.
        """
        if not (ref_text or "").strip():
            return []
        model = self.cfg.ollama.reasoning
        system = (
            "You are cataloguing the reference list of a research paper so a "
            "reviewer can judge its relevance. Respond with STRICT JSON only — "
            "no prose, no markdown fences, no trailing commas."
        )
        user = (
            f"Paper under review: {paper.title}\n"
            f"Abstract: {(paper.abstract or '')[:600]}\n\n"
            "References (as they appear in the paper):\n"
            f"{ref_text}\n\n"
            "Return STRICT JSON with exactly this shape:\n"
            "{\n"
            '  "references": [\n'
            "    {\n"
            '      "index": 1,\n'
            '      "title": "paper title",\n'
            '      "authors": "First Author et al.",\n'
            '      "year": 2019,\n'
            '      "doi": "10.xxxx/... (ONLY if visible in the reference text; '
            'omit otherwise)",\n'
            '      "relevance": "high | medium | low",\n'
            '      "role": "background | method | comparison | dataset | theory '
            '| application | other",\n'
            '      "notes": "one short sentence: what this reference is used '
            'for in the paper"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- List every distinct reference you can identify in the text.\n"
            "- relevance: 'high' = core prior work the paper builds on or "
            "compares against; 'medium' = useful context; 'low' = tangential "
            "/ general background.\n"
            "- role: pick the single best-fit label.\n"
            "- Never invent a DOI you do not see in the text.\n"
        )
        raw = self.models.complete_json(
            user, model=model, system=system, temperature=0.1, fallback=None
        )
        if not isinstance(raw, dict):
            return []
        refs = raw.get("references")
        if not isinstance(refs, list):
            return []
        out: list[dict] = []
        for i, r in enumerate(refs, start=1):
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or "").strip()
            authors = str(r.get("authors") or "").strip()
            if not title and not authors:
                continue
            year = r.get("year")
            try:
                year = int(year) if year is not None else None
            except (TypeError, ValueError):
                year = None
            rel = str(r.get("relevance") or "").strip().lower()
            if rel not in ("high", "medium", "low"):
                rel = "medium"
            role = str(r.get("role") or "").strip().lower() or "other"
            out.append(
                {
                    "index": r.get("index") or i,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "doi": str(r.get("doi") or "").strip() or None,
                    "relevance": rel,
                    "role": role,
                    "notes": str(r.get("notes") or "").strip(),
                }
            )
        return out

    def _resolve_reference(self, ref: dict, *, enabled: bool) -> dict:
        """Best-effort external lookup for ONE reference (fail-soft)."""
        if not enabled:
            return {"found": None, "note": "external lookup disabled"}
        query = (ref.get("title") or "").strip()
        if ref.get("authors"):
            query = f"{query} {ref['authors']}".strip()
        if not query:
            return {"found": None, "note": "no title to search"}
        try:
            from .mcp.tools import search_literature

            works = search_literature(query, limit=1, cfg=self.cfg)
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            logger.warning("reference lookup failed (%s): %s", query, exc)
            return {"found": None, "note": f"lookup failed: {exc}"}
        if not works:
            return {"found": False, "note": "not found in external sources"}
        w = works[0]
        return {
            "found": True,
            "match_title": w.get("paper") or w.get("title"),
            "match_year": w.get("year"),
            "match_doi": w.get("doi"),
            "citations": w.get("cited_by"),
            "source": w.get("source"),
        }

    # ── orchestration ────────────────────────────────────────────────────
    def review(
        self,
        paper_id: str,
        *,
        external: bool | None = None,
        resolve_limit: int = 25,
    ) -> ReviewReport:
        """Run the full five-part review of one paper. Never raises for model
        or network failures — those are recorded in ``notes``."""
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"paper not found: {paper_id}")

        chunks = _paper_chunks(self.store, paper_id)
        text = _paper_text(chunks)
        outline = _section_outline(chunks)
        effective_external = (
            self.cfg.external.enabled if external is None else external
        )

        report = ReviewReport(
            paper_id=paper_id,
            title=paper.title or paper.path,
            authors=list(paper.authors or []),
            year=paper.year,
            doi=paper.doi,
            n_chunks=len(chunks),
            model=self.cfg.ollama.reasoning,
        )

        # ── Phase A: summary + strengths/weaknesses + incorrect + undefined ─
        analysis = self._analyze(paper, text, outline) if text else None
        if analysis is None:
            report.notes.append(
                "local reasoning model unavailable — using the abstract as the "
                "summary and a deterministic acronym scan for undefined terms"
            )
            report.summary = (
                (paper.abstract or "").strip()
                or f"{paper.title or paper.path} — {len(chunks)} indexed "
                   "section(s). (No abstract or model summary available.)"
            )
            report.strengths = []
            report.weaknesses = []
            report.incorrect_statements = []
            # No model → still surface the model-free undefined-term scan.
            report.undefined_terms = find_undefined_acronyms(text)
        else:
            report.summary = str(analysis.get("summary") or "").strip()
            report.strengths = _str_list(analysis.get("strengths"))
            report.weaknesses = _str_list(analysis.get("weaknesses"))
            report.incorrect_statements = _dict_list(
                analysis.get("incorrect_statements")
            )
            model_terms = _dict_list(analysis.get("undefined_terms"))
            # Merge the model's terms with the deterministic acronym scan,
            # de-duplicated by term.
            report.undefined_terms = _merge_terms(model_terms, text)

        # ── Phase B: references (structure + optional external resolution) ──
        ref_text = references_text(chunks)
        refs = self._references(paper, ref_text) if ref_text else []
        if not refs:
            if ref_text:
                report.notes.append(
                    "reference list could not be structured (model unavailable)"
                )
            else:
                report.notes.append(
                    "no reference list found in the paper's chunks"
                )

        # Resolve up to `resolve_limit` references externally (high relevance
        # first), so the review confirms whether the cited works exist.
        if refs and effective_external:
            order = sorted(
                range(len(refs)),
                key=lambda i: {"high": 0, "medium": 1, "low": 2}.get(
                    refs[i].get("relevance"), 3
                ),
            )
            resolved_idx = order[:max(0, resolve_limit)]
            for i in resolved_idx:
                refs[i]["resolved"] = self._resolve_reference(
                    refs[i], enabled=True
                )
            for i in range(len(refs)):
                if "resolved" not in refs[i]:
                    refs[i]["resolved"] = {
                        "found": None,
                        "note": f"not checked ({len(resolved_idx)} cap reached)",
                    }
        elif refs:
            for r in refs:
                r["resolved"] = {
                    "found": None,
                    "note": "external lookup disabled",
                }

        report.references = refs
        return report


# ── normalizers ──────────────────────────────────────────────────────────────
def _str_list(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


def _dict_list(v) -> list[dict]:
    if not isinstance(v, list):
        return []
    return [d for d in v if isinstance(d, dict)]


def _merge_terms(model_terms: list[dict], text: str) -> list[dict]:
    """Prefer model-identified terms, then add model-free acronym candidates
    the model did not already cover (dedup by lowercased term)."""
    out = list(model_terms)
    covered = {str(t.get("term", "")).strip().lower() for t in out}
    for t in find_undefined_acronyms(text):
        if t["term"].lower() not in covered:
            out.append(t)
            covered.add(t["term"].lower())
    return out


# ── public entry point ───────────────────────────────────────────────────────
def review_paper(
    paper_id: str,
    *,
    external: bool | None = None,
    resolve_limit: int = 25,
    cfg: EvidenceConfig | None = None,
) -> dict:
    """Review one corpus paper and return a JSON-serializable dict.

    ``paper_id`` may be an exact store id or a title substring (the most-chunked
    match wins). ``external`` overrides ``cfg.external.enabled`` for reference
    resolution (None = follow config). See ``ReviewReport`` for the shape.
    """
    cfg = cfg or EvidenceConfig.load()
    if not cfg.resolved_store_path.exists():
        return {
            "error": (
                f"No corpus index found at {cfg.resolved_index_dir}. "
                "Run `evidence ingest` first."
            )
        }
    store = Store(cfg.resolved_store_path)
    svc = ReviewService(cfg, store=store)
    try:
        resolved = resolve_paper(store, paper_id)
        if resolved is None:
            return {
                "error": f"paper not found: {paper_id!r}",
                "hint": "Run `evidence papers` to list indexed papers.",
            }
        pid, _paper = resolved
        report = svc.review(
            pid, external=external, resolve_limit=resolve_limit
        )
        return report.as_dict()
    finally:
        store.close()


def review_paper_markdown(
    paper_id: str,
    *,
    external: bool | None = None,
    resolve_limit: int = 25,
    cfg: EvidenceConfig | None = None,
) -> str:
    """Convenience: the review rendered as Markdown (see ``ReviewReport``)."""
    cfg = cfg or EvidenceConfig.load()
    if not cfg.resolved_store_path.exists():
        return "_(no corpus index — run `evidence ingest` first)_"
    store = Store(cfg.resolved_store_path)
    svc = ReviewService(cfg, store=store)
    try:
        resolved = resolve_paper(store, paper_id)
        if resolved is None:
            return f"_(paper not found: {paper_id!r})_"
        pid, _paper = resolved
        return svc.review(
            pid, external=external, resolve_limit=resolve_limit
        ).to_markdown()
    finally:
        store.close()
