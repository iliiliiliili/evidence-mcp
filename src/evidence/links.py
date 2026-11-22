"""Chunk-graph: deterministic + LLM links between chunks (hierarchical navigation).

The pipeline never hard-fails: when no model is available or the model
returns garbage, we still emit *deterministic* links (continuation, same-
heading "see_also") so the graph is always at least as good as a linked list.
When a reasoning model is available, we additionally ask it for semantic
links (see_also, depends_on, cites, contrasts, example) between non-adjacent
chunks, with a short reason and a confidence in [0, 1].

All links are persisted in the `Store.chunk_links` table (src, dst,
relation, confidence, reason) so they can be queried across sessions without
re-invoking the model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import EvidenceConfig
from .indexes.store import Store
from .models.ollama import Models

logger = logging.getLogger(__name__)


# Relation taxonomy. Anything outside this set is coerced to "related".
_KNOWN_RELATIONS = {
    "continuation",   # this chunk follows / continues the other (adjacent)
    "see_also",       # thematically related, not adjacent
    "depends_on",     # this chunk's claim depends on the other's setup
    "cites",          # this chunk references work introduced in the other
    "contrasts",      # this chunk contrasts / disagrees with the other
    "example",        # this chunk gives a concrete instance of the other
    "related",        # generic LLM-assigned relation
}


@dataclass
class LinkReport:
    """Summary of a chunk-graph build run."""

    papers_processed: int = 0
    links_emitted: int = 0
    llm_links: int = 0
    deterministic_links: int = 0
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "papers_processed": self.papers_processed,
            "links_emitted": self.links_emitted,
            "llm_links": self.llm_links,
            "deterministic_links": self.deterministic_links,
            "skipped": self.skipped,
        }


# ── helpers ────────────────────────────────────────────────────────────────

def _chunk_brief(chunk, max_chars: int = 200) -> dict:
    """Compact per-chunk view for prompts and link payloads."""
    meta = chunk.meta or {}
    return {
        "chunk_id": chunk.chunk_id,
        "heading": chunk.heading or "",
        "section_id": meta.get("section_id", ""),
        "page": chunk.page,
        "ordinal": chunk.ordinal,
        "tokens": chunk.tokens,
        "preview": (chunk.text or "")[:max_chars],
    }


def _deterministic_links(chunks: list) -> list[dict]:
    """Links that hold even with no model: continuation (adjacent) and
    same-section-see_also (non-adjacent chunks that share a section_id or
    heading)."""
    links: list[dict] = []
    by_section: dict[str, list] = {}
    for ch in chunks:
        meta = ch.meta or {}
        key = meta.get("section_id") or f"__h__{ch.heading}"
        by_section.setdefault(key, []).append(ch)

    # Continuation: each chunk points to its ordinal successor (if present).
    by_ordinal = {ch.ordinal: ch for ch in chunks}
    for ch in chunks:
        nxt = by_ordinal.get(ch.ordinal + 1)
        if nxt is not None and nxt.chunk_id != ch.chunk_id:
            links.append(
                {
                    "src": ch.chunk_id,
                    "dst": nxt.chunk_id,
                    "relation": "continuation",
                    "confidence": 1.0,
                    "reason": "adjacent chunk in document order",
                }
            )

    # see_also: non-adjacent chunks within the same detected section.
    for key, group in by_section.items():
        if len(group) < 3:
            continue  # only worth linking when there's a real span
        for i, a in enumerate(group):
            for b in group[i + 2:]:  # skip the immediate successor (already continuation)
                if a.chunk_id == b.chunk_id:
                    continue
                links.append(
                    {
                        "src": a.chunk_id,
                        "dst": b.chunk_id,
                        "relation": "see_also",
                        "confidence": 0.5,
                        "reason": f"same section: {a.heading or key}",
                    }
                )
    return links


def _llm_links_for_paper(
    chunks: list, models: Models, cfg: EvidenceConfig, *, cap: int = 30
) -> list[dict]:
    """Ask the reasoning model for semantic links between a paper's chunks.

    Returns a (possibly empty) list of link dicts. Returns [] on any model
    failure — the caller is expected to have already emitted the deterministic
    links, so a model failure never loses the base graph.
    """
    if models is None or not chunks:
        return []
    if len(chunks) < 2:
        return []
    model = cfg.ollama.reasoning
    briefs = [_chunk_brief(c, max_chars=160) for c in chunks[:cap]]
    system = (
        "You are linking the chunks of a research paper so a reader can "
        "navigate between them. Return STRICT JSON only — no prose, no "
        "markdown fences, no trailing commas."
    )
    user = (
        "Paper chunks (in document order):\n"
        f"{_dump_chunks_json(briefs)}\n\n"
        "Task: return the non-obvious semantic links between these chunks — "
        "places where one chunk's claim depends on, cites, contrasts, or "
        "exemplifies another chunk's content. DO NOT return continuation links "
        "(adjacent chunks are already linked). For each link give:\n"
        "- src_chunk_id, dst_chunk_id (must both appear in the list above)\n"
        "- relation: one of 'see_also' | 'depends_on' | 'cites' | 'contrasts' "
        "| 'example' | 'related'\n"
        "- confidence: number in [0, 1]\n"
        "- reason: a SHORT PHRASE, 3-8 words max (NOT a full sentence). It is a "
        "label another model will read to decide whether to open the linked "
        "chunk, so make it specific and navigational — e.g. 'derives objective "
        "from', 'baseline method', 'defines KG schema', 'ablation study'. "
        "Do NOT explain; do NOT repeat the relation word.\n\n"
        "Return STRICT JSON with exactly this shape:\n"
        '{"links": [{"src_chunk_id": "...", "dst_chunk_id": "...", '
        '"relation": "see_also", "confidence": 0.7, "reason": "derives objective from"}]}\n'
        "An empty list is a valid answer when no non-obvious links exist."
    )
    raw = models.complete_json(
        user, model=model, system=system, temperature=0.0, fallback=None
    )
    if not isinstance(raw, dict):
        return []
    links_in = raw.get("links")
    if not isinstance(links_in, list):
        return []
    valid_ids = {b["chunk_id"] for b in briefs}
    out: list[dict] = []
    for ln in links_in:
        if not isinstance(ln, dict):
            continue
        src = str(ln.get("src_chunk_id") or "").strip()
        dst = str(ln.get("dst_chunk_id") or "").strip()
        if src not in valid_ids or dst not in valid_ids or src == dst:
            continue
        rel = str(ln.get("relation") or "").strip().lower()
        if rel not in _KNOWN_RELATIONS:
            rel = "related"
        try:
            conf = float(ln.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        # `reason` is a short navigation phrase (3-8 words), not a sentence.
        # Truncate defensively to ~60 chars so a chatty model can't blow it up.
        reason = re.sub(r"\s+", " ", str(ln.get("reason") or "").strip())[:60]
        out.append(
            {
                "src": src,
                "dst": dst,
                "relation": rel,
                "confidence": conf,
                "reason": reason,
            }
        )
    return out


def _dump_chunks_json(briefs: list[dict]) -> str:
    import json

    return json.dumps(briefs, indent=2)


# ── public API ─────────────────────────────────────────────────────────────

def build_chunk_links(
    store: Store,
    *,
    models: Models | None = None,
    cfg: EvidenceConfig | None = None,
    paper_id: str | None = None,
    reset: bool = True,
) -> LinkReport:
    """Build (or update) the chunk graph over `store`.

    Emits deterministic links (always) and, when `models` is available,
    LLM links on top. Existing links are wiped first if `reset=True`
    (scoped to `paper_id` when given).

    The pipeline never hard-fails: a model error just means fewer links.
    """
    report = LinkReport()
    papers = store.list_papers()
    if not papers:
        logger.warning("build_chunk_links: no papers in store; nothing to do")
        return report
    if paper_id is not None:
        papers = [p for p in papers if p.paper_id == paper_id]
        if not papers:
            logger.warning("build_chunk_links: paper %s not found", paper_id)
            return report

    if reset:
        store.clear_chunk_links(paper_id)

    for p in papers:
        chunks = store.chunks_for_paper(p.paper_id)
        if len(chunks) < 2:
            report.papers_processed += 1
            continue
        det = _deterministic_links(chunks)
        store.upsert_chunk_links(det)
        report.deterministic_links += len(det)
        paper_llm = 0
        try:
            llm = _llm_links_for_paper(chunks, models, cfg) if (models and cfg) else []
            if llm:
                store.upsert_chunk_links(llm)
                paper_llm = len(llm)
        except Exception as exc:  # noqa: BLE001 - never hard-fail
            logger.warning("LLM link pass failed for %s: %s", p.paper_id, exc)
        report.llm_links += paper_llm
        report.links_emitted += len(det) + paper_llm
        report.papers_processed += 1
    return report


def linked_chunks(
    chunk_id: str,
    *,
    store: Store | None = None,
    direction: str = "both",
    min_confidence: float = 0.0,
    include_self: bool = False,
) -> dict:
    """Neighbour chunks of `chunk_id`, with the relation + reason of each link.

    `direction` is "out", "in", or "both". Each neighbour entry carries the
    linked chunk's preview (first 200 chars) so a client can show the link
    target inline without a second round-trip.
    """
    from . import facade  # local import to avoid a cycle at module load

    if store is None:
        store = facade.get_app().store
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be one of: out, in, both")
    rec = store.get_chunk(chunk_id)
    if rec is None:
        return {"chunk_id": chunk_id, "paper_id": None, "links": [], "neighbours": []}
    links = store.links_for_chunk(chunk_id, direction=direction)
    links = [
        ln for ln in links
        if ln.get("confidence", 0.0) >= min_confidence
        and (direction != "out" or ln["src_chunk_id"] == chunk_id)
        and (direction != "in" or ln["dst_chunk_id"] == chunk_id)
    ]
    by_id: dict[str, dict] = {}
    for ln in links:
        if direction == "out":
            other = ln["dst_chunk_id"]
            other_dir = "out"
        elif direction == "in":
            other = ln["src_chunk_id"]
            other_dir = "in"
        else:
            other = ln["dst_chunk_id"] if ln["src_chunk_id"] == chunk_id else ln["src_chunk_id"]
            other_dir = "out" if ln["src_chunk_id"] == chunk_id else "in"
        if other == chunk_id and not include_self:
            continue
        existing = by_id.get(other)
        if existing is None:
            other_rec = store.get_chunk(other)
            other_paper = store.get_paper(other_rec.paper_id) if other_rec else None
            existing = {
                "chunk_id": other,
                "paper_id": other_rec.paper_id if other_rec else None,
                "paper_title": (facade._best_title(other_paper, store) if other_paper else ""),
                "heading": other_rec.heading if other_rec else "",
                "page": other_rec.page if other_rec else None,
                "ordinal": other_rec.ordinal if other_rec else None,
                "preview": (other_rec.text or "")[:200] if other_rec else "",
                "links": [],
            }
            by_id[other] = existing
        existing["links"].append(
            {
                "relation": ln.get("relation", "related"),
                "confidence": ln.get("confidence", 1.0),
                "reason": ln.get("reason", ""),
                "direction": other_dir,
            }
        )
    return {
        "chunk_id": chunk_id,
        "paper_id": rec.paper_id,
        "n_neighbours": len(by_id),
        "links": links,
        "neighbours": list(by_id.values()),
    }


def links_for_paper(
    paper_id: str,
    *,
    store: Store | None = None,
    min_confidence: float = 0.0,
) -> dict:
    """All links whose source chunk belongs to `paper_id`, in confidence order.

    Cross-paper links (where the destination is in another paper) are included
    so a reader can follow a "cites" edge to the cited paper's chunk.
    """
    from . import facade  # local import to avoid a cycle at module load

    if store is None:
        store = facade.get_app().store
    p = store.get_paper(paper_id)
    if p is None:
        return {"paper_id": paper_id, "n_links": 0, "links": []}
    links = store.links_for_paper(paper_id, min_confidence=min_confidence)
    for ln in links:
        src_rec = store.get_chunk(ln["src_chunk_id"])
        dst_rec = store.get_chunk(ln["dst_chunk_id"])
        ln["src_preview"] = (src_rec.text or "")[:160] if src_rec else ""
        ln["dst_preview"] = (dst_rec.text or "")[:160] if dst_rec else ""
        ln["dst_paper_id"] = dst_rec.paper_id if dst_rec else None
    return {
        "paper_id": paper_id,
        "n_links": len(links),
        "links": links,
    }
