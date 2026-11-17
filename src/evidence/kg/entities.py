"""Per-paper entity extraction (PLAN.MD offline pipeline, step 3).

Produces a `PaperEntities` bundle of `(Paper, Author, Topic, Method)` records
plus the relations the knowledge graph needs:

    (Paper) -[:BY]-> (Author)
    (Paper) -[:ON_TOPIC]-> (Topic)
    (Paper) -[:USES_METHOD]-> (Method)

Extraction is LLM-driven when a model is available, with a deterministic
keyword/section fallback so the pipeline always yields *something* (PLAN.MD
"Key risks": prefer rule-based paths where they are reliable).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..models.ollama import Models

logger = logging.getLogger(__name__)

# A small, open-vocabulary stop-list of method-ish / topic-ish words we keep
# out of the deterministic fallback so it doesn't return generic noise.
_METHOD_HINTS = (
    "transformer",
    "attention",
    "fine-tuning",
    "rlhf",
    "rlaif",
    "dpo",
    "retrieval",
    "rag",
    "retrieval-augmented",
    "retrieval augmented",
    "rerank",
    "cross-encoder",
    "cross encoder",
    "bm25",
    "faiss",
    "reciprocal rank fusion",
    "quantization",
    "distillation",
    "contrastive learning",
    "supervised",
    "self-supervised",
    "pretrain",
    "pre-train",
    "pre-training",
    "in-context",
    "in context learning",
    "prompt",
    "chain-of-thought",
    "chain of thought",
    "mixture-of-experts",
    "mixture of experts",
    "kv cache",
    "speculative decoding",
    "llm",
    "language model",
    "multimodal",
    "vision-language",
    "vision language",
    "diffusion",
    "tokeniz",
)

_TOPIC_HINTS = (
    "retrieval-augmented generation",
    "retrieval augmented generation",
    "large language model",
    "large language models",
    "llm",
    "nlp",
    "natural language",
    "machine translation",
    "translation",
    "question answering",
    "qa",
    "summarization",
    "summarisation",
    "document retrieval",
    "information retrieval",
    "rag",
    "agents",
    "agentic",
    "reasoning",
    "citation",
    "bibliometric",
    "scientific literature",
    "technical literature",
    "multimodal",
    "grounding",
    "hallucination",
    "benchmark",
    "sota",
    "state of the art",
)


@dataclass
class PaperEntities:
    """Extracted entities for one paper (PLAN.MD offline pipeline, step 3)."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "topics": list(self.topics),
            "methods": list(self.methods),
        }


_SYSTEM = (
    "Extract entities from a technical paper. Respond with JSON: "
    "{\"authors\": [str], \"topics\": [str], \"methods\": [str]}. "
    "topics = the subject areas / application domains the paper is about. "
    "methods = the concrete techniques, algorithms, or model families it uses. "
    "Use normalized, lowercase, hyphenated labels (e.g. 'retrieval-augmented-generation'). "
    "Keep to the 3-8 most salient of each. Do not invent authors."
)


def _norm_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_]+", " ", s)
    return s


def extract_entities(
    *,
    paper_id: str,
    title: str,
    text: str,
    authors: list[str] | None = None,
    models: Models | None = None,
    cfg=None,
) -> PaperEntities:
    """Extract (authors, topics, methods) for one paper.

    `authors` may be supplied from the source (PDF metadata / LaTeX) and takes
    precedence for the author field; the LLM only fills topics + methods and
    can add authors it finds in the body.
    """
    authors = [a.strip() for a in (authors or []) if a.strip()]
    topics: list[str] = []
    methods: list[str] = []

    if models is not None:
        data = models.complete_json(
            prompt=(
                f"Title: {title}\n\n"
                f"Paper (first ~3000 chars):\n{text[:3000]}\n\n"
                "Extract the entities."
            ),
            model=getattr(cfg.ollama, "light", None) if cfg else None,
            system=_SYSTEM,
            temperature=0.0,
            fallback=None,
        )
        if isinstance(data, dict):
            for kind, bucket in (("topics", topics), ("methods", methods), ("authors", authors)):
                raw = data.get(kind)
                if isinstance(raw, list):
                    for item in raw:
                        label = _norm_label(str(item))
                        if label and label not in bucket:
                            bucket.append(label)
        else:
            logger.debug("LLM entity extraction returned non-dict; using fallback")

    # Deterministic fallback / augmentation (cheap, reliable for methods/topics).
    haystack = f"{title} {text[:4000]}".lower()
    for hint in _METHOD_HINTS:
        if hint in haystack and hint not in methods:
            methods.append(_norm_label(hint))
    for hint in _TOPIC_HINTS:
        if hint in haystack and hint not in topics:
            topics.append(_norm_label(hint))

    topics = topics[:8]
    methods = methods[:8]

    return PaperEntities(
        paper_id=paper_id,
        title=title,
        authors=authors,
        topics=topics,
        methods=methods,
    )
