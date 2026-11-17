from evidence.config import ChunkingConfig
from evidence.ingestion.chunker import chunk_document, count_tokens, split_sections


def test_count_tokens_positive():
    assert count_tokens("hello world") >= 2


def test_split_sections_on_markdown():
    text = "# Intro\n\nFirst para.\n\n## Method\n\nSecond para.\n"
    secs = split_sections(text)
    assert [s.heading for s in secs] == ["Intro", "Method"]


def test_split_sections_noheadings_single_section():
    secs = split_sections("just some words\nwith no headings")
    assert len(secs) == 1 and secs[0].heading == ""


def test_chunk_document_nonempty_and_text_preserved():
    cfg = ChunkingConfig()
    text = ("Intro paragraph with enough words to be a real chunk of text.\n\n"
            "## Details\n\n"
            "Another paragraph that carries distinct information about the method.\n")
    chunks = chunk_document(text, cfg)
    assert len(chunks) >= 1
    # Every chunk has non-empty text and a token estimate.
    for c in chunks:
        assert c.text.strip()
        assert c.tokens >= 0


def test_chunk_document_respects_max_tokens_loosely():
    cfg = ChunkingConfig(max_tokens=30, overlap_tokens=5, min_chunk_tokens=4)
    text = (" ".join(f"word{i}" for i in range(400)) +
            "\n\n" + " ".join(f"tail{i}" for i in range(20)))
    chunks = chunk_document(text, cfg)
    assert len(chunks) > 1  # must split a long document
    # No chunk should wildly exceed the budget (allow tokenizer slack).
    max_len = max(len(c.text) for c in chunks)
    assert max_len < 4000
