"""Chunking behaviour: bounded size, overlap, and table rows kept intact."""

from __future__ import annotations

from prag.retrieval.chunking import chunk_text


def test_chunks_respect_size_budget() -> None:
    text = "\n".join(f"line number {i} with some words" for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=40)
    assert len(chunks) > 1
    # Allow a little slack for the line that tips a chunk over the budget + overlap.
    assert all(len(c.text) <= 300 + 60 for c in chunks)


def test_ordinals_are_sequential() -> None:
    chunks = chunk_text("a\nb\nc\nd\ne\nf", chunk_size=4, overlap=0)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_markdown_table_rows_not_split_mid_row() -> None:
    table = "| h1 | h2 |\n| -- | -- |\n| 1 | 2 |\n| 3 | 4 |"
    text = f"intro paragraph\n{table}\noutro paragraph"
    chunks = chunk_text(text, chunk_size=40, overlap=0)
    # Every '|' line that appears must appear as a complete row in some chunk.
    joined = "\n".join(c.text for c in chunks)
    for row in table.splitlines():
        assert row in joined


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("   \n  \n", chunk_size=100) == []
