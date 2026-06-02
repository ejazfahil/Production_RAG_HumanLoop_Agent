"""Chunking tuned for technical documents.

Technical specs are full of tables and unit-bearing lines (e.g. "Accuracy: ±0.5%").
We keep markdown table blocks intact and avoid splitting mid-line so units and their
values stay together in the same chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    ordinal: int


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Greedy line-aware chunker that never splits inside a markdown table row.

    Adjacent table lines are grouped so a chunk contains whole rows, and a small
    character overlap preserves context across boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    lines = text.splitlines()

    # Group consecutive table lines into atomic blocks.
    blocks: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if _is_table_line(line):
            buffer.append(line)
            continue
        if buffer:
            blocks.append("\n".join(buffer))
            buffer = []
        blocks.append(line)
    if buffer:
        blocks.append("\n".join(buffer))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    ordinal = 0

    def flush() -> None:
        nonlocal current, current_len, ordinal
        body = "\n".join(current).strip()
        if body:
            chunks.append(Chunk(text=body, ordinal=ordinal))
            ordinal += 1
        current = []
        current_len = 0

    for block in blocks:
        block_len = len(block)
        if current_len + block_len > chunk_size and current:
            flush()
            if overlap > 0 and chunks:
                tail = chunks[-1].text[-overlap:]
                current = [tail]
                current_len = len(tail)
        current.append(block)
        current_len += block_len + 1

    flush()
    return chunks
