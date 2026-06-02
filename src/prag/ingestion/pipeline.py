"""Offline ingestion: read documents -> chunk -> index into the vector store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from prag.config import Settings
from prag.observability.logging import get_logger
from prag.observability.metrics import INGESTED_CHUNKS
from prag.retrieval.chunking import chunk_text
from prag.retrieval.store import Document, VectorStore

log = get_logger("ingestion")

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _doc_id(source: str, ordinal: int) -> str:
    return hashlib.sha1(f"{source}:{ordinal}".encode()).hexdigest()[:16]


def ingest_path(store: VectorStore, settings: Settings, path: str | Path) -> int:
    """Ingest a single file or every supported file in a directory. Returns chunk count."""
    root = Path(path)
    files = (
        [root]
        if root.is_file()
        else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    )

    total = 0
    for file in files:
        if file.suffix.lower() not in SUPPORTED_SUFFIXES:
            log.warning("skip_unsupported", file=str(file))
            continue
        text = _read_file(file)
        chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        docs = [
            Document(
                doc_id=_doc_id(file.name, c.ordinal),
                source=file.name,
                ordinal=c.ordinal,
                text=c.text,
            )
            for c in chunks
        ]
        added = store.add(docs)
        INGESTED_CHUNKS.inc(added)
        total += added
        log.info("ingested_file", file=file.name, chunks=added)

    log.info("ingestion_complete", files=len(files), chunks=total)
    return total
