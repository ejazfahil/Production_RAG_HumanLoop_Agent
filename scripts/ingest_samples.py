"""CLI: ingest documents into the configured vector store.

Usage:
    python scripts/ingest_samples.py [PATH]   # defaults to data/sample
"""

from __future__ import annotations

import sys

from prag.engine import Engine
from prag.observability.logging import configure_logging


def main() -> int:
    configure_logging()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    engine = Engine().start()
    try:
        count = engine.ingest(path)
    finally:
        engine.close()
    print(f"Ingested {count} chunks from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
