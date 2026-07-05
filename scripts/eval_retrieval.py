#!/usr/bin/env python3
"""Retrieval-quality eval: fake (hash bag-of-words) vs a real embedder.

The repo previously shipped only the offline `FakeEmbedder`, so retrieval quality
was never validated. This script wires the real Ollama `nomic-embed-text` embedder
and measures, on the sample corpus, whether each embedder retrieves the *right
source document* for a set of hand-labelled, deliberately paraphrased queries
(keyword overlap alone is weak, so semantics matter).

Metrics per embedder: hit@1, hit@k (top_k), mean top-1 cosine score. Results are
written to results/retrieval_eval.json. Real numbers only — no fabrication.

    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prag.config import Settings  # noqa: E402
from prag.ingestion.pipeline import ingest_path  # noqa: E402
from prag.llm.factory import build_embedder  # noqa: E402
from prag.retrieval.store import InMemoryVectorStore  # noqa: E402

SAMPLES = ROOT / "data" / "sample"
SPEC = "meter_spec_acme_x200.md"
MANUAL = "maintenance_manual_excerpt.md"

# (paraphrased query, expected source doc). Wording avoids exact term overlap so a
# hash bag-of-words embedder cannot win on lexical match alone.
QUERIES: list[tuple[str, str]] = [
    ("What is the measurement accuracy class of the meter?", SPEC),
    ("What is the maximum current the meter can handle?", SPEC),
    ("What supply voltage does the device run on?", SPEC),
    ("What is the rated reference current?", SPEC),
    ("How long does the backup battery last before replacement?", MANUAL),
    ("Why might the meter fail to respond over the serial connection?", MANUAL),
    ("What should a technician do if the terminal cover seal is broken?", MANUAL),
    ("What is the default communication speed?", MANUAL),
]


def run(provider: str, model: str, dim: int) -> dict:
    settings = Settings(embedding_provider=provider, embedding_model=model, embedding_dim=dim)
    store = InMemoryVectorStore(embedder=build_embedder(settings))
    ingest_path(store, settings, SAMPLES)

    hit1 = hitk = 0
    top1_scores: list[float] = []
    per_query = []
    t0 = time.time()
    for query, expected in QUERIES:
        res = store.search(query, top_k=settings.top_k)
        top_src = res[0].document.source if res else None
        in_topk = any(r.document.source == expected for r in res)
        h1 = int(top_src == expected)
        hit1 += h1
        hitk += int(in_topk)
        top1_scores.append(res[0].score if res else 0.0)
        per_query.append({"query": query, "expected": expected, "top_source": top_src,
                          "top1_score": round(res[0].score, 3) if res else 0.0, "hit@1": h1})

    n = len(QUERIES)
    return {
        "embedder": f"{provider}/{model}",
        "n": n,
        "hit@1": round(hit1 / n, 3),
        "hit@k": round(hitk / n, 3),
        "mean_top1_score": round(sum(top1_scores) / n, 3),
        "seconds": round(time.time() - t0, 1),
        "per_query": per_query,
    }


def main() -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    results = [
        run("fake", "fake-embed", 384),
        run("ollama", "nomic-embed-text", 768),
    ]
    payload = {"corpus": [SPEC, MANUAL], "queries": len(QUERIES), "top_k": 4, "results": results}
    (out_dir / "retrieval_eval.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'embedder':28s} {'hit@1':>6s} {'hit@k':>6s} {'top1':>6s}")
    for r in results:
        print(f"{r['embedder']:28s} {r['hit@1']:6.2f} {r['hit@k']:6.2f} {r['mean_top1_score']:6.3f}")
    print(f"\nwrote {out_dir / 'retrieval_eval.json'}")


if __name__ == "__main__":
    main()
