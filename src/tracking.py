"""Cost and latency tracking. 2026-06-02"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import time

@dataclass
class QueryRecord:
    query_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "gpt-4o-mini"
    approved: bool = False

    @property
    def cost_usd(self):
        pricing = {"gpt-4o-mini": {"i": 0.15, "o": 0.60}}
        p = pricing.get(self.model, {"i": 5.0, "o": 15.0})
        return (self.input_tokens*p["i"] + self.output_tokens*p["o"]) / 1e6

@dataclass
class QueryTracker:
    records: List[QueryRecord] = field(default_factory=list)
    def record(self, **kw): r=QueryRecord(**kw); self.records.append(r); return r
    @property
    def total_cost(self): return sum(r.cost_usd for r in self.records)
    @property
    def avg_latency_ms(self): return sum(r.latency_ms for r in self.records)/len(self.records) if self.records else 0

# ts:2026-06-02T11:45:00
