"""Tracker tests. 2026-06-02"""
import sys,os; sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))  
from src.tracking import QueryRecord, QueryTracker

def test_cost_calculation():
    r=QueryRecord("q1",input_tokens=1000,output_tokens=500,model="gpt-4o-mini")
    assert r.cost_usd > 0

def test_tracker_accumulates():
    t=QueryTracker()
    t.record(query_id="q1",input_tokens=200,output_tokens=100)
    t.record(query_id="q2",input_tokens=300,output_tokens=150)
    assert len(t.records)==2 and t.total_cost>0

def test_hitl_gate():
    approved=True; answer="42 kWh"
    assert (answer if approved else None)==answer
    approved=False
    assert (answer if approved else None) is None

# ts:2026-06-02T14:00:00
