"""Full-path smoke tests exercising the LangGraph HITL flow via the Engine."""

from __future__ import annotations

from prag.config import Settings
from prag.engine import Engine


def test_query_pauses_for_human_approval(engine: Engine) -> None:
    res = engine.query("What is the accuracy class of the X200?", thread_id="t-approve")
    assert res.status == "pending_approval"
    assert res.draft
    assert res.sources  # at least one cited source surfaced to the reviewer


def test_approve_returns_grounded_answer(engine: Engine) -> None:
    engine.query("What is the accuracy class of the X200?", thread_id="t1")
    final = engine.resume("t1", action="approve")
    assert final.status == "answered"
    assert final.answer


def test_edit_overrides_the_draft(engine: Engine) -> None:
    engine.query("What is the operating temperature range?", thread_id="t2")
    final = engine.resume("t2", action="edit", text="Operating range is -25C to +70C.")
    assert final.status == "answered"
    assert final.answer == "Operating range is -25C to +70C."


def test_reject_produces_no_answer_but_is_audited(engine: Engine) -> None:
    engine.query("How do I replace the backup battery?", thread_id="t3")
    final = engine.resume("t3", action="reject")
    assert final.status == "rejected"
    assert not final.answer
    record = engine.audit.get("t3")
    assert record is not None
    assert record.decision == "reject"


def test_unanswerable_query_abstains() -> None:
    settings = Settings(
        llm_provider="fake",
        embedding_provider="fake",
        vector_backend="memory",
        checkpointer="memory",
        min_retrieval_score=0.99,  # force the low-confidence branch
    )
    eng = Engine(settings).start()
    try:
        eng.ingest("data/sample")
        res = eng.query("What is the capital of France?", thread_id="t4")
        assert res.status == "abstained"
        assert "Not supported" in (res.answer or "")
    finally:
        eng.close()


def test_audit_record_captures_cost_and_latency(engine: Engine) -> None:
    engine.query("What is the accuracy class of the X200?", thread_id="t5")
    engine.resume("t5", action="approve")
    record = engine.audit.get("t5")
    assert record is not None
    assert record.latency_seconds > 0
    assert record.input_tokens > 0
    assert record.retrieved_doc_ids
