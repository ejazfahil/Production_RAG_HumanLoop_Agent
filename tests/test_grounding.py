"""The grounding proxy flags supported vs unsupported drafts."""

from __future__ import annotations

from prag.agent.nodes import _groundedness


def test_supported_draft_scores_high() -> None:
    context = "The accuracy class is 0.5S and active accuracy is plus minus 0.5 percent."
    draft = "The accuracy class is 0.5S."
    assert _groundedness(draft, context) >= 0.6


def test_unsupported_draft_scores_low() -> None:
    context = "The accuracy class is 0.5S."
    draft = "The meter supports quantum entanglement and faster than light telemetry."
    assert _groundedness(draft, context) < 0.6


def test_empty_draft_is_not_grounded() -> None:
    assert _groundedness("", "some context") == 0.0
