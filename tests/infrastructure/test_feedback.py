from pathlib import Path

import pytest

from sop_guardrail.domain.models import FeedbackCard, FeedbackStage, ReviewGate
from sop_guardrail.infrastructure.feedback import InMemoryFeedbackStore, JsonFeedbackStore


def _pending_feedback() -> FeedbackCard:
    return FeedbackCard(
        feedback_id="feedback-1",
        stage=FeedbackStage.POLICY_EXTRACTION,
        lesson="Keep the actor separate from the required action.",
        source_run_id="run-1",
        source_gate=ReviewGate.POLICY,
    )


def test_store_exposes_feedback_only_after_activation() -> None:
    store = InMemoryFeedbackStore()
    pending = _pending_feedback()
    store.add(pending)

    assert store.list_active(FeedbackStage.POLICY_EXTRACTION) == ()

    active = store.activate(pending.feedback_id, approved_by="curator@example.test")

    assert store.get(pending.feedback_id) == active
    assert store.list_active(FeedbackStage.POLICY_EXTRACTION) == (active,)


def test_add_is_idempotent_but_rejects_conflicting_id() -> None:
    store = InMemoryFeedbackStore()
    first = _pending_feedback()
    store.add(first)
    store.add(first)

    conflicting = first.model_copy(update={"lesson": "Different lesson."})
    with pytest.raises(ValueError, match="already exists"):
        store.add(conflicting)


def _card(feedback_id: str = "feedback-1") -> FeedbackCard:
    return FeedbackCard(
        feedback_id=feedback_id,
        stage=FeedbackStage.POLICY_EXTRACTION,
        lesson="Name the approving role explicitly.",
        source_run_id="run-1",
        source_gate=ReviewGate.POLICY,
    )


def test_a_lesson_survives_the_process_that_recorded_it(tmp_path: Path) -> None:
    """Curated memory is worthless if it dies with the run that wrote it."""

    path = tmp_path / "feedback.json"
    JsonFeedbackStore(path).add(_card())

    reopened = JsonFeedbackStore(path)

    assert reopened.get("feedback-1").lesson == "Name the approving role explicitly."
    assert reopened.list_active(FeedbackStage.POLICY_EXTRACTION) == ()
    assert [card.feedback_id for card in reopened.pending()] == ["feedback-1"]


def test_activation_is_persisted_and_then_served(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    store = JsonFeedbackStore(path)
    store.add(_card())

    store.activate("feedback-1", approved_by="curator@example.test")
    reopened = JsonFeedbackStore(path)

    active = reopened.list_active(FeedbackStage.POLICY_EXTRACTION)
    assert [card.feedback_id for card in active] == ["feedback-1"]
    assert active[0].approved_by == "curator@example.test"
    assert reopened.pending() == ()


def test_a_missing_file_starts_an_empty_quarantine(tmp_path: Path) -> None:
    store = JsonFeedbackStore(tmp_path / "nested" / "feedback.json")

    assert store.pending() == ()
    assert store.list_active(FeedbackStage.POLICY_EXTRACTION) == ()
