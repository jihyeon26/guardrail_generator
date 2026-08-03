import pytest

from sop_guardrail.domain.models import FeedbackCard, FeedbackStage, ReviewGate
from sop_guardrail.infrastructure.feedback import InMemoryFeedbackStore


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
