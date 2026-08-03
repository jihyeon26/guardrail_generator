"""In-memory feedback quarantine used by tests and local prototypes."""

from sop_guardrail.domain.models import FeedbackCard, FeedbackStage, FeedbackStatus


class InMemoryFeedbackStore:
    """Idempotent store that never returns pending feedback as active memory."""

    def __init__(self) -> None:
        self._items: dict[str, FeedbackCard] = {}

    def add(self, feedback: FeedbackCard) -> None:
        current = self._items.get(feedback.feedback_id)
        if current is not None and current != feedback:
            raise ValueError(f"feedback id already exists: {feedback.feedback_id}")
        self._items[feedback.feedback_id] = feedback

    def get(self, feedback_id: str) -> FeedbackCard:
        return self._items[feedback_id]

    def list_active(self, stage: FeedbackStage) -> tuple[FeedbackCard, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in self._items.values()
                    if item.stage is stage and item.status is FeedbackStatus.ACTIVE
                ),
                key=lambda item: item.feedback_id,
            )
        )

    def activate(self, feedback_id: str, *, approved_by: str) -> FeedbackCard:
        current = self._items[feedback_id]
        active = current.model_copy(
            update={"status": FeedbackStatus.ACTIVE, "approved_by": approved_by}
        )
        active = FeedbackCard.model_validate(active.model_dump())
        self._items[feedback_id] = active
        return active
