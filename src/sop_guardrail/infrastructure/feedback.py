"""Feedback quarantine: pending lessons are stored, only approved ones are served."""

from __future__ import annotations

import json
from pathlib import Path

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


class JsonFeedbackStore(InMemoryFeedbackStore):
    """The same quarantine, kept in a JSON file so it outlives one run.

    Curated memory is worth nothing if it dies with the process that recorded it:
    a lesson is written by one run, approved by a person, and read by a later one.
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if path.is_file():
            for raw in json.loads(path.read_text(encoding="utf-8")):
                card = FeedbackCard.model_validate(raw)
                self._items[card.feedback_id] = card

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cards = [self._items[key].model_dump(mode="json") for key in sorted(self._items)]
        self.path.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, feedback: FeedbackCard) -> None:
        super().add(feedback)
        self._flush()

    def activate(self, feedback_id: str, *, approved_by: str) -> FeedbackCard:
        card = super().activate(feedback_id, approved_by=approved_by)
        self._flush()
        return card

    def pending(self) -> tuple[FeedbackCard, ...]:
        return tuple(
            sorted(
                (item for item in self._items.values() if item.status is FeedbackStatus.PENDING),
                key=lambda item: item.feedback_id,
            )
        )
