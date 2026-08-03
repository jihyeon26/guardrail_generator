"""Infrastructure-independent interfaces used by the workflow."""

from __future__ import annotations

from typing import Protocol, TypeVar

from sop_guardrail.domain.models import (
    ContractModel,
    FeedbackCard,
    FeedbackStage,
    ModelResult,
    ModelTask,
)

OutputT = TypeVar("OutputT", bound=ContractModel)


class StructuredModelGateway(Protocol):
    """Invoke a provider and validate the response against a Pydantic contract."""

    def invoke(
        self,
        *,
        task: ModelTask,
        prompt: str,
        response_model: type[OutputT],
    ) -> ModelResult[OutputT]: ...


class FeedbackStore(Protocol):
    """Quarantine feedback and expose only explicitly activated lessons."""

    def add(self, feedback: FeedbackCard) -> None: ...

    def list_active(self, stage: FeedbackStage) -> tuple[FeedbackCard, ...]: ...

    def activate(self, feedback_id: str, *, approved_by: str) -> FeedbackCard: ...
