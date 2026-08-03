from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sop_guardrail.application.workflow import build_workflow
from sop_guardrail.domain.models import (
    FeedbackStage,
    ModelTask,
    ReviewGate,
    ReviewVerdict,
)
from sop_guardrail.infrastructure.feedback import InMemoryFeedbackStore
from sop_guardrail.infrastructure.providers.demo import ScriptedModelGateway
from tests.helpers import (
    guardrail_compilation,
    passing_assessment,
    policy_extraction,
    synthetic_document,
)


def _responses(document_id: str = "synthetic-sop") -> dict[ModelTask, list[Any]]:
    document = synthetic_document(document_id)
    return {
        ModelTask.POLICY_EXTRACTION: [policy_extraction(document)],
        ModelTask.GUARDRAIL_COMPILATION: [guardrail_compilation(document)],
        ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
    }


def _config(run_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": run_id}}


def _decision(
    gate: ReviewGate,
    verdict: ReviewVerdict,
    *,
    lesson: str | None = None,
) -> dict[str, str | None]:
    return {
        "gate": gate.value,
        "verdict": verdict.value,
        "reviewer": "reviewer@example.test",
        "comment": "Synthetic reviewer decision.",
        "reusable_lesson": lesson,
    }


def test_happy_path_pauses_at_both_human_gates_and_releases() -> None:
    document = synthetic_document()
    gateway = ScriptedModelGateway(_responses())
    store = InMemoryFeedbackStore()
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=store,
        checkpointer=InMemorySaver(),
    )
    config = _config("run-happy")

    first = cast(
        dict[str, Any],
        graph.invoke(
            {"run_id": "run-happy", "document": document.model_dump(mode="json")},
            config=config,
        ),
    )
    assert first["status"] == "policies_validated"
    assert first["__interrupt__"]

    second = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )
    assert second["status"] == "llm_assessed"
    assert second["__interrupt__"]

    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )
    assert final["status"] == "released"
    assert final["release"]["approved_by"] == "reviewer@example.test"
    assert len(final["model_runs"]) == 3


def test_policy_rejection_creates_pending_feedback_and_stops() -> None:
    document = synthetic_document()
    gateway = ScriptedModelGateway(_responses())
    store = InMemoryFeedbackStore()
    graph = build_workflow(model_gateway=gateway, feedback_store=store)
    config = _config("run-reject")

    graph.invoke(
        {"run_id": "run-reject", "document": document.model_dump(mode="json")},
        config=config,
    )
    final = cast(
        dict[str, Any],
        graph.invoke(
            Command(
                resume=_decision(
                    ReviewGate.POLICY,
                    ReviewVerdict.REJECT,
                    lesson="Reject policies that lack an explicit actor.",
                )
            ),
            config,
        ),
    )

    assert final["status"] == "feedback_pending"
    assert "release" not in final
    feedback = store.get(final["feedback_ids"][0])
    assert feedback.lesson == "Reject policies that lack an explicit actor."
    assert store.list_active(FeedbackStage.POLICY_EXTRACTION) == ()


def test_activated_feedback_is_used_by_a_later_run() -> None:
    first_document = synthetic_document("synthetic-first")
    store = InMemoryFeedbackStore()
    first_gateway = ScriptedModelGateway(_responses("synthetic-first"))
    first_graph = build_workflow(model_gateway=first_gateway, feedback_store=store)
    first_config = _config("run-first")
    first_graph.invoke(
        {"run_id": "run-first", "document": first_document.model_dump(mode="json")},
        config=first_config,
    )
    rejected = cast(
        dict[str, Any],
        first_graph.invoke(
            Command(
                resume=_decision(
                    ReviewGate.POLICY,
                    ReviewVerdict.REVISE,
                    lesson="Keep the actor separate from the required action.",
                )
            ),
            first_config,
        ),
    )
    store.activate(rejected["feedback_ids"][0], approved_by="curator@example.test")

    second_document = synthetic_document("synthetic-second")
    second_gateway = ScriptedModelGateway(_responses("synthetic-second"))
    second_graph = build_workflow(model_gateway=second_gateway, feedback_store=store)
    second_graph.invoke(
        {"run_id": "run-second", "document": second_document.model_dump(mode="json")},
        config=_config("run-second"),
    )

    prompt = second_gateway.prompts[ModelTask.POLICY_EXTRACTION][0]
    assert "Keep the actor separate from the required action." in prompt


def test_unknown_evidence_fails_before_human_review() -> None:
    document = synthetic_document()
    gateway = ScriptedModelGateway(
        {ModelTask.POLICY_EXTRACTION: [policy_extraction(document, evidence_id="evidence-unknown")]}
    )
    graph = build_workflow(model_gateway=gateway, feedback_store=InMemoryFeedbackStore())

    final = cast(
        dict[str, Any],
        graph.invoke(
            {"run_id": "run-invalid", "document": document.model_dump(mode="json")},
            config=_config("run-invalid"),
        ),
    )

    assert final["status"] == "policy_validation_failed"
    assert "__interrupt__" not in final
    assert "unknown evidence" in final["validation_errors"][0]
