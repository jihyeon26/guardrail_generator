from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sop_guardrail.application.workflow import WorkflowDependencies, build_workflow
from sop_guardrail.domain.models import (
    EvidenceSpan,
    FeedbackStage,
    GuardrailCompilation,
    GuardrailDecision,
    GuardrailRule,
    ModelTask,
    PolicyCandidate,
    PolicyExtraction,
    ReviewGate,
    ReviewVerdict,
    RuleTestCase,
    Severity,
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


def _numbered_policies(document_id: str, count: int) -> PolicyExtraction:
    document = synthetic_document(document_id)
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    return PolicyExtraction(
        policies=tuple(
            PolicyCandidate(
                policy_id=f"policy-{index:02d}",
                title=f"Control {index}",
                statement="A reviewer must approve the step.",
                actor="reviewer",
                action=f"approve step {index}",
                evidence_refs=(evidence_id,),
                confidence=0.9,
            )
            for index in range(count)
        )
    )


def _rules_for(policies: tuple[PolicyCandidate, ...], evidence_id: str) -> GuardrailCompilation:
    return GuardrailCompilation(
        rules=tuple(
            GuardrailRule(
                rule_id=f"rule-{policy.policy_id}",
                policy_id=policy.policy_id,
                decision=GuardrailDecision.ESCALATE,
                condition="the step has no recorded approval",
                rationale="The policy requires approval.",
                severity=Severity.MEDIUM,
                evidence_refs=(evidence_id,),
                test_cases=(
                    RuleTestCase(
                        name="missing approval escalates",
                        input_summary="step without approval",
                        expected_decision=GuardrailDecision.ESCALATE,
                    ),
                ),
            )
            for policy in policies
        )
    )


def test_compilation_batches_policies_into_separate_model_calls() -> None:
    document = synthetic_document("batched-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("batched-sop", 5)
    batches = [extraction.policies[0:2], extraction.policies[2:4], extraction.policies[4:5]]
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [_rules_for(batch, evidence_id) for batch in batches],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=2,
    )
    config = _config("run-batched")

    graph.invoke({"run_id": "run-batched", "document": document.model_dump(mode="json")}, config)
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )

    assert len(gateway.prompts[ModelTask.GUARDRAIL_COMPILATION]) == 3
    assert len(final["guardrails"]) == 5
    assert final["validation_errors"] == []
    assert final["status"] == "released"
    # one extraction run, three compilation runs, one assessment run
    assert len(final["model_runs"]) == 5


def test_each_compilation_batch_sees_only_its_own_policies() -> None:
    document = synthetic_document("batched-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("batched-sop", 4)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [
                _rules_for(extraction.policies[0:2], evidence_id),
                _rules_for(extraction.policies[2:4], evidence_id),
            ],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=2,
    )
    config = _config("run-batch-prompts")

    graph.invoke(
        {"run_id": "run-batch-prompts", "document": document.model_dump(mode="json")}, config
    )
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)

    first, second = gateway.prompts[ModelTask.GUARDRAIL_COMPILATION]
    assert "policy-00" in first
    assert "policy-01" in first
    assert "policy-02" not in first
    assert "policy-02" in second
    assert "policy-03" in second
    assert "policy-00" not in second
    assert "(2 to compile)" in first


def test_a_policy_left_without_a_rule_blocks_the_release() -> None:
    document = synthetic_document("uncovered-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("uncovered-sop", 2)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            # The model answers the whole batch with a rule for one policy only.
            ModelTask.GUARDRAIL_COMPILATION: [_rules_for(extraction.policies[:1], evidence_id)],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=10,
    )
    config = _config("run-uncovered")

    graph.invoke({"run_id": "run-uncovered", "document": document.model_dump(mode="json")}, config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )

    assert final["status"] == "guardrail_validation_failed"
    assert final["validation_errors"] == ["policies without a guardrail rule: ['policy-01']"]
    assert "release" not in final
    assert "__interrupt__" not in final
    assert gateway.prompts[ModelTask.GUARDRAIL_ASSESSMENT] == []


def test_a_batch_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="compilation_batch_size must be at least 1"):
        WorkflowDependencies(
            model_gateway=ScriptedModelGateway({}),
            feedback_store=InMemoryFeedbackStore(),
            compilation_batch_size=0,
        )
