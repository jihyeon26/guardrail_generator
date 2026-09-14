from collections import defaultdict
from re import findall
from threading import Lock
from time import sleep
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sop_guardrail.application.workflow import WorkflowDependencies, build_workflow
from sop_guardrail.domain.models import (
    AssessmentFinding,
    AssessmentVerdict,
    EvidenceSpan,
    FeedbackStage,
    GuardrailCompilation,
    GuardrailDecision,
    GuardrailRule,
    LLMAssessment,
    ModelMetadata,
    ModelResult,
    ModelTask,
    PolicyCandidate,
    PolicyExtraction,
    PolicyModality,
    ReviewGate,
    ReviewVerdict,
    RuleTestCase,
    Severity,
    SopDocument,
)
from sop_guardrail.domain.segmentation import segment_document
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
                modality=PolicyModality.REQUIRED,
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
            # The model answers with a rule for one policy only, every attempt.
            ModelTask.GUARDRAIL_COMPILATION: [_rules_for(extraction.policies[:1], evidence_id)] * 3,
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
    # Every attempt was spent before the gap was reported.
    assert len(gateway.prompts[ModelTask.GUARDRAIL_COMPILATION]) == 3


def test_a_batch_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="compilation_batch_size must be at least 1"):
        WorkflowDependencies(
            model_gateway=ScriptedModelGateway({}),
            feedback_store=InMemoryFeedbackStore(),
            compilation_batch_size=0,
        )


_SECTIONED_SOP = """1. Purpose
This procedure governs the release of outbound payments.

2. Approval
A reviewer must approve a high-impact change before the operator completes it.
"""


def _sectioned_document() -> SopDocument:
    return SopDocument.from_text(
        document_id="sectioned-sop",
        source_name="payment-release.txt",
        text=_SECTIONED_SOP,
    )


def test_ingest_cites_sections_rather_than_the_whole_document() -> None:
    document = _sectioned_document()
    spans = segment_document(document)
    gateway = ScriptedModelGateway(
        {ModelTask.POLICY_EXTRACTION: [_policy_citing(spans[1].evidence_id)]}
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
    )
    config = _config("run-sectioned")

    state = cast(
        dict[str, Any],
        graph.invoke(
            {"run_id": "run-sectioned", "document": document.model_dump(mode="json")}, config
        ),
    )

    assert len(state["evidence"]) == 2
    assert [span["evidence_id"] for span in state["evidence"]] == [
        spans[0].evidence_id,
        spans[1].evidence_id,
    ]
    assert state["evidence"][1]["quote"].startswith("2. Approval")
    assert state["validation_errors"] == []
    assert state["__interrupt__"]


def _policy_citing(evidence_id: str) -> PolicyExtraction:
    return PolicyExtraction(
        policies=(
            PolicyCandidate(
                policy_id="policy-approval",
                title="Approval requirement",
                statement="A reviewer must approve a high-impact change.",
                modality=PolicyModality.REQUIRED,
                actor="reviewer",
                action="approve the change",
                evidence_refs=(evidence_id,),
                confidence=0.9,
            ),
        )
    )


def test_a_span_that_no_longer_quotes_the_document_blocks_the_policy_gate() -> None:
    """A checkpoint whose document and spans drifted apart must not reach a reviewer."""

    document = _sectioned_document()
    spans = segment_document(document)
    # Same length, so the span still satisfies its own contract and the document
    # comparison is what has to catch it.
    tampered = spans[1].model_copy(update={"quote": "X" + spans[1].quote[1:]})
    gateway = ScriptedModelGateway(
        {ModelTask.POLICY_EXTRACTION: [_policy_citing(spans[1].evidence_id)]}
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
    )
    config = _config("run-tampered")
    graph.update_state(
        config,
        {
            "run_id": "run-tampered",
            "document": document.model_dump(mode="json"),
            "policies": [_policy_citing(spans[1].evidence_id).policies[0].model_dump(mode="json")],
            "evidence": [spans[0].model_dump(mode="json"), tampered.model_dump(mode="json")],
        },
        as_node="extract_policies",
    )

    final = cast(dict[str, Any], graph.invoke(None, config))

    assert final["status"] == "policy_validation_failed"
    assert final["validation_errors"] == [
        f"evidence {tampered.evidence_id} does not quote "
        f"[{tampered.char_start}, {tampered.char_end}) of the document"
    ]
    assert "__interrupt__" not in final


def test_a_batch_is_retried_for_the_policies_the_model_left_out() -> None:
    document = synthetic_document("retried-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("retried-sop", 3)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [
                _rules_for(extraction.policies[:1], evidence_id),
                _rules_for(extraction.policies[1:], evidence_id),
            ],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=3,
    )
    config = _config("run-retried")

    graph.invoke({"run_id": "run-retried", "document": document.model_dump(mode="json")}, config)
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )

    first, retry = gateway.prompts[ModelTask.GUARDRAIL_COMPILATION]
    assert "(3 to compile)" in first
    # The retry carries only what is still missing.
    assert "(2 to compile)" in retry
    assert "policy-00" not in retry
    assert "policy-01" in retry
    assert "policy-02" in retry

    assert len(final["guardrails"]) == 3
    assert final["validation_errors"] == []
    assert final["status"] == "released"
    # extraction + two compilation attempts + assessment
    assert len(final["model_runs"]) == 4


def test_a_retry_that_re_answers_a_covered_policy_does_not_duplicate_it() -> None:
    """The retry asked only about the missing policies; anything else is unsolicited."""

    document = synthetic_document("repeating-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("repeating-sop", 2)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [
                _rules_for(extraction.policies[:1], evidence_id),
                # Answers the missing policy and repeats the one already covered.
                _rules_for(extraction.policies, evidence_id),
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
    config = _config("run-repeating")

    graph.invoke({"run_id": "run-repeating", "document": document.model_dump(mode="json")}, config)
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )

    assert [rule["policy_id"] for rule in final["guardrails"]] == ["policy-00", "policy-01"]
    assert final["validation_errors"] == []
    assert final["status"] == "released"


def test_a_complete_first_answer_is_not_retried() -> None:
    document = synthetic_document("complete-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("complete-sop", 2)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [_rules_for(extraction.policies, evidence_id)],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=2,
    )
    config = _config("run-complete")

    graph.invoke({"run_id": "run-complete", "document": document.model_dump(mode="json")}, config)
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)

    assert len(gateway.prompts[ModelTask.GUARDRAIL_COMPILATION]) == 1


def test_a_max_attempts_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="compilation_max_attempts must be at least 1"):
        WorkflowDependencies(
            model_gateway=ScriptedModelGateway({}),
            feedback_store=InMemoryFeedbackStore(),
            compilation_max_attempts=0,
        )


def test_assessment_batches_carry_only_the_context_their_rules_cite() -> None:
    document = synthetic_document("assessed-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("assessed-sop", 4)
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [_rules_for(extraction.policies, evidence_id)],
            ModelTask.GUARDRAIL_ASSESSMENT: [
                LLMAssessment(verdict=AssessmentVerdict.PASS, summary="First half holds."),
                LLMAssessment(
                    verdict=AssessmentVerdict.ABSTAIN,
                    summary="Second half is thin.",
                    uncertainty="The evidence does not define the threshold.",
                ),
            ],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=4,
        assessment_batch_size=2,
    )
    config = _config("run-assessed")

    graph.invoke({"run_id": "run-assessed", "document": document.model_dump(mode="json")}, config)
    state = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )

    first, second = gateway.prompts[ModelTask.GUARDRAIL_ASSESSMENT]
    assert "policy-00" in first
    assert "policy-01" in first
    assert "policy-02" not in first
    assert "policy-02" in second
    assert "policy-03" in second
    assert "policy-00" not in second
    # Every batch still receives the span its rules cite.
    assert evidence_id in first
    assert evidence_id in second

    # The abstaining batch decides the merged verdict.
    assert state["llm_assessment"]["verdict"] == "abstain"
    assert state["llm_assessment"]["summary"] == "First half holds. Second half is thin."
    assert state["llm_assessment"]["uncertainty"] == "The evidence does not define the threshold."
    # extraction + one compilation + two assessment calls
    assert len(state["model_runs"]) == 4


def test_an_assessment_batch_size_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="assessment_batch_size must be at least 1"):
        WorkflowDependencies(
            model_gateway=ScriptedModelGateway({}),
            feedback_store=InMemoryFeedbackStore(),
            assessment_batch_size=0,
        )


class _PromptDrivenGateway:
    """Answer from the prompt rather than a queue, so concurrent calls stay sound."""

    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id
        self._lock = Lock()
        self.calls: dict[ModelTask, int] = defaultdict(int)
        self.peak_in_flight = 0
        self._in_flight = 0

    def invoke(
        self, *, task: ModelTask, prompt: str, response_model: type[Any]
    ) -> ModelResult[Any]:
        with self._lock:
            self.calls[task] += 1
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        sleep(0.02)  # hold the slot so overlap is observable
        try:
            asked = tuple(
                PolicyCandidate(
                    policy_id=policy_id,
                    title="Control",
                    statement="A reviewer must approve the step.",
                    modality=PolicyModality.REQUIRED,
                    actor="reviewer",
                    action=f"approve {policy_id}",
                    evidence_refs=(self.evidence_id,),
                    confidence=0.9,
                )
                for policy_id in sorted(set(findall(r'"policy_id": "([^"]+)"', prompt)))
            )
            if task is ModelTask.GUARDRAIL_COMPILATION:
                value: Any = _rules_for(asked, self.evidence_id)
            else:
                value = LLMAssessment(
                    verdict=AssessmentVerdict.PASS,
                    summary=f"Reviewed {len(asked)} rules.",
                )
            return ModelResult(
                output=response_model.model_validate(value.model_dump(mode="json")),
                metadata=ModelMetadata(provider="prompt-driven", model="fixture-v1"),
            )
        finally:
            with self._lock:
                self._in_flight -= 1


def _parallel_state(concurrency: int) -> tuple[dict[str, Any], _PromptDrivenGateway]:
    document = synthetic_document("parallel-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("parallel-sop", 8)
    gateway = _PromptDrivenGateway(evidence_id)
    gateway_with_extraction = ScriptedModelGateway({ModelTask.POLICY_EXTRACTION: [extraction]})

    class _Routing:
        def invoke(self, *, task: ModelTask, prompt: str, response_model: type[Any]) -> Any:
            if task is ModelTask.POLICY_EXTRACTION:
                return gateway_with_extraction.invoke(
                    task=task, prompt=prompt, response_model=response_model
                )
            return gateway.invoke(task=task, prompt=prompt, response_model=response_model)

    graph = build_workflow(
        model_gateway=_Routing(),
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
        compilation_batch_size=2,
        assessment_batch_size=2,
        max_concurrent_batches=concurrency,
    )
    config = _config(f"run-parallel-{concurrency}")
    graph.invoke(
        {"run_id": f"run-parallel-{concurrency}", "document": document.model_dump(mode="json")},
        config,
    )
    graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )
    return final, gateway


def test_concurrent_batches_produce_the_same_ordered_result_as_sequential_ones() -> None:
    sequential, serial_gateway = _parallel_state(1)
    concurrent, parallel_gateway = _parallel_state(4)

    assert serial_gateway.peak_in_flight == 1
    assert parallel_gateway.peak_in_flight > 1

    assert [rule["rule_id"] for rule in concurrent["guardrails"]] == [
        rule["rule_id"] for rule in sequential["guardrails"]
    ]
    assert concurrent["status"] == sequential["status"] == "released"
    assert concurrent["validation_errors"] == []
    assert serial_gateway.calls == parallel_gateway.calls


def test_a_concurrency_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_concurrent_batches must be at least 1"):
        WorkflowDependencies(
            model_gateway=ScriptedModelGateway({}),
            feedback_store=InMemoryFeedbackStore(),
            max_concurrent_batches=0,
        )


def test_a_rule_that_blocks_nothing_stops_the_release() -> None:
    """A rule can cite everything correctly and still enforce nothing."""

    document = synthetic_document("inert-sop")
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    extraction = _numbered_policies("inert-sop", 1)
    inert = GuardrailCompilation(
        rules=(
            GuardrailRule(
                rule_id="rule-inert",
                policy_id="policy-00",
                decision=GuardrailDecision.ALLOW,
                condition="the reviewer approved the step",
                rationale="The policy requires approval.",
                severity=Severity.MEDIUM,
                evidence_refs=(evidence_id,),
                test_cases=(
                    RuleTestCase(
                        name="approved step is allowed",
                        input_summary="step with a recorded approval",
                        expected_decision=GuardrailDecision.ALLOW,
                    ),
                ),
            ),
        )
    )
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [extraction],
            ModelTask.GUARDRAIL_COMPILATION: [inert],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
    )
    config = _config("run-inert")

    graph.invoke({"run_id": "run-inert", "document": document.model_dump(mode="json")}, config)
    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )

    assert final["status"] == "guardrail_validation_failed"
    assert final["validation_errors"] == [
        "rule rule-inert decides 'allow', so it blocks nothing",
        "rule rule-inert has no test case that denies or escalates, so nothing can trip it",
    ]
    assert "release" not in final
    assert gateway.prompts[ModelTask.GUARDRAIL_ASSESSMENT] == []


def _assessment_with_findings() -> LLMAssessment:
    return LLMAssessment(
        verdict=AssessmentVerdict.ABSTAIN,
        summary="The evidence is thin for two of the rules.",
        findings=(
            AssessmentFinding(
                code="COVERAGE_GAP",
                message="rule-require-review omits the approval record requirement.",
                severity=Severity.HIGH,
            ),
            AssessmentFinding(
                code="AMBIGUITY",
                message="'high-impact' is not defined by the cited evidence.",
                severity=Severity.MEDIUM,
            ),
        ),
        uncertainty="A domain reviewer must confirm the intended control.",
    )


def _run_to_release_gate(run_id: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    document = synthetic_document()
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [policy_extraction(document)],
            ModelTask.GUARDRAIL_COMPILATION: [guardrail_compilation(document)],
            ModelTask.GUARDRAIL_ASSESSMENT: [_assessment_with_findings()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
    )
    config = _config(run_id)
    graph.invoke({"run_id": run_id, "document": document.model_dump(mode="json")}, config)
    state = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )
    return graph, config, cast(dict[str, Any], state["__interrupt__"][0].value)


def test_the_release_gate_hands_the_reviewer_the_findings() -> None:
    """The reviewer used to get the verdict alone, with the findings left in state."""

    _, _, request = _run_to_release_gate("run-gate-findings")

    assert request["gate"] == "release"
    assert request["assessment"]["verdict"] == "abstain"
    assert [finding["code"] for finding in request["assessment"]["findings"]] == [
        "COVERAGE_GAP",
        "AMBIGUITY",
    ]
    assert request["assessment"]["uncertainty"]


def test_the_release_summary_counts_the_findings_by_severity() -> None:
    _, _, request = _run_to_release_gate("run-gate-summary")

    assert request["summary"] == (
        "Final release review of 1 rule. Advisory verdict: abstain; 2 findings (1 high, 1 medium)."
    )


def test_a_release_records_the_advisory_verdict_it_was_approved_over() -> None:
    """The published artifact must not hide an approval made over an abstention."""

    graph, config, _ = _run_to_release_gate("run-gate-verdict")

    final = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.RELEASE, ReviewVerdict.APPROVE)), config),
    )

    assert final["status"] == "released"
    assert final["release"]["advisory_verdict"] == "abstain"


def test_a_clean_assessment_says_so_in_the_summary() -> None:
    document = synthetic_document()
    gateway = ScriptedModelGateway(
        {
            ModelTask.POLICY_EXTRACTION: [policy_extraction(document)],
            ModelTask.GUARDRAIL_COMPILATION: [guardrail_compilation(document)],
            ModelTask.GUARDRAIL_ASSESSMENT: [passing_assessment()],
        }
    )
    graph = build_workflow(
        model_gateway=gateway,
        feedback_store=InMemoryFeedbackStore(),
        checkpointer=InMemorySaver(),
    )
    config = _config("run-gate-clean")
    graph.invoke({"run_id": "run-gate-clean", "document": document.model_dump(mode="json")}, config)
    state = cast(
        dict[str, Any],
        graph.invoke(Command(resume=_decision(ReviewGate.POLICY, ReviewVerdict.APPROVE)), config),
    )

    assert "no findings" in state["__interrupt__"][0].value["summary"]
