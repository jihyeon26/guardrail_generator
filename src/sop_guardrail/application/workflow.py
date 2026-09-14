"""Explicit LangGraph workflow with deterministic, LLM, and human gates."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from sop_guardrail.application.prompts import (
    assessment_prompt,
    guardrail_compilation_prompt,
    policy_extraction_prompt,
)
from sop_guardrail.application.state import WorkflowState
from sop_guardrail.domain.assessment import merge_assessments
from sop_guardrail.domain.models import (
    EvidenceSpan,
    FeedbackCard,
    FeedbackStage,
    GuardrailCompilation,
    GuardrailRelease,
    GuardrailRule,
    LLMAssessment,
    ModelMetadata,
    ModelResult,
    ModelTask,
    PolicyCandidate,
    PolicyExtraction,
    ReviewDecision,
    ReviewGate,
    ReviewRequest,
    ReviewVerdict,
    SopDocument,
)
from sop_guardrail.domain.ports import FeedbackStore, StructuredModelGateway
from sop_guardrail.domain.segmentation import DEFAULT_MAX_SPAN_CHARS, segment_document
from sop_guardrail.domain.validation import (
    validate_evidence_spans,
    validate_guardrail_coverage,
    validate_guardrail_enforcement,
    validate_guardrail_references,
    validate_policy_references,
)

DEFAULT_COMPILATION_BATCH_SIZE = 5
DEFAULT_COMPILATION_MAX_ATTEMPTS = 3
DEFAULT_ASSESSMENT_BATCH_SIZE = 5
# Sequential by default: a hosted provider's rate limit should not be hit because a
# workflow quietly fanned out. A caller that knows its provider opts in.
DEFAULT_MAX_CONCURRENT_BATCHES = 1


@dataclass(frozen=True)
class WorkflowDependencies:
    model_gateway: StructuredModelGateway
    feedback_store: FeedbackStore
    compilation_batch_size: int = DEFAULT_COMPILATION_BATCH_SIZE
    compilation_max_attempts: int = DEFAULT_COMPILATION_MAX_ATTEMPTS
    assessment_batch_size: int = DEFAULT_ASSESSMENT_BATCH_SIZE
    max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_BATCHES
    max_span_chars: int = DEFAULT_MAX_SPAN_CHARS

    def __post_init__(self) -> None:
        if self.compilation_batch_size < 1:
            raise ValueError("compilation_batch_size must be at least 1")
        if self.compilation_max_attempts < 1:
            raise ValueError("compilation_max_attempts must be at least 1")
        if self.assessment_batch_size < 1:
            raise ValueError("assessment_batch_size must be at least 1")
        if self.max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be at least 1")
        if self.max_span_chars < 1:
            raise ValueError("max_span_chars must be at least 1")


def _batches[ItemT](items: tuple[ItemT, ...], size: int) -> tuple[tuple[ItemT, ...], ...]:
    return tuple(tuple(items[start : start + size]) for start in range(0, len(items), size))


def _map_batches[BatchT, ResultT](
    batches: tuple[BatchT, ...],
    worker: Callable[[BatchT], ResultT],
    max_concurrent: int,
) -> tuple[ResultT, ...]:
    """Run independent batches, returning results in batch order either way.

    Batches never read each other's output, so the only thing concurrency may not
    change is the order results are merged in; that is why the results are always
    collected in the order the batches were cut.
    """

    if max_concurrent == 1 or len(batches) < 2:
        return tuple(worker(batch) for batch in batches)
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(batches))) as pool:
        return tuple(pool.map(worker, batches))


def _document(state: WorkflowState) -> SopDocument:
    return SopDocument.model_validate(state["document"])


def _evidence(state: WorkflowState) -> tuple[EvidenceSpan, ...]:
    return tuple(EvidenceSpan.model_validate(item) for item in state.get("evidence", []))


def _policies(state: WorkflowState) -> tuple[PolicyCandidate, ...]:
    return tuple(PolicyCandidate.model_validate(item) for item in state.get("policies", []))


def _guardrails(state: WorkflowState) -> tuple[GuardrailRule, ...]:
    return tuple(GuardrailRule.model_validate(item) for item in state.get("guardrails", []))


def _review(
    state: WorkflowState, key: Literal["policy_review", "release_review"]
) -> ReviewDecision:
    return ReviewDecision.model_validate(state[key])


def _append_model_run(state: WorkflowState, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("model_runs", []), metadata]


def _ingest_node(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Cut the SOP into section-level spans so a citation names a passage, not a file."""

    document = _document(state)
    evidence = segment_document(document, max_span_chars=dependencies.max_span_chars)
    return {
        "evidence": [span.model_dump(mode="json") for span in evidence],
        "validation_errors": [],
        "status": "ingested",
    }


def _extract_policies_node(
    state: WorkflowState, dependencies: WorkflowDependencies
) -> WorkflowState:
    document = _document(state)
    evidence = _evidence(state)
    feedback = dependencies.feedback_store.list_active(FeedbackStage.POLICY_EXTRACTION)
    result = dependencies.model_gateway.invoke(
        task=ModelTask.POLICY_EXTRACTION,
        prompt=policy_extraction_prompt(document=document, evidence=evidence, feedback=feedback),
        response_model=PolicyExtraction,
    )
    return {
        "policies": [item.model_dump(mode="json") for item in result.output.policies],
        "model_runs": _append_model_run(state, result.metadata.model_dump(mode="json")),
        "status": "policies_extracted",
    }


def _validate_policies_node(state: WorkflowState) -> WorkflowState:
    evidence = _evidence(state)
    errors = (
        *validate_evidence_spans(evidence, _document(state)),
        *validate_policy_references(_policies(state), evidence),
    )
    return {
        "validation_errors": list(errors),
        "status": "policy_validation_failed" if errors else "policies_validated",
    }


def _route_policy_validation(state: WorkflowState) -> Literal["review", "end"]:
    return "end" if state.get("validation_errors") else "review"


def _policy_review_node(state: WorkflowState) -> WorkflowState:
    request = ReviewRequest(
        run_id=state["run_id"],
        gate=ReviewGate.POLICY,
        item_ids=tuple(item.policy_id for item in _policies(state)),
        summary="Review extracted policies and their cited SOP evidence.",
    )
    decision = ReviewDecision.model_validate(interrupt(request.model_dump(mode="json")))
    if decision.gate is not ReviewGate.POLICY:
        raise ValueError("policy gate received a decision for a different gate")
    return {
        "policy_review": decision.model_dump(mode="json"),
        "status": f"policy_{decision.verdict.value}",
    }


def _route_policy_review(state: WorkflowState) -> Literal["compile", "feedback"]:
    decision = _review(state, "policy_review")
    return "compile" if decision.verdict is ReviewVerdict.APPROVE else "feedback"


def _compile_guardrails_node(
    state: WorkflowState, dependencies: WorkflowDependencies
) -> WorkflowState:
    """Compile in batches so a long SOP cannot be answered with a single rule.

    One call per batch keeps each prompt small enough that the model is asked for a
    handful of rules rather than an open-ended set. Coverage is still proved by
    deterministic validation, never by the batching itself.
    """

    policies = _policies(state)
    feedback = dependencies.feedback_store.list_active(FeedbackStage.GUARDRAIL_COMPILATION)
    rules: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []

    results = _map_batches(
        _batches(policies, dependencies.compilation_batch_size),
        lambda batch: _compile_batch(batch, dependencies, feedback),
        dependencies.max_concurrent_batches,
    )
    for batch_rules, batch_metadata in results:
        rules.extend(rule.model_dump(mode="json") for rule in batch_rules)
        metadata.extend(item.model_dump(mode="json") for item in batch_metadata)

    return {
        "guardrails": rules,
        "model_runs": [*state.get("model_runs", []), *metadata],
        "status": "guardrails_compiled",
    }


def _compile_batch(
    batch: tuple[PolicyCandidate, ...],
    dependencies: WorkflowDependencies,
    feedback: tuple[FeedbackCard, ...],
) -> tuple[tuple[GuardrailRule, ...], tuple[ModelMetadata, ...]]:
    """Ask for the batch, then re-ask for whatever the model left out.

    A model that answers five policies with one rule is the observed failure mode, so
    each retry carries only the policies still missing. The list shrinks every attempt
    until it is a single policy, which is the request a small model does answer.

    Only rules for policies still awaited are kept: the retry asked about those alone,
    so a rule for an already-answered policy would be an unsolicited duplicate.
    """

    rules: list[GuardrailRule] = []
    metadata: list[ModelMetadata] = []
    pending = batch

    for _ in range(dependencies.compilation_max_attempts):
        result = dependencies.model_gateway.invoke(
            task=ModelTask.GUARDRAIL_COMPILATION,
            prompt=guardrail_compilation_prompt(policies=pending, feedback=feedback),
            response_model=GuardrailCompilation,
        )
        metadata.append(result.metadata)
        awaited = {policy.policy_id for policy in pending}
        rules.extend(rule for rule in result.output.rules if rule.policy_id in awaited)

        covered = {rule.policy_id for rule in rules}
        pending = tuple(policy for policy in pending if policy.policy_id not in covered)
        if not pending:
            break

    return tuple(rules), tuple(metadata)


def _validate_guardrails_node(state: WorkflowState) -> WorkflowState:
    guardrails = _guardrails(state)
    policies = _policies(state)
    errors = (
        *validate_guardrail_references(guardrails, policies, _evidence(state)),
        *validate_guardrail_coverage(guardrails, policies),
        *validate_guardrail_enforcement(guardrails),
    )
    return {
        "validation_errors": list(errors),
        "status": "guardrail_validation_failed" if errors else "guardrails_validated",
    }


def _route_guardrail_validation(state: WorkflowState) -> Literal["assess", "end"]:
    return "end" if state.get("validation_errors") else "assess"


def _llm_assessment_node(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Review the rules in batches, each carrying only the context those rules cite.

    Sending every policy, rule, and span in one call made the prompt grow with the
    SOP until a local model could not finish it. A batch needs the rules under review
    and nothing else, so the context stays flat as the document gets longer.
    """

    policies = _policies(state)
    evidence = _evidence(state)
    feedback = dependencies.feedback_store.list_active(FeedbackStage.LLM_ASSESSMENT)
    parts: list[LLMAssessment] = []
    metadata: list[dict[str, Any]] = []

    def assess(batch: tuple[GuardrailRule, ...]) -> ModelResult[LLMAssessment]:
        cited_policies = _policies_behind(batch, policies)
        return dependencies.model_gateway.invoke(
            task=ModelTask.GUARDRAIL_ASSESSMENT,
            prompt=assessment_prompt(
                policies=cited_policies,
                guardrails=batch,
                evidence=_evidence_behind(batch, cited_policies, evidence),
                feedback=feedback,
            ),
            response_model=LLMAssessment,
        )

    for result in _map_batches(
        _batches(_guardrails(state), dependencies.assessment_batch_size),
        assess,
        dependencies.max_concurrent_batches,
    ):
        parts.append(result.output)
        metadata.append(result.metadata.model_dump(mode="json"))

    return {
        "llm_assessment": merge_assessments(tuple(parts)).model_dump(mode="json"),
        "model_runs": [*state.get("model_runs", []), *metadata],
        "status": "llm_assessed",
    }


def _policies_behind(
    rules: tuple[GuardrailRule, ...], policies: tuple[PolicyCandidate, ...]
) -> tuple[PolicyCandidate, ...]:
    wanted = {rule.policy_id for rule in rules}
    return tuple(policy for policy in policies if policy.policy_id in wanted)


def _evidence_behind(
    rules: tuple[GuardrailRule, ...],
    policies: tuple[PolicyCandidate, ...],
    evidence: tuple[EvidenceSpan, ...],
) -> tuple[EvidenceSpan, ...]:
    wanted = {ref for rule in rules for ref in rule.evidence_refs}
    wanted |= {ref for policy in policies for ref in policy.evidence_refs}
    return tuple(span for span in evidence if span.evidence_id in wanted)


def _release_review_node(state: WorkflowState) -> WorkflowState:
    assessment = LLMAssessment.model_validate(state["llm_assessment"])
    request = ReviewRequest(
        run_id=state["run_id"],
        gate=ReviewGate.RELEASE,
        item_ids=tuple(item.rule_id for item in _guardrails(state)),
        summary=f"Final release review. Advisory LLM verdict: {assessment.verdict.value}.",
    )
    decision = ReviewDecision.model_validate(interrupt(request.model_dump(mode="json")))
    if decision.gate is not ReviewGate.RELEASE:
        raise ValueError("release gate received a decision for a different gate")
    return {
        "release_review": decision.model_dump(mode="json"),
        "status": f"release_{decision.verdict.value}",
    }


def _route_release_review(state: WorkflowState) -> Literal["publish", "feedback"]:
    decision = _review(state, "release_review")
    return "publish" if decision.verdict is ReviewVerdict.APPROVE else "feedback"


def _record_feedback_node(
    state: WorkflowState, dependencies: WorkflowDependencies
) -> WorkflowState:
    if "release_review" in state:
        decision = _review(state, "release_review")
        stage = FeedbackStage.LLM_ASSESSMENT
    else:
        decision = _review(state, "policy_review")
        stage = FeedbackStage.POLICY_EXTRACTION

    lesson = decision.reusable_lesson or decision.comment
    digest = sha256(f"{state['run_id']}:{decision.gate.value}:{lesson}".encode()).hexdigest()[:12]
    feedback = FeedbackCard(
        feedback_id=f"feedback-{digest}",
        stage=stage,
        lesson=lesson,
        source_run_id=state["run_id"],
        source_gate=decision.gate,
    )
    dependencies.feedback_store.add(feedback)
    return {
        "feedback_ids": [*state.get("feedback_ids", []), feedback.feedback_id],
        "status": "feedback_pending",
    }


def _publish_node(state: WorkflowState) -> WorkflowState:
    decision = _review(state, "release_review")
    if decision.verdict is not ReviewVerdict.APPROVE:
        raise ValueError("only an approved release decision can publish")
    release = GuardrailRelease(
        release_id=f"release-{state['run_id']}",
        version="0.1.0",
        source_document_id=_document(state).document_id,
        rules=_guardrails(state),
        approved_by=decision.reviewer,
    )
    return {"release": release.model_dump(mode="json"), "status": "released"}


def build_workflow(
    *,
    model_gateway: StructuredModelGateway,
    feedback_store: FeedbackStore,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    compilation_batch_size: int = DEFAULT_COMPILATION_BATCH_SIZE,
    compilation_max_attempts: int = DEFAULT_COMPILATION_MAX_ATTEMPTS,
    assessment_batch_size: int = DEFAULT_ASSESSMENT_BATCH_SIZE,
    max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_BATCHES,
    max_span_chars: int = DEFAULT_MAX_SPAN_CHARS,
) -> Any:
    """Compile a workflow whose external dependencies are closed over by thin nodes."""

    dependencies = WorkflowDependencies(
        model_gateway=model_gateway,
        feedback_store=feedback_store,
        compilation_batch_size=compilation_batch_size,
        compilation_max_attempts=compilation_max_attempts,
        assessment_batch_size=assessment_batch_size,
        max_concurrent_batches=max_concurrent_batches,
        max_span_chars=max_span_chars,
    )
    graph = StateGraph(WorkflowState)

    graph.add_node("ingest", lambda state: _ingest_node(state, dependencies))
    graph.add_node("extract_policies", lambda state: _extract_policies_node(state, dependencies))
    graph.add_node("validate_policies", _validate_policies_node)
    graph.add_node("policy_review", _policy_review_node)
    graph.add_node(
        "compile_guardrails", lambda state: _compile_guardrails_node(state, dependencies)
    )
    graph.add_node("validate_guardrails", _validate_guardrails_node)
    graph.add_node("llm_assessment", lambda state: _llm_assessment_node(state, dependencies))
    graph.add_node("release_review", _release_review_node)
    graph.add_node("record_feedback", lambda state: _record_feedback_node(state, dependencies))
    graph.add_node("publish", _publish_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract_policies")
    graph.add_edge("extract_policies", "validate_policies")
    graph.add_conditional_edges(
        "validate_policies",
        _route_policy_validation,
        {"review": "policy_review", "end": END},
    )
    graph.add_conditional_edges(
        "policy_review",
        _route_policy_review,
        {"compile": "compile_guardrails", "feedback": "record_feedback"},
    )
    graph.add_edge("compile_guardrails", "validate_guardrails")
    graph.add_conditional_edges(
        "validate_guardrails",
        _route_guardrail_validation,
        {"assess": "llm_assessment", "end": END},
    )
    graph.add_edge("llm_assessment", "release_review")
    graph.add_conditional_edges(
        "release_review",
        _route_release_review,
        {"publish": "publish", "feedback": "record_feedback"},
    )
    graph.add_edge("record_feedback", END)
    graph.add_edge("publish", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
