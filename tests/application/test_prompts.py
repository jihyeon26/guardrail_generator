from sop_guardrail.application.prompts import (
    assessment_prompt,
    guardrail_compilation_prompt,
    policy_extraction_prompt,
)
from sop_guardrail.domain.models import (
    EvidenceSpan,
    FeedbackCard,
    FeedbackStage,
    FeedbackStatus,
    ReviewGate,
)
from tests.helpers import (
    guardrail_compilation,
    passing_assessment,
    policy_extraction,
    synthetic_document,
)


def _active_feedback() -> FeedbackCard:
    return FeedbackCard(
        feedback_id="feedback-active",
        stage=FeedbackStage.POLICY_EXTRACTION,
        lesson="Keep the actor separate from the required action.",
        source_run_id="run-previous",
        source_gate=ReviewGate.POLICY,
        status=FeedbackStatus.ACTIVE,
        approved_by="curator@example.test",
    )


def test_policy_prompt_contains_evidence_and_approved_feedback() -> None:
    document = synthetic_document()
    evidence = EvidenceSpan.from_document(document)

    prompt = policy_extraction_prompt(
        document=document,
        evidence=(evidence,),
        feedback=(_active_feedback(),),
    )

    assert evidence.evidence_id in prompt
    assert "Keep the actor separate from the required action." in prompt


def test_compilation_and_assessment_prompts_state_their_boundaries() -> None:
    document = synthetic_document()
    policies = policy_extraction(document).policies
    guardrails = guardrail_compilation(document).rules
    evidence = (EvidenceSpan.from_document(document),)

    compilation = guardrail_compilation_prompt(policies=policies, feedback=())
    assessment = assessment_prompt(
        policies=policies,
        guardrails=guardrails,
        evidence=evidence,
        feedback=(),
    )

    assert "No approved lessons" in compilation
    assert "cannot approve a release" in assessment
    assert passing_assessment().summary not in assessment
