"""Synthetic fixtures shared by unit and graph tests."""

from sop_guardrail.domain.models import (
    AssessmentVerdict,
    EvidenceSpan,
    GuardrailCompilation,
    GuardrailDecision,
    GuardrailRule,
    LLMAssessment,
    PolicyCandidate,
    PolicyExtraction,
    RuleTestCase,
    Severity,
    SopDocument,
)


def synthetic_document(document_id: str = "synthetic-sop") -> SopDocument:
    return SopDocument.from_text(
        document_id=document_id,
        source_name="synthetic-review-procedure.txt",
        text=(
            "A reviewer must approve a high-impact change before the operator completes it. "
            "The approval record must identify the reviewer."
        ),
    )


def policy_extraction(document: SopDocument, *, evidence_id: str | None = None) -> PolicyExtraction:
    reference = evidence_id or EvidenceSpan.from_document(document).evidence_id
    return PolicyExtraction(
        policies=(
            PolicyCandidate(
                policy_id="policy-review-approval",
                title="High-impact change approval",
                statement="A reviewer must approve a high-impact change before completion.",
                actor="reviewer",
                action="approve the high-impact change",
                condition="before the operator completes it",
                evidence_refs=(reference,),
                confidence=0.95,
            ),
        )
    )


def guardrail_compilation(document: SopDocument) -> GuardrailCompilation:
    evidence_id = EvidenceSpan.from_document(document).evidence_id
    return GuardrailCompilation(
        rules=(
            GuardrailRule(
                rule_id="rule-require-review",
                policy_id="policy-review-approval",
                decision=GuardrailDecision.ESCALATE,
                condition="a high-impact change has no recorded reviewer approval",
                rationale="The SOP requires approval before completion.",
                severity=Severity.HIGH,
                evidence_refs=(evidence_id,),
                test_cases=(
                    RuleTestCase(
                        name="missing approval escalates",
                        input_summary="high-impact change without reviewer approval",
                        expected_decision=GuardrailDecision.ESCALATE,
                    ),
                ),
            ),
        )
    )


def passing_assessment() -> LLMAssessment:
    return LLMAssessment(
        verdict=AssessmentVerdict.PASS,
        summary="The proposed rule is supported by the cited evidence.",
        uncertainty="The definition of high-impact remains domain-specific.",
    )
