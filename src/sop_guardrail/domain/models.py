"""Typed contracts shared across workflow boundaries."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base configuration for public workflow contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ReviewGate(StrEnum):
    POLICY = "policy"
    RELEASE = "release"


class ReviewVerdict(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class FeedbackStage(StrEnum):
    POLICY_EXTRACTION = "policy_extraction"
    GUARDRAIL_COMPILATION = "guardrail_compilation"
    LLM_ASSESSMENT = "llm_assessment"


class FeedbackStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class PolicyModality(StrEnum):
    """What the SOP obliges of the actor.

    A permission or an efficiency preference fits neither member, so requiring this
    field makes the extractor commit rather than turn every sentence into a policy.
    """

    REQUIRED = "required"
    PROHIBITED = "prohibited"


class GuardrailDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AssessmentVerdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ModelTask(StrEnum):
    POLICY_EXTRACTION = "policy_extraction"
    GUARDRAIL_COMPILATION = "guardrail_compilation"
    GUARDRAIL_ASSESSMENT = "guardrail_assessment"


class SopDocument(ContractModel):
    document_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @classmethod
    def from_text(cls, *, document_id: str, source_name: str, text: str) -> Self:
        normalized = text.strip()
        return cls(
            document_id=document_id,
            source_name=source_name,
            text=normalized,
            sha256=sha256(normalized.encode("utf-8")).hexdigest(),
        )


class EvidenceSpan(ContractModel):
    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    quote: str = Field(min_length=1)
    quote_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if len(self.quote) != self.char_end - self.char_start:
            raise ValueError("quote length must match the character range")
        return self

    @classmethod
    def from_slice(
        cls, document: SopDocument, *, char_start: int, char_end: int, index: int
    ) -> Self:
        """Cut one span out of the document, keeping offsets and quote in step.

        Surrounding whitespace is trimmed off the offsets before the quote is taken,
        because contract models strip strings and a padded quote would no longer
        match the character range it claims.
        """

        start = char_start
        end = min(char_end, len(document.text))
        while start < end and document.text[start].isspace():
            start += 1
        while end > start and document.text[end - 1].isspace():
            end -= 1
        if start >= end:
            raise ValueError(f"slice [{char_start}, {char_end}) of {document.document_id} is empty")

        quote = document.text[start:end]
        return cls(
            evidence_id=f"evidence-{document.sha256[:8]}-{index:03d}",
            document_id=document.document_id,
            char_start=start,
            char_end=end,
            quote=quote,
            quote_sha256=sha256(quote.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_document(cls, document: SopDocument) -> Self:
        """The degenerate single-span case: the whole document is the evidence."""

        return cls.from_slice(document, char_start=0, char_end=len(document.text), index=0)


class PolicyCandidate(ContractModel):
    policy_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    modality: PolicyModality
    actor: str = Field(min_length=1)
    action: str = Field(min_length=1)
    condition: str | None = None
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class PolicyExtraction(ContractModel):
    policies: tuple[PolicyCandidate, ...] = Field(min_length=1)


class RuleTestCase(ContractModel):
    name: str = Field(min_length=1)
    input_summary: str = Field(min_length=1)
    expected_decision: GuardrailDecision


class GuardrailRule(ContractModel):
    rule_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    decision: GuardrailDecision
    condition: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    severity: Severity
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    test_cases: tuple[RuleTestCase, ...] = Field(min_length=1)


class GuardrailCompilation(ContractModel):
    rules: tuple[GuardrailRule, ...] = Field(min_length=1)


class AssessmentFinding(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Severity
    evidence_refs: tuple[str, ...] = ()


class LLMAssessment(ContractModel):
    verdict: AssessmentVerdict
    summary: str = Field(min_length=1)
    findings: tuple[AssessmentFinding, ...] = ()
    uncertainty: str | None = None


class ReviewRequest(ContractModel):
    """What a human is handed at a gate.

    The advisory assessment exists only to inform this decision, so it travels with
    the request. A reviewer who is shown a verdict but not the findings behind it is
    approving blind.
    """

    run_id: str = Field(min_length=1)
    gate: ReviewGate
    item_ids: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1)
    assessment: LLMAssessment | None = None


class ReviewDecision(ContractModel):
    gate: ReviewGate
    verdict: ReviewVerdict
    reviewer: str = Field(min_length=1)
    comment: str = Field(min_length=1)
    reusable_lesson: str | None = None


class FeedbackCard(ContractModel):
    feedback_id: str = Field(min_length=1)
    stage: FeedbackStage
    lesson: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    source_gate: ReviewGate
    status: FeedbackStatus = FeedbackStatus.PENDING
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def active_feedback_requires_approver(self) -> Self:
        if self.status is FeedbackStatus.ACTIVE and not self.approved_by:
            raise ValueError("active feedback requires approved_by")
        return self


class ModelMetadata(ContractModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class ModelResult[ModelOutputT: ContractModel](ContractModel):
    output: ModelOutputT
    metadata: ModelMetadata


class GuardrailRelease(ContractModel):
    release_id: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_document_id: str = Field(min_length=1)
    rules: tuple[GuardrailRule, ...] = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    # The advisory verdict this release was approved over, so the artifact does not
    # hide that a human released it while the reviewer abstained or objected.
    advisory_verdict: AssessmentVerdict
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
