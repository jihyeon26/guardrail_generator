import pytest

from sop_guardrail.application.prompts import (
    assessment_prompt,
    guardrail_compilation_prompt,
    policy_extraction_prompt,
)
from sop_guardrail.domain.models import (
    EvidenceSpan,
    GuardrailCompilation,
    LLMAssessment,
    ModelTask,
    PolicyExtraction,
)
from sop_guardrail.infrastructure.providers.demo import DemoModelGateway
from tests.helpers import synthetic_document


def test_demo_gateway_supports_the_full_structured_sequence() -> None:
    document = synthetic_document()
    evidence = (EvidenceSpan.from_document(document),)
    gateway = DemoModelGateway()

    extraction = gateway.invoke(
        task=ModelTask.POLICY_EXTRACTION,
        prompt=policy_extraction_prompt(document=document, evidence=evidence, feedback=()),
        response_model=PolicyExtraction,
    )
    compilation = gateway.invoke(
        task=ModelTask.GUARDRAIL_COMPILATION,
        prompt=guardrail_compilation_prompt(policies=extraction.output.policies, feedback=()),
        response_model=GuardrailCompilation,
    )
    assessment = gateway.invoke(
        task=ModelTask.GUARDRAIL_ASSESSMENT,
        prompt=assessment_prompt(
            policies=extraction.output.policies,
            guardrails=compilation.output.rules,
            evidence=evidence,
            feedback=(),
        ),
        response_model=LLMAssessment,
    )

    assert extraction.output.policies[0].evidence_refs == (evidence[0].evidence_id,)
    assert compilation.output.rules[0].policy_id == extraction.output.policies[0].policy_id
    assert assessment.output.uncertainty is not None


def test_demo_gateway_rejects_prompt_without_evidence_id() -> None:
    gateway = DemoModelGateway()

    with pytest.raises(ValueError, match="evidence_id"):
        gateway.invoke(
            task=ModelTask.POLICY_EXTRACTION,
            prompt="No evidence inventory is present.",
            response_model=PolicyExtraction,
        )
