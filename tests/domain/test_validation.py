from sop_guardrail.domain.models import EvidenceSpan
from sop_guardrail.domain.validation import (
    validate_guardrail_references,
    validate_policy_references,
)
from tests.helpers import guardrail_compilation, policy_extraction, synthetic_document


def test_valid_artifacts_have_no_reference_errors() -> None:
    document = synthetic_document()
    evidence = (EvidenceSpan.from_document(document),)
    policies = policy_extraction(document).policies
    rules = guardrail_compilation(document).rules

    assert validate_policy_references(policies, evidence) == ()
    assert validate_guardrail_references(rules, policies, evidence) == ()


def test_unknown_evidence_is_reported() -> None:
    document = synthetic_document()
    evidence = (EvidenceSpan.from_document(document),)
    policies = policy_extraction(document, evidence_id="evidence-missing").policies

    errors = validate_policy_references(policies, evidence)

    assert errors == ("policy policy-review-approval has unknown evidence: ['evidence-missing']",)
