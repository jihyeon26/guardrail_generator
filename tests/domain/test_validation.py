from sop_guardrail.domain.models import EvidenceSpan, PolicyCandidate
from sop_guardrail.domain.validation import (
    validate_guardrail_coverage,
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
    assert validate_guardrail_coverage(rules, policies) == ()


def test_unknown_evidence_is_reported() -> None:
    document = synthetic_document()
    evidence = (EvidenceSpan.from_document(document),)
    policies = policy_extraction(document, evidence_id="evidence-missing").policies

    errors = validate_policy_references(policies, evidence)

    assert errors == ("policy policy-review-approval has unknown evidence: ['evidence-missing']",)


def _extra_policy(policy_id: str, evidence_id: str) -> PolicyCandidate:
    return PolicyCandidate(
        policy_id=policy_id,
        title="Retention requirement",
        statement="Approval records must be retained.",
        actor="operator",
        action="retain the approval record",
        evidence_refs=(evidence_id,),
        confidence=0.9,
    )


def test_policies_without_a_rule_are_reported() -> None:
    """A compilation that drops policies must fail before a human sees it."""

    document = synthetic_document()
    evidence = (EvidenceSpan.from_document(document),)
    policies = (
        *policy_extraction(document).policies,
        _extra_policy("policy-retention", evidence[0].evidence_id),
        _extra_policy("policy-audit-trail", evidence[0].evidence_id),
    )
    rules = guardrail_compilation(document).rules

    assert validate_guardrail_references(rules, policies, evidence) == ()
    assert validate_guardrail_coverage(rules, policies) == (
        "policies without a guardrail rule: ['policy-audit-trail', 'policy-retention']",
    )


def test_coverage_ignores_rules_that_point_at_an_unknown_policy() -> None:
    """The reference check owns unknown ids; coverage only reports missing rules."""

    document = synthetic_document()
    policies = policy_extraction(document).policies
    rules = tuple(
        rule.model_copy(update={"policy_id": "policy-does-not-exist"})
        for rule in guardrail_compilation(document).rules
    )

    assert validate_guardrail_coverage(rules, policies) == (
        "policies without a guardrail rule: ['policy-review-approval']",
    )
