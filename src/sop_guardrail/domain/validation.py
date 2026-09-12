"""Pure deterministic validation for model-proposed artifacts."""

from collections.abc import Iterable

from sop_guardrail.domain.models import EvidenceSpan, GuardrailRule, PolicyCandidate


def _duplicate_ids(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate_policy_references(
    policies: tuple[PolicyCandidate, ...], evidence: tuple[EvidenceSpan, ...]
) -> tuple[str, ...]:
    errors: list[str] = []
    evidence_ids = {item.evidence_id for item in evidence}
    duplicates = _duplicate_ids(policy.policy_id for policy in policies)
    if duplicates:
        errors.append(f"duplicate policy ids: {sorted(duplicates)}")
    for policy in policies:
        unknown = set(policy.evidence_refs) - evidence_ids
        if unknown:
            errors.append(f"policy {policy.policy_id} has unknown evidence: {sorted(unknown)}")
    return tuple(errors)


def validate_guardrail_coverage(
    rules: tuple[GuardrailRule, ...], policies: tuple[PolicyCandidate, ...]
) -> tuple[str, ...]:
    """Report approved policies that no rule enforces.

    Reference validation only proves that each rule points at a real policy, so a
    compilation that silently drops most policies still passes it. A release that
    leaves approved policies unenforced is a coverage failure, not a model opinion.
    """

    covered = {rule.policy_id for rule in rules}
    uncovered = {policy.policy_id for policy in policies} - covered
    if not uncovered:
        return ()
    return (f"policies without a guardrail rule: {sorted(uncovered)}",)


def validate_guardrail_references(
    rules: tuple[GuardrailRule, ...],
    policies: tuple[PolicyCandidate, ...],
    evidence: tuple[EvidenceSpan, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    policy_ids = {item.policy_id for item in policies}
    evidence_ids = {item.evidence_id for item in evidence}
    duplicates = _duplicate_ids(rule.rule_id for rule in rules)
    if duplicates:
        errors.append(f"duplicate rule ids: {sorted(duplicates)}")
    for rule in rules:
        if rule.policy_id not in policy_ids:
            errors.append(f"rule {rule.rule_id} has unknown policy: {rule.policy_id}")
        unknown = set(rule.evidence_refs) - evidence_ids
        if unknown:
            errors.append(f"rule {rule.rule_id} has unknown evidence: {sorted(unknown)}")
    return tuple(errors)
