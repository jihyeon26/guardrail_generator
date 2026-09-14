"""Pure deterministic validation for model-proposed artifacts."""

from collections.abc import Iterable
from hashlib import sha256

from sop_guardrail.domain.models import (
    EvidenceSpan,
    GuardrailDecision,
    GuardrailRule,
    PolicyCandidate,
    SopDocument,
)

RESTRICTIVE_DECISIONS = frozenset({GuardrailDecision.DENY, GuardrailDecision.ESCALATE})


def _duplicate_ids(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate_evidence_spans(
    evidence: tuple[EvidenceSpan, ...], document: SopDocument
) -> tuple[str, ...]:
    """Prove every span still quotes the document it claims to cite.

    A citation is only worth as much as the span behind it. Offsets, quote text, and
    quote hash are checked against the stored document, so a span that drifted from
    the source — or was never in it — fails before a reviewer reads a policy.
    """

    errors: list[str] = []
    duplicates = _duplicate_ids(span.evidence_id for span in evidence)
    if duplicates:
        errors.append(f"duplicate evidence ids: {sorted(duplicates)}")

    for span in evidence:
        if span.document_id != document.document_id:
            errors.append(
                f"evidence {span.evidence_id} cites document {span.document_id}, "
                f"expected {document.document_id}"
            )
            continue
        if span.char_end > len(document.text):
            errors.append(
                f"evidence {span.evidence_id} ends at {span.char_end}, "
                f"past the {len(document.text)} character document"
            )
            continue
        if document.text[span.char_start : span.char_end] != span.quote:
            errors.append(
                f"evidence {span.evidence_id} does not quote "
                f"[{span.char_start}, {span.char_end}) of the document"
            )
            continue
        if sha256(span.quote.encode("utf-8")).hexdigest() != span.quote_sha256:
            errors.append(f"evidence {span.evidence_id} has a quote hash mismatch")

    return tuple(errors)


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


def validate_guardrail_enforcement(rules: tuple[GuardrailRule, ...]) -> tuple[str, ...]:
    """Report rules that cannot refuse anything.

    A rule that states the compliant case and allows it passes every other check — it
    cites a policy, cites evidence, and carries a test case — while blocking nothing.
    Two things have to hold. The rule must decide to deny or escalate, because that
    decision is what an engine acts on; and its own test cases must contain the
    violation, because that is the evidence it can fire at all.
    """

    errors: list[str] = []
    for rule in rules:
        if rule.decision not in RESTRICTIVE_DECISIONS:
            errors.append(
                f"rule {rule.rule_id} decides '{rule.decision.value}', so it blocks nothing"
            )
        if not any(case.expected_decision in RESTRICTIVE_DECISIONS for case in rule.test_cases):
            errors.append(
                f"rule {rule.rule_id} has no test case that denies or escalates, "
                "so nothing can trip it"
            )
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
