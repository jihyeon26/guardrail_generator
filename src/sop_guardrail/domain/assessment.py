"""Combine per-batch advisory assessments into the one a reviewer reads."""

from __future__ import annotations

from sop_guardrail.domain.models import AssessmentVerdict, LLMAssessment

# Most to least serious. A batch that rejects decides the run; a batch that cannot
# tell outranks one that passed, because an unexamined rule is not an approved rule.
VERDICT_PRECEDENCE = (
    AssessmentVerdict.REJECT,
    AssessmentVerdict.REVISE,
    AssessmentVerdict.ABSTAIN,
    AssessmentVerdict.PASS,
)


def merge_assessments(parts: tuple[LLMAssessment, ...]) -> LLMAssessment:
    """Fold batch assessments into one, keeping the most serious verdict.

    Splitting the review into batches must not let a clean batch soften a damning
    one, so the verdict is the worst any batch reported and every finding is kept.
    """

    if not parts:
        raise ValueError("cannot merge an empty set of assessments")
    if len(parts) == 1:
        return parts[0]

    verdict = next(
        candidate
        for candidate in VERDICT_PRECEDENCE
        if any(part.verdict is candidate for part in parts)
    )
    uncertainties = [part.uncertainty for part in parts if part.uncertainty]
    return LLMAssessment(
        verdict=verdict,
        summary=" ".join(part.summary for part in parts),
        findings=tuple(finding for part in parts for finding in part.findings),
        uncertainty=" ".join(uncertainties) if uncertainties else None,
    )
