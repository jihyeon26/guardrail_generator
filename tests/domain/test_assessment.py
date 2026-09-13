import pytest

from sop_guardrail.domain.assessment import merge_assessments
from sop_guardrail.domain.models import (
    AssessmentFinding,
    AssessmentVerdict,
    LLMAssessment,
    Severity,
)


def _assessment(
    verdict: AssessmentVerdict,
    *,
    summary: str = "Reviewed the batch.",
    uncertainty: str | None = None,
    finding: str | None = None,
) -> LLMAssessment:
    findings = (
        (AssessmentFinding(code="GAP", message=finding, severity=Severity.MEDIUM),)
        if finding
        else ()
    )
    return LLMAssessment(
        verdict=verdict, summary=summary, findings=findings, uncertainty=uncertainty
    )


def test_a_single_batch_is_returned_unchanged() -> None:
    only = _assessment(AssessmentVerdict.PASS)

    assert merge_assessments((only,)) is only


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        ([AssessmentVerdict.PASS, AssessmentVerdict.PASS], AssessmentVerdict.PASS),
        ([AssessmentVerdict.PASS, AssessmentVerdict.ABSTAIN], AssessmentVerdict.ABSTAIN),
        ([AssessmentVerdict.ABSTAIN, AssessmentVerdict.REVISE], AssessmentVerdict.REVISE),
        ([AssessmentVerdict.REVISE, AssessmentVerdict.REJECT], AssessmentVerdict.REJECT),
        ([AssessmentVerdict.REJECT, AssessmentVerdict.PASS], AssessmentVerdict.REJECT),
    ],
)
def test_the_most_serious_batch_verdict_wins(
    verdicts: list[AssessmentVerdict], expected: AssessmentVerdict
) -> None:
    """A clean batch must not soften a damning one."""

    merged = merge_assessments(tuple(_assessment(verdict) for verdict in verdicts))

    assert merged.verdict is expected


def test_every_finding_and_uncertainty_survives_the_merge() -> None:
    parts = (
        _assessment(AssessmentVerdict.PASS, summary="Batch one is supported.", finding="gap one"),
        _assessment(
            AssessmentVerdict.REVISE,
            summary="Batch two overreaches.",
            uncertainty="The threshold is undefined.",
            finding="gap two",
        ),
    )

    merged = merge_assessments(parts)

    assert merged.summary == "Batch one is supported. Batch two overreaches."
    assert [finding.message for finding in merged.findings] == ["gap one", "gap two"]
    assert merged.uncertainty == "The threshold is undefined."


def test_no_uncertainty_from_any_batch_stays_absent() -> None:
    parts = (_assessment(AssessmentVerdict.PASS), _assessment(AssessmentVerdict.PASS))

    assert merge_assessments(parts).uncertainty is None


def test_merging_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot merge an empty set"):
        merge_assessments(())
