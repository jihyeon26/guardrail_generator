from hashlib import sha256

import pytest
from pydantic import ValidationError

from sop_guardrail.domain.models import (
    EvidenceSpan,
    FeedbackCard,
    FeedbackStage,
    FeedbackStatus,
    ReviewGate,
    SopDocument,
)


def test_document_factory_normalizes_and_hashes_text() -> None:
    document = SopDocument.from_text(
        document_id="sample-1", source_name="sample.txt", text="  Public SOP text.  "
    )

    assert document.text == "Public SOP text."
    assert document.sha256 == sha256(b"Public SOP text.").hexdigest()


def test_evidence_span_rejects_range_that_does_not_match_quote() -> None:
    with pytest.raises(ValidationError, match="quote length"):
        EvidenceSpan(
            evidence_id="evidence-1",
            document_id="sample-1",
            char_start=0,
            char_end=4,
            quote="too long",
            quote_sha256=sha256(b"too long").hexdigest(),
        )


def test_active_feedback_requires_approver() -> None:
    with pytest.raises(ValidationError, match="approved_by"):
        FeedbackCard(
            feedback_id="feedback-1",
            stage=FeedbackStage.POLICY_EXTRACTION,
            lesson="Cite the exact normative sentence.",
            source_run_id="run-1",
            source_gate=ReviewGate.POLICY,
            status=FeedbackStatus.ACTIVE,
        )
