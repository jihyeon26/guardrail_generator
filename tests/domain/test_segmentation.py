import pytest

from sop_guardrail.domain.models import EvidenceSpan, SopDocument
from sop_guardrail.domain.segmentation import segment_document
from sop_guardrail.domain.validation import validate_evidence_spans
from tests.helpers import synthetic_document

SECTIONED_SOP = """Payment Release Procedure
Version 2

1. Purpose
This procedure governs the release of outbound payments.

2. Responsibilities
The operator prepares the payment.
The reviewer approves the payment.

3.1 Approval
A reviewer must approve a payment before the operator releases it.

3.2 Records
The approval record must identify the reviewer and the payment reference.
"""


def _sectioned_document() -> SopDocument:
    return SopDocument.from_text(
        document_id="sectioned-sop",
        source_name="payment-release.txt",
        text=SECTIONED_SOP,
    )


def _assert_spans_tile(spans: tuple[EvidenceSpan, ...], document: SopDocument) -> None:
    """Spans must be ordered, non-overlapping, and lose no non-blank text."""

    cursor = 0
    for span in spans:
        assert span.char_start >= cursor
        assert document.text[span.char_start : span.char_end] == span.quote
        assert not document.text[cursor : span.char_start].strip()
        cursor = span.char_end
    assert not document.text[cursor:].strip()


def test_numbered_headings_become_separate_spans() -> None:
    document = _sectioned_document()

    spans = segment_document(document)

    assert [span.quote.splitlines()[0] for span in spans] == [
        "Payment Release Procedure",
        "1. Purpose",
        "2. Responsibilities",
        "3.1 Approval",
        "3.2 Records",
    ]
    _assert_spans_tile(spans, document)
    assert validate_evidence_spans(spans, document) == ()


def test_span_ids_are_unique_ordered_and_tied_to_the_document() -> None:
    document = _sectioned_document()

    spans = segment_document(document)

    assert [span.evidence_id for span in spans] == [
        f"evidence-{document.sha256[:8]}-{index:03d}" for index in range(len(spans))
    ]
    assert len({span.evidence_id for span in spans}) == len(spans)


def test_a_document_without_headings_yields_one_span() -> None:
    document = synthetic_document()

    spans = segment_document(document)

    assert spans == (EvidenceSpan.from_document(document),)


def test_a_long_section_is_split_on_line_boundaries() -> None:
    document = _sectioned_document()

    spans = segment_document(document, max_span_chars=40)

    assert len(spans) > 5
    assert all(span.quote.strip() == span.quote for span in spans)
    _assert_spans_tile(spans, document)
    assert validate_evidence_spans(spans, document) == ()


def test_a_section_longer_than_the_limit_with_no_newline_is_kept_whole() -> None:
    """Splitting must never cut a line, even when the line exceeds the limit."""

    document = SopDocument.from_text(
        document_id="one-line-sop",
        source_name="one-line.txt",
        text="1. Purpose\n" + "A reviewer must approve the payment. " * 20,
    )

    spans = segment_document(document, max_span_chars=50)

    _assert_spans_tile(spans, document)
    assert any(len(span.quote) > 50 for span in spans)


def test_a_non_positive_span_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_span_chars must be at least 1"):
        segment_document(synthetic_document(), max_span_chars=0)
