"""Split an SOP into section-level evidence spans with exact character offsets."""

from __future__ import annotations

import re

from sop_guardrail.domain.models import EvidenceSpan, SopDocument

DEFAULT_MAX_SPAN_CHARS = 1500

# "1. Purpose", "5.1 Receiving and Logging Invoices", "7 Review and Revision".
_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?[ \t]+\S.*$", re.MULTILINE)


def segment_document(
    document: SopDocument, *, max_span_chars: int = DEFAULT_MAX_SPAN_CHARS
) -> tuple[EvidenceSpan, ...]:
    """Cut the document into ordered, non-overlapping spans.

    Numbered headings are the primary boundary because an SOP states one obligation
    per section; a section longer than ``max_span_chars`` is split again on line
    boundaries so a policy can cite the paragraph it came from rather than the
    whole procedure.
    """

    if max_span_chars < 1:
        raise ValueError("max_span_chars must be at least 1")

    spans: list[EvidenceSpan] = []
    for start, end in _section_bounds(document.text):
        for piece_start, piece_end in _split_to_fit(document.text, start, end, max_span_chars):
            if document.text[piece_start:piece_end].strip():
                spans.append(
                    EvidenceSpan.from_slice(
                        document,
                        char_start=piece_start,
                        char_end=piece_end,
                        index=len(spans),
                    )
                )

    if not spans:  # a document with no non-blank content cannot reach here
        raise ValueError(f"{document.document_id} produced no evidence spans")
    return tuple(spans)


def _section_bounds(text: str) -> list[tuple[int, int]]:
    """Return [start, end) for each heading-led section, preamble included."""

    starts = [match.start() for match in _HEADING.finditer(text)]
    if not starts:
        return [(0, len(text))]
    if starts[0] > 0:
        starts.insert(0, 0)
    return [(start, end) for start, end in zip(starts, [*starts[1:], len(text)], strict=True)]


def _split_to_fit(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Break [start, end) on line boundaries so each piece fits within the limit."""

    if end - start <= limit:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    piece_start = start
    cursor = start
    while cursor < end:
        newline = text.find("\n", cursor, end)
        line_end = end if newline < 0 else newline + 1
        if line_end - piece_start > limit and cursor > piece_start:
            pieces.append((piece_start, cursor))
            piece_start = cursor
        cursor = line_end
    pieces.append((piece_start, end))
    return pieces
