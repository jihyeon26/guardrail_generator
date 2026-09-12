from pathlib import Path
from typing import Any

import pytest

from sop_guardrail.infrastructure.documents import (
    document_id_from_path,
    load_sop_document,
    normalize_sop_text,
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_text_sop_becomes_a_hashed_document(tmp_path: Path) -> None:
    path = _write(tmp_path, "Payment Release SOP.txt", "  A reviewer must approve the payment.  ")

    document = load_sop_document(path)

    assert document.document_id == "payment-release-sop"
    assert document.source_name == "Payment Release SOP.txt"
    assert document.text == "A reviewer must approve the payment."


def test_an_explicit_document_id_overrides_the_file_name(tmp_path: Path) -> None:
    path = _write(tmp_path, "sop.md", "A reviewer must approve the payment.")

    assert load_sop_document(path, document_id="ap-v1").document_id == "ap-v1"


def test_document_id_rejects_a_name_with_no_usable_characters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot derive a document id"):
        document_id_from_path(tmp_path / "___.txt")


def test_unsupported_file_types_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "sop.docx", "content")

    with pytest.raises(ValueError, match="unsupported SOP file type"):
        load_sop_document(path)


def test_normalization_drops_extraction_artifacts_but_keeps_prose() -> None:
    raw = "1 | P a g e\n• Invoice Receipt:\n    o Invoices arrive by mail.\n\n\n\nor by email.   \n"

    assert normalize_sop_text(raw) == "Invoice Receipt:\nInvoices arrive by mail.\n\nor by email."


def test_normalization_keeps_a_word_that_starts_with_the_bullet_letter() -> None:
    """A PDF sub-bullet is "o " followed by a space; "or" is prose."""

    assert normalize_sop_text("or processing payments") == "or processing payments"


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, pages: list[str]) -> None:
        self.pages = [_FakePage(text) for text in pages]


def _patch_reader(monkeypatch: pytest.MonkeyPatch, pages: list[str]) -> None:
    pypdf = pytest.importorskip("pypdf")
    monkeypatch.setattr(pypdf, "PdfReader", lambda _path: _FakeReader(pages))


def test_pdf_pages_are_joined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_reader(monkeypatch, ["1 | P a g e\nFirst page.", "  ", "Second page."])
    path = _write(tmp_path, "sop.pdf", "binary placeholder")

    document = load_sop_document(path)

    assert document.text == "First page.\n\nSecond page."


def test_a_scanned_pdf_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_reader(monkeypatch, ["", "   "])
    path = _write(tmp_path, "scan.pdf", "binary placeholder")

    with pytest.raises(ValueError, match="no extractable text"):
        load_sop_document(path)


def test_pages_without_text_return_an_empty_string(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """pypdf returns None for a page it cannot decode."""

    class _NonePage:
        def extract_text(self) -> Any:
            return None

    pypdf = pytest.importorskip("pypdf")
    reader = type("_Reader", (), {"pages": [_NonePage(), _FakePage("Only page.")]})
    monkeypatch.setattr(pypdf, "PdfReader", lambda _path: reader())
    path = _write(tmp_path, "sop.pdf", "binary placeholder")

    assert load_sop_document(path).text == "Only page."
