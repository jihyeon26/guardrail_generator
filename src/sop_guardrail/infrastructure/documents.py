"""Load an SOP file from disk into the domain's evidence-bearing contract."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from sop_guardrail.domain.models import SopDocument

TEXT_SUFFIXES = frozenset({".txt", ".md"})
PDF_SUFFIXES = frozenset({".pdf"})

_SOFT_HYPHEN = "­"
# A lone "o" or "-" is a PDF sub-bullet only when a space follows it; "or ..." is prose.
_BULLETS = re.compile(r"^[ \t]*(?:[\u2022\u25cf\u25e6\u25aa]|o(?=\s)|-(?=\s))[ \t]*", re.MULTILINE)
_PAGE_FURNITURE = re.compile(r"^\s*\d+\s*\|\s*P\s*a\s*g\s*e\s*$", re.MULTILINE | re.IGNORECASE)
_BLANK_RUN = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def document_id_from_path(path: Path) -> str:
    """Derive an id that satisfies the SopDocument id pattern."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").lower()
    if not slug:
        raise ValueError(f"cannot derive a document id from {path.name}")
    return slug


def load_sop_document(path: Path, *, document_id: str | None = None) -> SopDocument:
    """Read a .txt, .md, or .pdf SOP.

    Evidence spans are character offsets into the stored text, so the text is
    normalized once here and never rewritten downstream.
    """

    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8")
    elif suffix in PDF_SUFFIXES:
        raw = _read_pdf(path)
    else:
        supported = ", ".join(sorted(TEXT_SUFFIXES | PDF_SUFFIXES))
        raise ValueError(f"unsupported SOP file type {suffix or path.name!r}; expected {supported}")

    return SopDocument.from_text(
        document_id=document_id or document_id_from_path(path),
        source_name=path.name,
        text=normalize_sop_text(raw),
    )


def normalize_sop_text(raw: str) -> str:
    """Remove extraction artifacts that would otherwise be quoted as evidence."""

    text = unicodedata.normalize("NFKC", raw).replace(_SOFT_HYPHEN, "").replace("\r\n", "\n")
    text = _PAGE_FURNITURE.sub("", text)
    text = _BULLETS.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    return _BLANK_RUN.sub("\n\n", text).strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("install the project with the 'documents' extra to read PDFs") from exc

    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        raise ValueError(f"{path.name} contains no extractable text; it may be a scanned image")
    return text
