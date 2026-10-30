import re
from datetime import date, datetime
from io import BytesIO

from pypdf import PdfReader


class UnsupportedDocumentError(ValueError):
    """Raised when an uploaded file type is not supported."""


class EmptyDocumentError(ValueError):
    """Raised when a document has no extractable text."""


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving the extracted wording."""
    return re.sub(r"\s+", " ", text).strip()


def extract_effective_date(text: str) -> date | None:
    """Read a simple `Effective date:` value from extracted document text."""
    match = re.search(
        r"\bEffective\s+date\s*:\s*(\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    raw_value = match.group(1)
    for date_format in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue
    return None


def parse_uploaded_file(filename: str, content: bytes) -> str:
    """Extract text from a PDF or UTF-8 plain-text upload."""
    suffix = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if suffix == "pdf":
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"Could not extract text from PDF '{filename}'.") from exc
    elif suffix == "txt":
        try:
            extracted = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Plain-text uploads must use UTF-8 encoding.") from exc
    else:
        raise UnsupportedDocumentError("Only PDF and .txt files are supported.")

    text = clean_text(extracted)
    if not text:
        raise EmptyDocumentError(f"No text could be extracted from '{filename}'.")
    return text
