import io
import re
from collections import Counter

from pypdf import PdfReader

from app.config import settings
from app.services.text_utils import truncate_chars


_PAGE_NUMBER_RE = re.compile(
    r"^(page\s*)?[\divxlc]{1,6}(\s*(of|/)\s*[\divxlc]{1,6})?$", re.IGNORECASE
)


def _looks_like_sentence(line: str) -> bool:
    """True for prose lines, which must never be treated as page furniture."""
    words = line.split()
    if len(words) < 5:
        return False
    return line.rstrip().endswith((".", "!", "?"))


def _is_header_candidate(line: str) -> bool:
    """Gate repetition on shape, not repetition alone.

    A running header and a definition repeated on every page look identical
    structurally, so repetition by itself would silently delete real content.
    Page furniture is either page-number-like or a short line that does not
    read as a sentence.
    """
    if not line or len(line) >= 80:
        return False
    if _PAGE_NUMBER_RE.match(line):
        return True
    return not _looks_like_sentence(line)


def _detect_page_noise(pages: list[str]) -> set[str]:
    """Find running headers/footers/watermarks shared across most pages."""
    if len(pages) < 3:
        return set()

    per_page = [
        {line.strip() for line in page.split("\n") if _is_header_candidate(line.strip())}
        for page in pages
    ]

    candidates: Counter[str] = Counter()
    for lines in per_page:
        candidates.update(lines)

    # Must appear on at least 60% of pages to count as a running header.
    threshold = max(3, int(len(pages) * 0.6))
    return {line for line, count in candidates.items() if count >= threshold}


def _collapse_blank_runs(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(file_bytes: bytes) -> str:
    """Extract capped, cleaned text from a PDF."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = reader.pages[: settings.MAX_PDF_PAGES]

        raw_pages: list[str] = []
        for page in pages:
            try:
                raw_pages.append(page.extract_text() or "")
            except Exception:
                raw_pages.append("")

        noise = _detect_page_noise(raw_pages)

        parts: list[str] = []
        for number, text in enumerate(raw_pages, start=1):
            kept = [line for line in text.split("\n") if line.strip() not in noise]
            body = "\n".join(kept).strip()
            if body:
                parts.append(f"--- page {number} ---\n{body}")

        text = _collapse_blank_runs("\n".join(parts))
        return truncate_chars(text, settings.MAX_INPUT_CHARS)
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


def parse_text_file(file_bytes: bytes) -> str:
    """Decode a .txt/.md upload, then apply the same cleanup as PDFs."""
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        text = _collapse_blank_runs(text)
        return truncate_chars(text, settings.MAX_INPUT_CHARS)
    except Exception as exc:
        raise ValueError(f"Failed to parse text file: {exc}") from exc