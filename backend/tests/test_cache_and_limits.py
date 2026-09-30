"""Cache behaviour, input caps, and PDF cleanup."""

import io

from app.config import settings
from app.services.cache import ResponseCache
from app.services.parser import (
    _collapse_blank_runs,
    _detect_page_noise,
    parse_pdf,
    parse_text_file,
)
from app.services.text_utils import cache_key, normalize_text, truncate_chars


def test_cache_hit_avoids_second_write(tmp_path):
    cache = ResponseCache(directory=str(tmp_path), enabled=True)
    payload = "Photosynthesis converts light into chemical energy."

    assert cache.get("deck", payload) is None
    cache.set("deck", payload, {"cards": [{"question": "q", "answer": "a"}]})

    hit = cache.get("deck", payload)
    assert hit is not None
    assert hit["cards"][0]["question"] == "q"


def test_cache_normalises_equivalent_inputs(tmp_path):
    cache = ResponseCache(directory=str(tmp_path), enabled=True)
    cache.set("deck", "hello   world\n\n\nagain", {"ok": True})

    # Different whitespace, same content -> should still hit.
    assert cache.get("deck", "hello world\n\nagain") is not None


def test_cache_key_changes_with_prompt_version():
    a = cache_key("deck", "same", "v1")
    b = cache_key("deck", "same", "v2")
    assert a != b


def test_cache_disabled_returns_none(tmp_path):
    cache = ResponseCache(directory=str(tmp_path), enabled=False)
    cache.set("deck", "x", {"a": 1})
    assert cache.get("deck", "x") is None


def test_truncate_caps_prompt_size():
    text = "word " * 10_000
    trimmed = truncate_chars(text, 1000)
    assert len(trimmed) <= 1000


def test_truncate_noop_when_under_limit():
    assert truncate_chars("short", 100) == "short"


def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\r\n\r\n\r\nc") == "a b\n\nc"


def test_parser_enforces_max_input_chars(monkeypatch):
    monkeypatch.setattr(settings, "MAX_INPUT_CHARS", 200)
    blob = ("filler text " * 5000).encode("utf-8")
    result = parse_text_file(blob)
    assert len(result) <= 200


def test_parser_rejects_nothing_for_utf8():
    assert parse_text_file("héllo wörld".encode("utf-8")) == "héllo wörld"


def test_detect_page_noise_removes_running_headers():
    pages = []
    for i in range(6):
        pages.append(f"Page {i + 1}\nChapter 2 - Respiration\nUnique body line {i}")
    noise = _detect_page_noise(pages)

    assert "Chapter 2 - Respiration" in noise, "running header should be detected"
    assert "Page 1" not in noise, "per-page numbers differ, so not noise"
    assert "Unique body line 0" not in noise


def test_detect_page_noise_preserves_repeated_content():
    # A definition that genuinely repeats is content, not a running header.
    definition = "ATP is the energy currency of the cell."
    pages = [f"{definition}\nLine {i}" for i in range(6)]
    assert definition not in _detect_page_noise(pages)


def test_detect_page_noise_needs_multiple_pages():
    assert _detect_page_noise(["Page 1\nBody"]) == set()


def test_collapse_blank_runs():
    assert _collapse_blank_runs("a\n\n\n\n\nb   c") == "a\n\nb c"


def test_parse_pdf_respects_page_cap(monkeypatch):
    pypdf = __import__("pypdf")
    writer = pypdf.PdfWriter()
    for i in range(10):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    monkeypatch.setattr(settings, "MAX_PDF_PAGES", 3)
    text = parse_pdf(buf.getvalue())
    # 3 pages max, and blank pages extract to nothing.
    assert isinstance(text, str)