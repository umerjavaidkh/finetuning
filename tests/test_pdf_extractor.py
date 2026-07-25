import os

import fitz
import pytest

from datagen.pdf_extractor import (
    _forward_fill_unit_lesson,
    _stable_doc_id,
    extract_pdf,
    extract_pdf_usable_only,
)

REAL_SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__), "..", "books_data", "BilArabi_TG07_BookPages_65_66_only.pdf"
)


def _make_pdf(tmp_path, texts: list[str]):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_stable_doc_id_is_deterministic_from_filename():
    assert _stable_doc_id("/some/path/book.pdf") == _stable_doc_id("/other/path/book.pdf")
    assert _stable_doc_id("/some/path/book_a.pdf") != _stable_doc_id("/some/path/book_b.pdf")


def test_extract_pdf_produces_expected_shape_and_page_numbering(tmp_path):
    long_text = "This is a sample page of extracted text content for testing. " * 5
    pdf_path = _make_pdf(tmp_path, [long_text, long_text])

    entries = extract_pdf(pdf_path, doc_id="fixture-doc")

    assert len(entries) == 2
    assert [e["Page"] for e in entries] == [1, 2]
    assert [e["id"] for e in entries] == ["fixture-doc_1", "fixture-doc_2"]
    assert all(e["DocId"] == "fixture-doc" for e in entries)
    assert all(e["SourceFormat"] == "pdf" for e in entries)
    assert all(e["ContentType"] is None for e in entries)
    assert all("_extraction_quality" in e for e in entries)
    assert all(set(e["_extraction_quality"].keys()) == {
        "passed", "confidence", "hard_failures", "soft_warnings", "metrics"
    } for e in entries)


def test_extract_pdf_flags_non_arabic_content_as_failing_quality(tmp_path):
    pdf_path = _make_pdf(tmp_path, ["Plain English content with no Arabic at all here. " * 5])
    entries = extract_pdf(pdf_path)
    assert entries[0]["_extraction_quality"]["passed"] is False
    assert "low_arabic_content" in entries[0]["_extraction_quality"]["hard_failures"]


def test_extract_pdf_usable_only_filters_out_failing_pages(tmp_path):
    pdf_path = _make_pdf(tmp_path, ["x", "This is a longer plain english page of text content. " * 5])
    all_entries = extract_pdf(pdf_path)
    usable_entries = extract_pdf_usable_only(pdf_path)
    assert len(all_entries) == 2
    assert len(usable_entries) == 0


def test_forward_fill_unit_lesson_propagates_from_last_explicit_marker():
    entries = [
        {"Unit": None, "Lesson": None},
        {"Unit": "1", "Lesson": None},
        {"Unit": None, "Lesson": "4"},
        {"Unit": None, "Lesson": None},
        {"Unit": "2", "Lesson": None},
    ]

    _forward_fill_unit_lesson(entries)

    assert [e["Unit"] for e in entries] == [None, "1", "1", "1", "2"]
    assert [e["Lesson"] for e in entries] == [None, None, "4", "4", "4"]
    # entry 2's Unit and entry 4's Lesson are each filled even though the same
    # entry also carries an explicit value for the other field.
    assert [e["_unit_lesson_forward_filled"] for e in entries] == [False, False, True, True, True]


@pytest.mark.skipif(
    not os.path.exists(REAL_SAMPLE_PDF),
    reason="real book sample not present in this environment (gitignored, local-only data)",
)
def test_real_book_sample_extracts_usable_arabic_pages_with_time_estimates():
    entries = extract_pdf(REAL_SAMPLE_PDF)
    assert len(entries) == 2
    assert all(e["_extraction_quality"]["passed"] for e in entries)
    assert all(e["_extraction_quality"]["metrics"]["arabic_ratio"] > 0.9 for e in entries)
    time_estimates = [e["TimeEstimate"] for e in entries]
    assert "12 دقائق" in time_estimates
    assert "10 دقائق" in time_estimates
