from __future__ import annotations

import hashlib
import os

import fitz

from .extraction_validator import validate_extraction
from .heuristics import extract_heuristic_fields


def _stable_doc_id(pdf_path: str) -> str:
    return hashlib.md5(os.path.basename(pdf_path).encode("utf-8")).hexdigest()


def _forward_fill_unit_lesson(entries: list[dict]) -> None:
    last_unit: str | None = None
    last_lesson: str | None = None
    for entry in entries:
        entry["_unit_lesson_forward_filled"] = False

        if entry["Unit"] is not None:
            last_unit = entry["Unit"]
        elif last_unit is not None:
            entry["Unit"] = last_unit
            entry["_unit_lesson_forward_filled"] = True

        if entry["Lesson"] is not None:
            last_lesson = entry["Lesson"]
        elif last_lesson is not None:
            entry["Lesson"] = last_lesson
            entry["_unit_lesson_forward_filled"] = True


def extract_pdf(pdf_path: str, doc_id: str | None = None) -> list[dict]:
    doc_id = doc_id or _stable_doc_id(pdf_path)
    file_name = os.path.basename(pdf_path)
    entries: list[dict] = []

    pdf = fitz.open(pdf_path)
    try:
        for page_index, page in enumerate(pdf, start=1):
            content_text = page.get_text()
            quality = validate_extraction(content_text)
            heuristic_fields = extract_heuristic_fields(content_text)

            entries.append(
                {
                    "id": f"{doc_id}_{page_index}",
                    "DocId": doc_id,
                    "FileName": file_name,
                    "Page": page_index,
                    "BookPageNumber": None,
                    "SourceFormat": "pdf",
                    "Language": "ar",
                    "Unit": heuristic_fields["Unit"],
                    "UnitTitle": None,
                    "Lesson": heuristic_fields["Lesson"],
                    "LessonTitle": None,
                    "ContentType": None,
                    "Content": content_text,
                    "LearningObjectives": heuristic_fields["LearningObjectives"],
                    "Instructions": heuristic_fields["Instructions"],
                    "TimeEstimate": heuristic_fields["TimeEstimate"],
                    "Materials": [],
                    "Skills": [],
                    "Difficulty": None,
                    "Characters": [],
                    "Answers": [],
                    "_extraction_quality": {
                        "passed": quality.passed,
                        "confidence": quality.confidence,
                        "hard_failures": quality.hard_failures,
                        "soft_warnings": quality.soft_warnings,
                        "metrics": quality.metrics,
                    },
                }
            )
    finally:
        pdf.close()

    _forward_fill_unit_lesson(entries)
    return entries


def extract_pdf_usable_only(pdf_path: str, doc_id: str | None = None) -> list[dict]:
    return [e for e in extract_pdf(pdf_path, doc_id=doc_id) if e["_extraction_quality"]["passed"]]
