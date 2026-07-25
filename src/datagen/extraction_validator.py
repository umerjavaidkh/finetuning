from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_ARABIC_BLOCK = r"؀-ۿݐ-ݿࢠ-ࣿ"
_PRESENTATION_FORMS = r"ﭐ-﷿ﹰ-﻿"
_ARABIC_LETTER_RE = re.compile(f"[{_ARABIC_BLOCK}]")
_PRESENTATION_FORM_RE = re.compile(f"[{_PRESENTATION_FORMS}]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_REPLACEMENT_CHAR_RE = re.compile("�")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

MIN_CONTENT_CHARS = 30
MIN_ARABIC_RATIO = 0.25
MAX_PRESENTATION_FORM_RATIO = 0.15
MAX_ENCODING_ERROR_RATIO = 0.01
MAX_LINE_REPETITION_RATIO = 0.35
MAX_AVG_WORD_LENGTH = 18
FRONT_MATTER_SCAN_LINES = 6

# Front-matter pages (title page, author bio, curriculum introduction) are
# conventionally numbered with bare lowercase roman numerals rather than
# Arabic numerals. That page-number line survives PyMuPDF extraction as its
# own line, giving a free, reliable signal to exclude non-lesson content
# that would otherwise pass every other quality check.
def _roman_numerals(max_value: int = 100) -> set[str]:
    values = [
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    numerals = set()
    for n in range(1, max_value + 1):
        remaining, roman = n, ""
        for value, symbol in values:
            while remaining >= value:
                roman += symbol
                remaining -= value
        numerals.add(roman)
    return numerals


_ROMAN_NUMERALS = _roman_numerals()

# A 4-level assessment-rubric appendix (below/approaching/meets/exceeds standard,
# per learning indicator) is real curriculum reference material but not itself a
# lesson activity. Unlike a legitimate unit-overview page (which may mention
# "المعايير" once as a section heading alongside real learning objectives), a
# rubric-descriptor page repeats several of these exact scale phrases together.
_RUBRIC_SCALE_TERMS = ("دون المعيار", "يقترب من المعيار", "ضمن المعيار", "يفوق المعيار")
MIN_RUBRIC_SCALE_TERM_HITS = 2


def _looks_like_rubric_boilerplate(text: str) -> bool:
    hits = sum(1 for term in _RUBRIC_SCALE_TERMS if term in text)
    return hits >= MIN_RUBRIC_SCALE_TERM_HITS


@dataclass
class ExtractionQuality:
    passed: bool
    confidence: float
    hard_failures: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _looks_like_front_matter(text: str) -> bool:
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    return any(ln in _ROMAN_NUMERALS for ln in lines[:FRONT_MATTER_SCAN_LINES])


def _line_repetition_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 10]
    if len(lines) < 4:
        return 0.0
    counts = Counter(lines)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(lines)


def _avg_word_length(text: str) -> float:
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def validate_extraction(content_text: str, *, expect_arabic: bool = True) -> ExtractionQuality:
    hard_failures: list[str] = []
    soft_warnings: list[str] = []
    text = content_text or ""
    stripped = text.strip()

    metrics = {
        "length": len(stripped),
    }

    if len(stripped) < MIN_CONTENT_CHARS:
        hard_failures.append("too_short")
        return ExtractionQuality(
            passed=False, confidence=0.0, hard_failures=hard_failures, metrics=metrics
        )

    arabic_chars = len(_ARABIC_LETTER_RE.findall(stripped))
    presentation_chars = len(_PRESENTATION_FORM_RE.findall(stripped))
    latin_chars = len(_LATIN_LETTER_RE.findall(stripped))
    letter_chars = arabic_chars + presentation_chars + latin_chars

    arabic_ratio = (arabic_chars + presentation_chars) / letter_chars if letter_chars else 0.0
    presentation_ratio = presentation_chars / (arabic_chars + presentation_chars) if (arabic_chars + presentation_chars) else 0.0
    replacement_ratio = len(_REPLACEMENT_CHAR_RE.findall(stripped)) / len(stripped) if stripped else 0.0
    control_ratio = len(_CONTROL_CHAR_RE.findall(stripped)) / len(stripped) if stripped else 0.0
    encoding_error_ratio = replacement_ratio + control_ratio
    repetition_ratio = _line_repetition_ratio(stripped)
    avg_word_len = _avg_word_length(stripped)

    metrics.update(
        {
            "arabic_ratio": round(arabic_ratio, 3),
            "presentation_form_ratio": round(presentation_ratio, 3),
            "encoding_error_ratio": round(encoding_error_ratio, 4),
            "line_repetition_ratio": round(repetition_ratio, 3),
            "avg_word_length": round(avg_word_len, 1),
        }
    )

    if _looks_like_front_matter(stripped):
        hard_failures.append("front_matter_page")

    if _looks_like_rubric_boilerplate(stripped):
        hard_failures.append("rubric_boilerplate_page")

    if expect_arabic and arabic_ratio < MIN_ARABIC_RATIO:
        hard_failures.append("low_arabic_content")

    if presentation_ratio > MAX_PRESENTATION_FORM_RATIO:
        hard_failures.append("broken_text_layer_needs_ocr")

    if encoding_error_ratio > MAX_ENCODING_ERROR_RATIO:
        hard_failures.append("encoding_errors")

    if repetition_ratio > MAX_LINE_REPETITION_RATIO:
        soft_warnings.append("excessive_repetition")

    if avg_word_len > MAX_AVG_WORD_LENGTH:
        soft_warnings.append("abnormal_word_length_broken_tokenization")

    passed = len(hard_failures) == 0
    confidence = 1.0
    if passed:
        confidence -= 0.15 * len(soft_warnings)
        confidence -= max(0.0, (MIN_ARABIC_RATIO - arabic_ratio)) if arabic_ratio < MIN_ARABIC_RATIO else 0.0
    else:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return ExtractionQuality(
        passed=passed,
        confidence=round(confidence, 2),
        hard_failures=hard_failures,
        soft_warnings=soft_warnings,
        metrics=metrics,
    )
