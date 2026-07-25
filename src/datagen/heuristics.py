from __future__ import annotations

import re

_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_CLASS = f"0-9{_ARABIC_DIGITS}"

_OBJECTIVE_SENTENCE_RE = re.compile(
    r"أن\s+[يت][؀-ۿ\s]{3,200}?[.؟!]"
)

_TIME_ESTIMATE_NUMBER_FIRST_RE = re.compile(
    rf"\(?\s*([{_DIGIT_CLASS}]+)\s*(?:دقيقة|دقائق|دقيقه)\s*\)?"
)
# PDF text extraction of RTL runs often mirrors parenthesized numbers next to Arabic
# text (e.g. "(12 دقيقة)" comes out as "دقيقة)12("), so the keyword-before-number
# ordering has to be matched too, not just treated as a fallback edge case.
_TIME_ESTIMATE_KEYWORD_FIRST_RE = re.compile(
    rf"(?:دقيقة|دقائق|دقيقه)\s*\)?\s*([{_DIGIT_CLASS}]+)\s*\(?"
)

_INSTRUCTION_VERBS = (
    "اطلب", "قسّم", "قسم", "وزّع", "وزع", "ناقش", "اعرض",
    "استمع", "اقرأ", "اسأل", "شجّع", "شجع", "ذكّر", "ذكر",
)
_LEADING_CONNECTORS = ("ثم", "بعد ذلك", "وبعدها", "بعدها", "بعد ذلك،")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.؟!])\s+")

# Running-header unit/lesson markers (printed in the page margin) extract in
# reversed "N الوحدة" order more reliably than forward "الوحدة N" order here —
# likely a bidi artifact of a vertically-set sidebar label. A plausibility cap
# rejects incidental matches (a stray exercise number sitting next to an
# unrelated mention of "الدرس" elsewhere on the page).
_MAX_PLAUSIBLE_UNIT_OR_LESSON = 12
_UNIT_REVERSED_RE = re.compile(rf"([{_DIGIT_CLASS}]+)\s*الوحدةُ?")
_UNIT_FORWARD_RE = re.compile(rf"الوحدةُ?\s*([{_DIGIT_CLASS}]+)")
_LESSON_REVERSED_RE = re.compile(rf"([{_DIGIT_CLASS}]+)\s*الدرسُ?")
_LESSON_FORWARD_RE = re.compile(rf"الدرسُ?\s*([{_DIGIT_CLASS}]+)")


def _normalize_arabic_number(num_str: str) -> int:
    translation = str.maketrans(_ARABIC_DIGITS, "0123456789")
    return int(num_str.translate(translation))


def extract_learning_objectives(text: str) -> list[str]:
    if not text:
        return []
    matches = _OBJECTIVE_SENTENCE_RE.findall(text)
    seen: set[str] = set()
    results: list[str] = []
    for m in _OBJECTIVE_SENTENCE_RE.finditer(text):
        sentence = m.group(0).strip()
        if sentence not in seen:
            seen.add(sentence)
            results.append(sentence)
    return results


def extract_time_estimate(text: str) -> str | None:
    if not text:
        return None
    match = _TIME_ESTIMATE_NUMBER_FIRST_RE.search(text) or _TIME_ESTIMATE_KEYWORD_FIRST_RE.search(text)
    if not match:
        return None
    minutes = _normalize_arabic_number(match.group(1))
    unit = "دقيقة" if minutes == 1 else "دقائق"
    return f"{minutes} {unit}"


def _extract_bounded_number(reversed_re: re.Pattern, forward_re: re.Pattern, text: str) -> str | None:
    for pattern in (reversed_re, forward_re):
        match = pattern.search(text)
        if match:
            value = _normalize_arabic_number(match.group(1))
            if 1 <= value <= _MAX_PLAUSIBLE_UNIT_OR_LESSON:
                return str(value)
    return None


def extract_unit(text: str) -> str | None:
    if not text:
        return None
    return _extract_bounded_number(_UNIT_REVERSED_RE, _UNIT_FORWARD_RE, text)


def extract_lesson(text: str) -> str | None:
    if not text:
        return None
    return _extract_bounded_number(_LESSON_REVERSED_RE, _LESSON_FORWARD_RE, text)


def extract_instructions(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    results: list[str] = []
    for raw_sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = raw_sentence.strip(" .؟!\n")
        if not sentence:
            continue
        core = sentence
        for connector in _LEADING_CONNECTORS:
            if core.startswith(connector + " "):
                core = core[len(connector):].strip()
                break
        first_word = core.split(" ", 1)[0] if core else ""
        if first_word in _INSTRUCTION_VERBS and core not in seen:
            seen.add(core)
            results.append(core)
    return results


def extract_heuristic_fields(text: str) -> dict:
    return {
        "LearningObjectives": extract_learning_objectives(text),
        "TimeEstimate": extract_time_estimate(text),
        "Instructions": "; ".join(extract_instructions(text)) or None,
        "Unit": extract_unit(text),
        "Lesson": extract_lesson(text),
    }
