from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    EXAM = "exam"
    WORKSHEET = "worksheet"
    INDICATOR_QUESTIONS = "indicator_questions"
    LESSON_PLAN = "lesson_plan"
    GRAMMAR_EXPLANATION = "grammar_explanation"
    VOCAB_ACTIVITY = "vocab_activity"
    STORY_COMPREHENSION = "story_comprehension"


_VOCAB_KEYWORDS = ("vocabulary", "dictionary", "root", "معجم", "مفردات", "جذر")
_GRAMMAR_KEYWORDS = ("grammar", "قواعد", "نحو", "صرف")
_MIN_USABLE_CONTENT_CHARS = 80
_MIN_EXAM_CONTENT_CHARS = 400
_MIN_WORKSHEET_INSTRUCTION_STEPS = 2

# ContentType only exists when the source went through the RAG pipeline's LLM
# classification; raw PDF extraction (pdf_extractor.py) always leaves it None.
# These are content types observed to be pure metadata/navigation stubs (never
# real assessable content), so they stay excluded from EXAM candidacy even
# when ContentType is available; everything else is judged on content alone.
_EXAM_EXCLUDED_CONTENT_TYPES = {"header", "classification", "answer"}


def _instruction_step_count(instructions: str | None) -> int:
    if not instructions:
        return 0
    return instructions.count(";") + 1


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def _extract_number(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _keyword_hit(keywords: tuple[str, ...], *texts: Any) -> bool:
    haystack = " ".join(
        " ".join(t) if isinstance(t, list) else str(t or "") for t in texts
    ).lower()
    return any(kw.lower() in haystack for kw in keywords)


@dataclass
class CurriculumContext:
    source_id: str
    doc_id: str
    page: int | None
    book_page_number: str | None

    raw_unit: str | None
    raw_lesson: str | None
    unit: int | None
    lesson: int | None
    unit_title: str | None
    lesson_title: str | None

    title: str | None
    section: str | None
    content_type: str | None
    topics: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    difficulty: str | None = None

    learning_objectives: list[str] = field(default_factory=list)
    instructions: str | None = None
    materials: list[str] = field(default_factory=list)
    time_estimate: str | None = None
    content_text: str | None = None

    characters: list[str] = field(default_factory=list)
    story_setting: str | None = None
    story_problem: str | None = None
    story_solution: str | None = None
    story_main_events: list[str] = field(default_factory=list)

    answers: list[str] = field(default_factory=list)


@dataclass
class SFTPromptCandidate:
    task_type: TaskType
    context_block: str
    source_entry_id: str
    doc_id: str
    unit: int | None
    lesson: int | None
    lesson_title: str | None
    content_type: str | None
    oversample_weight: float = 1.0


def adapt_entry(entry: dict) -> CurriculumContext:
    raw_unit = entry.get("Unit") or None
    raw_lesson = entry.get("Lesson") or None
    return CurriculumContext(
        source_id=entry.get("id", ""),
        doc_id=entry.get("DocId", ""),
        page=entry.get("Page"),
        book_page_number=entry.get("BookPageNumber") or None,
        raw_unit=raw_unit,
        raw_lesson=raw_lesson,
        unit=_extract_number(raw_unit),
        lesson=_extract_number(raw_lesson),
        unit_title=entry.get("UnitTitle") or None,
        lesson_title=entry.get("LessonTitle") or None,
        title=entry.get("Title") or None,
        section=entry.get("Section") or None,
        content_type=entry.get("ContentType") or None,
        topics=entry.get("Topics") or [],
        skills=entry.get("Skills") or [],
        difficulty=entry.get("Difficulty") or None,
        learning_objectives=entry.get("LearningObjectives") or [],
        instructions=entry.get("Instructions") or None,
        materials=entry.get("Materials") or [],
        time_estimate=entry.get("TimeEstimate") or None,
        content_text=entry.get("Content") or None,
        characters=entry.get("Characters") or [],
        story_setting=entry.get("StoryElementsSetting") or None,
        story_problem=entry.get("StoryElementsProblem") or None,
        story_solution=entry.get("StoryElementsSolution") or None,
        story_main_events=entry.get("StoryElementsMainEvents") or [],
        answers=entry.get("Answers") or [],
    )


def is_usable(ctx: CurriculumContext) -> bool:
    return _nonempty(ctx.content_text) and len(ctx.content_text.strip()) >= _MIN_USABLE_CONTENT_CHARS


def infer_task_types(ctx: CurriculumContext) -> list[TaskType]:
    if not is_usable(ctx):
        return []

    types: list[TaskType] = []
    ct = ctx.content_type or ""

    if ct in ("story", "tasnif_story") and _nonempty(ctx.content_text):
        types.append(TaskType.STORY_COMPREHENSION)

    if ct == "vocabulary" or _keyword_hit(_VOCAB_KEYWORDS, ctx.skills, ctx.topics, ctx.learning_objectives):
        types.append(TaskType.VOCAB_ACTIVITY)

    if _keyword_hit(_GRAMMAR_KEYWORDS, ctx.skills, ctx.topics, ctx.learning_objectives):
        types.append(TaskType.GRAMMAR_EXPLANATION)

    if _nonempty(ctx.learning_objectives):
        types.append(TaskType.INDICATOR_QUESTIONS)

    if _nonempty(ctx.instructions):
        types.append(TaskType.LESSON_PLAN)

    if _nonempty(ctx.instructions) and (
        ct == "exercise" or _instruction_step_count(ctx.instructions) >= _MIN_WORKSHEET_INSTRUCTION_STEPS
    ):
        types.append(TaskType.WORKSHEET)

    if ct not in _EXAM_EXCLUDED_CONTENT_TYPES and len(ctx.content_text.strip()) > _MIN_EXAM_CONTENT_CHARS:
        types.append(TaskType.EXAM)

    # dedupe while preserving order
    seen: set[TaskType] = set()
    ordered: list[TaskType] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def build_context_block(ctx: CurriculumContext, grade_level: int | None = None) -> str:
    lines: list[str] = ["<السياق>"]

    if grade_level is not None:
        lines.append(f"المستوى: {grade_level}")
    if ctx.unit is not None or ctx.raw_unit:
        unit_label = ctx.unit if ctx.unit is not None else ctx.raw_unit
        unit_line = f"الوحدة: {unit_label}"
        if ctx.unit_title:
            unit_line += f" — {ctx.unit_title}"
        lines.append(unit_line)
    if ctx.lesson is not None or ctx.raw_lesson:
        lesson_label = ctx.lesson if ctx.lesson is not None else ctx.raw_lesson
        lesson_line = f"الدرس: {lesson_label}"
        if ctx.lesson_title:
            lesson_line += f" — {ctx.lesson_title}"
        lines.append(lesson_line)

    if ctx.learning_objectives:
        lines.append("مؤشرات التعلّم:")
        lines.extend(f"- {obj}" for obj in ctx.learning_objectives)

    if ctx.skills:
        lines.append("المهارات: " + "، ".join(ctx.skills))

    if ctx.difficulty:
        lines.append(f"المستوى الصعوبة: {ctx.difficulty}")

    if ctx.characters or ctx.story_setting or ctx.story_problem or ctx.story_solution:
        if ctx.characters:
            lines.append("الشخصيات: " + "، ".join(ctx.characters))
        if ctx.story_setting:
            lines.append(f"المكان والزمان: {ctx.story_setting}")
        if ctx.story_problem:
            lines.append(f"المشكلة: {ctx.story_problem}")
        if ctx.story_solution:
            lines.append(f"الحل: {ctx.story_solution}")

    if ctx.materials:
        lines.append("الموادّ: " + "، ".join(ctx.materials))
    if ctx.time_estimate:
        lines.append(f"الوقت المقدّر: {ctx.time_estimate}")

    lines.append("")
    lines.append(ctx.content_text or "")
    lines.append("</السياق>")
    return "\n".join(lines)


_OVERSAMPLE_WEIGHTS = {
    TaskType.EXAM: 2.0,
    TaskType.INDICATOR_QUESTIONS: 2.0,
}


def to_sft_candidates(entry: dict, grade_level: int | None = None) -> list[SFTPromptCandidate]:
    ctx = adapt_entry(entry)
    task_types = infer_task_types(ctx)
    if not task_types:
        return []

    context_block = build_context_block(ctx, grade_level=grade_level)
    return [
        SFTPromptCandidate(
            task_type=t,
            context_block=context_block,
            source_entry_id=ctx.source_id,
            doc_id=ctx.doc_id,
            unit=ctx.unit,
            lesson=ctx.lesson,
            lesson_title=ctx.lesson_title,
            content_type=ctx.content_type,
            oversample_weight=_OVERSAMPLE_WEIGHTS.get(t, 1.0),
        )
        for t in task_types
    ]
