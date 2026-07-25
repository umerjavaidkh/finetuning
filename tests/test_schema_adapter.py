from datagen.schema_adapter import (
    TaskType,
    adapt_entry,
    build_context_block,
    infer_task_types,
    is_usable,
    to_sft_candidates,
)

LONG_CONTENT = "نص تعليمي تجريبي. " * 40


def make_entry(**overrides) -> dict:
    entry = {
        "id": "fixture_1",
        "DocId": "doc-fixture",
        "Page": 5,
        "BookPageNumber": "40",
        "Unit": "الوحدة 2",
        "Lesson": "الدرسُ 1",
        "UnitTitle": "وحدة تجريبية",
        "LessonTitle": "درس تجريبي",
        "Title": "",
        "Section": "Main Content",
        "ContentType": "exercise",
        "Topics": ["exercise"],
        "Skills": ["reading", "comprehension"],
        "Difficulty": "intermediate",
        "LearningObjectives": [],
        "Instructions": "",
        "Materials": [],
        "TimeEstimate": "",
        "Content": LONG_CONTENT,
        "Characters": [],
        "StoryElementsSetting": "",
        "StoryElementsProblem": "",
        "StoryElementsSolution": "",
        "StoryElementsMainEvents": [],
        "Answers": [],
    }
    entry.update(overrides)
    return entry


def test_unit_lesson_normalization_from_messy_arabic_text():
    ctx = adapt_entry(make_entry())
    assert ctx.unit == 2
    assert ctx.lesson == 1
    assert ctx.raw_unit == "الوحدة 2"
    assert ctx.raw_lesson == "الدرسُ 1"


def test_worksheet_inferred_from_exercise_with_instructions():
    entry = make_entry(Instructions="قسّم التلاميذ إلى مجموعات.", Materials=["أوراق"])
    ctx = adapt_entry(entry)
    types = infer_task_types(ctx)
    assert TaskType.WORKSHEET in types


def test_indicator_questions_inferred_from_learning_objectives():
    entry = make_entry(LearningObjectives=["أن يحدد التلميذ الفكرة الرئيسة."])
    types = infer_task_types(adapt_entry(entry))
    assert TaskType.INDICATOR_QUESTIONS in types


def test_exam_inferred_only_for_substantial_content():
    entry = make_entry(ContentType="objective", Content=LONG_CONTENT)
    types = infer_task_types(adapt_entry(entry))
    assert TaskType.EXAM in types

    short_entry = make_entry(ContentType="objective", Content="نص قصير جدا.")
    types_short = infer_task_types(adapt_entry(short_entry))
    assert TaskType.EXAM not in types_short


def test_story_comprehension_inferred_from_story_content_type():
    entry = make_entry(
        ContentType="story",
        Content=LONG_CONTENT,
        Characters=["سارة", "ليلى"],
        StoryElementsSetting="المدرسة",
        StoryElementsProblem="ضاعت الحقيبة",
        StoryElementsSolution="وجدتها في الصف",
    )
    types = infer_task_types(adapt_entry(entry))
    assert TaskType.STORY_COMPREHENSION in types


def test_vocab_activity_inferred_from_content_type_or_keywords():
    entry = make_entry(ContentType="vocabulary")
    assert TaskType.VOCAB_ACTIVITY in infer_task_types(adapt_entry(entry))

    keyword_entry = make_entry(Skills=["dictionary usage", "vocabulary building"])
    assert TaskType.VOCAB_ACTIVITY in infer_task_types(adapt_entry(keyword_entry))


def test_grammar_explanation_inferred_from_keywords():
    entry = make_entry(Skills=["قواعد اللغة"])
    assert TaskType.GRAMMAR_EXPLANATION in infer_task_types(adapt_entry(entry))


def test_exam_inferred_from_substantial_content_even_without_content_type():
    entry = make_entry(ContentType=None, Content=LONG_CONTENT)
    assert TaskType.EXAM in infer_task_types(adapt_entry(entry))


def test_exam_excluded_for_known_metadata_content_types_regardless_of_length():
    for excluded_type in ("header", "classification", "answer"):
        entry = make_entry(ContentType=excluded_type, Content=LONG_CONTENT)
        assert TaskType.EXAM not in infer_task_types(adapt_entry(entry))


def test_lesson_plan_inferred_from_instructions_without_content_type():
    entry = make_entry(ContentType=None, Instructions="اطلب إلى التلاميذ قراءة النص.")
    assert TaskType.LESSON_PLAN in infer_task_types(adapt_entry(entry))


def test_worksheet_requires_content_type_exercise_or_multistep_instructions():
    single_step = make_entry(ContentType=None, Instructions="اطلب إلى التلاميذ قراءة النص.")
    types_single = infer_task_types(adapt_entry(single_step))
    assert TaskType.LESSON_PLAN in types_single
    assert TaskType.WORKSHEET not in types_single

    multi_step = make_entry(
        ContentType=None,
        Instructions="اطلب إلى التلاميذ قراءة النص.; قسّم التلاميذ إلى مجموعات.",
    )
    assert TaskType.WORKSHEET in infer_task_types(adapt_entry(multi_step))


def test_junk_stub_with_short_content_yields_no_candidates():
    entry = make_entry(ContentType="header", Content="xvi", LearningObjectives=[])
    ctx = adapt_entry(entry)
    assert not is_usable(ctx)
    assert infer_task_types(ctx) == []
    assert to_sft_candidates(entry) == []


def test_context_block_includes_key_fields_and_is_wrapped():
    entry = make_entry(LearningObjectives=["هدف تجريبي"])
    ctx = adapt_entry(entry)
    block = build_context_block(ctx, grade_level=8)
    assert block.startswith("<السياق>")
    assert block.endswith("</السياق>")
    assert "المستوى: 8" in block
    assert "هدف تجريبي" in block
    assert LONG_CONTENT.strip() in block


def test_to_sft_candidates_oversamples_flagship_task_types():
    entry = make_entry(
        ContentType="objective",
        Content=LONG_CONTENT,
        LearningObjectives=["هدف تجريبي"],
    )
    candidates = to_sft_candidates(entry, grade_level=8)
    weights = {c.task_type: c.oversample_weight for c in candidates}
    assert weights[TaskType.EXAM] == 2.0
    assert weights[TaskType.INDICATOR_QUESTIONS] == 2.0
