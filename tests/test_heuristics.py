from datagen.heuristics import (
    extract_heuristic_fields,
    extract_instructions,
    extract_learning_objectives,
    extract_lesson,
    extract_time_estimate,
    extract_unit,
)


def test_extracts_single_objective_sentence():
    text = "مقدمة عامة عن الدرس. أن يتعرّف التلميذ على معاني المفردات الجديدة. نهاية الفقرة."
    objectives = extract_learning_objectives(text)
    assert len(objectives) == 1
    assert objectives[0].startswith("أن يتعرّف")


def test_extracts_multiple_distinct_objectives_and_dedupes():
    text = (
        "أن يحدد التلميذ الفكرة الرئيسة للنص. "
        "أن يستنتج التلميذ معنى الكلمة من السياق. "
        "أن يحدد التلميذ الفكرة الرئيسة للنص. "
    )
    objectives = extract_learning_objectives(text)
    assert len(objectives) == 2


def test_no_objective_returns_empty_list():
    assert extract_learning_objectives("نص عام لا يحتوي على أي هدف تعليمي واضح هنا.") == []
    assert extract_learning_objectives("") == []


def test_time_estimate_western_digits_with_parens():
    text = "نشاط جماعي (15 دقيقة) يتضمن نقاشا حول الموضوع."
    assert extract_time_estimate(text) == "15 دقائق"


def test_time_estimate_bidi_reversed_parens_from_pdf_extraction():
    text = "دقيقة)12(  نشاط جماعي يتضمن نقاشا حول الموضوع."
    assert extract_time_estimate(text) == "12 دقائق"


def test_time_estimate_arabic_indic_digits():
    text = "نشاط قصير ٥ دقائق فقط للمراجعة."
    assert extract_time_estimate(text) == "5 دقائق"


def test_no_time_estimate_returns_none():
    assert extract_time_estimate("نص عام بدون أي إشارة إلى الوقت.") is None


def test_extracts_instruction_sentence_starting_with_verb():
    text = "اطلب إلى التلاميذ قراءة النص بصوت عال. ثم ناقش معهم الأفكار الرئيسة."
    instructions = extract_instructions(text)
    assert any(s.startswith("اطلب") for s in instructions)
    assert any(s.startswith("ناقش") for s in instructions)


def test_extract_heuristic_fields_returns_expected_keys():
    text = (
        "أن يتعرّف التلميذ على معاني المفردات الجديدة. "
        "اطلب إلى التلاميذ العمل في مجموعات (10 دقائق)."
    )
    result = extract_heuristic_fields(text)
    assert set(result.keys()) == {
        "LearningObjectives", "TimeEstimate", "Instructions", "Unit", "Lesson"
    }
    assert len(result["LearningObjectives"]) == 1
    assert result["TimeEstimate"] == "10 دقائق"
    assert result["Instructions"] is not None


def test_extract_unit_reversed_order_marginal_label():
    assert extract_unit("نص عام\n1 الوحدة\nمزيد من النص") == "1"


def test_extract_unit_forward_order():
    assert extract_unit("نص عام الوحدة 2 مزيد من النص") == "2"


def test_extract_lesson_reversed_order_marginal_label():
    assert extract_lesson("نص عام\n4 الدرس\nمزيد من النص") == "4"


def test_extract_unit_or_lesson_rejects_implausible_adjacent_values():
    assert extract_lesson("نص عام 34 الدرس نص آخر") is None
    assert extract_unit("رقم 50 الوحدة غير معقول هنا") is None


def test_extract_unit_and_lesson_return_none_when_absent():
    assert extract_unit("نص عام لا يحتوي على أي إشارة إلى الوحدة.") is None
    assert extract_lesson("") is None
