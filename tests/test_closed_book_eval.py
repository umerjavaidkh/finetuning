from training.closed_book_eval import build_closed_book_prompt, overlap_score


def test_build_closed_book_prompt_omits_content_and_objectives():
    row = {
        "task_type": "exam",
        "unit": 2,
        "lesson": 4,
        "lesson_title": "الهوية",
    }
    prompt = build_closed_book_prompt(row)

    assert "الوحدة: 2" in prompt
    assert "الدرس: 4" in prompt
    assert "الهوية" in prompt
    assert "مؤشرات التعلّم" not in prompt


def test_build_closed_book_prompt_uses_task_specific_instruction():
    exam_prompt = build_closed_book_prompt({"task_type": "exam", "unit": 1, "lesson": 1})
    worksheet_prompt = build_closed_book_prompt({"task_type": "worksheet", "unit": 1, "lesson": 1})

    assert exam_prompt != worksheet_prompt
    assert "امتحانًا" in exam_prompt
    assert "ورقة عمل" in worksheet_prompt


def test_build_closed_book_prompt_handles_missing_unit_lesson():
    prompt = build_closed_book_prompt({"task_type": "exam"})

    assert "<السياق>" in prompt
    assert "</السياق>" in prompt


def test_overlap_score_full_match_is_one():
    text = "القراءة والفهم مهارة أساسية في اللغة العربية"
    assert overlap_score(text, text) == 1.0


def test_overlap_score_no_shared_words_is_zero():
    assert overlap_score("قطة صغيرة تلعب", "سيارة حمراء سريعة") == 0.0


def test_overlap_score_partial_match_between_zero_and_one():
    generated = "الدرس يتناول القراءة والفهم والاستيعاب"
    reference = "القراءة مهارة مهمة جدا في اللغة"
    score = overlap_score(generated, reference)
    assert 0.0 < score < 1.0


def test_overlap_score_empty_generated_is_zero():
    assert overlap_score("", "أي نص مرجعي هنا") == 0.0
