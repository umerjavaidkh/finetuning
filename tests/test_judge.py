import json

from datagen.judge import (
    PASS_THRESHOLD,
    build_judge_prompt,
    judge_dataset,
    judge_pair,
)

GOOD_SCORES = {
    "language_correctness": 5,
    "curriculum_fidelity": 5,
    "structural_adherence": 4,
    "level_calibration": 4,
    "usability": 5,
    "notes": "جيد جدًا.",
}

SCORES_WITH_ISSUES = {
    "issues_found": ["جمع غير قياسي لكلمة ما"],
    "language_correctness": 4,
    "curriculum_fidelity": 5,
    "structural_adherence": 5,
    "level_calibration": 5,
    "usability": 5,
    "notes": "عيب لغوي طفيف واحد.",
}

BAD_SCORES = {
    "language_correctness": 2,
    "curriculum_fidelity": 2,
    "structural_adherence": 3,
    "level_calibration": 2,
    "usability": 2,
    "notes": "ضعيف.",
}


class FakeJudgeClient:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


def make_pair(task_type="exam", context="<السياق>\nنص\n</السياق>", output="امتحان تجريبي."):
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": context},
            {"role": "assistant", "content": output},
        ],
        "task_type": task_type,
        "source_entry_id": "entry_1",
        "doc_id": "doc-1",
        "unit": 1,
        "lesson": 3,
        "lesson_title": "درس تجريبي",
        "oversample_weight": 1.0,
    }


def test_build_judge_prompt_includes_context_output_and_task_type():
    prompt = build_judge_prompt("<السياق>نص</السياق>", "مخرج", "exam")
    assert "<السياق>نص</السياق>" in prompt
    assert "مخرج" in prompt
    assert "exam" in prompt
    assert "language_correctness" in prompt


def test_judge_pair_parses_plain_json_and_computes_average():
    client = FakeJudgeClient(json.dumps(GOOD_SCORES, ensure_ascii=False))
    result = judge_pair(make_pair(), client)

    assert result.scores["language_correctness"] == 5.0
    assert result.average == 4.6
    assert result.passed is True
    assert result.notes == "جيد جدًا."


def test_judge_pair_strips_markdown_code_fences():
    wrapped = "```json\n" + json.dumps(GOOD_SCORES, ensure_ascii=False) + "\n```"
    client = FakeJudgeClient(wrapped)
    result = judge_pair(make_pair(), client)
    assert result.average == 4.6


def test_judge_pair_below_threshold_fails():
    client = FakeJudgeClient(json.dumps(BAD_SCORES, ensure_ascii=False))
    result = judge_pair(make_pair(), client)
    assert result.average < PASS_THRESHOLD
    assert result.passed is False


def test_judge_pair_captures_issues_found_list():
    client = FakeJudgeClient(json.dumps(SCORES_WITH_ISSUES, ensure_ascii=False))
    result = judge_pair(make_pair(), client)
    assert result.issues_found == ["جمع غير قياسي لكلمة ما"]


def test_judge_pair_defaults_issues_found_to_empty_list_when_absent():
    client = FakeJudgeClient(json.dumps(GOOD_SCORES, ensure_ascii=False))
    result = judge_pair(make_pair(), client)
    assert result.issues_found == []


def test_judge_dataset_attaches_judge_field_to_accepted_pairs():
    pairs = [make_pair(task_type="exam"), make_pair(task_type="vocab_activity")]
    client = FakeJudgeClient(json.dumps(GOOD_SCORES, ensure_ascii=False))

    judged, failures = judge_dataset(pairs, client)

    assert len(judged) == 2
    assert len(failures) == 0
    assert judged[0]["judge"]["passed"] is True
    assert judged[0]["judge"]["average"] == 4.6
    assert judged[0]["task_type"] == "exam"


def test_judge_dataset_reports_malformed_json_as_failure():
    pairs = [make_pair()]
    client = FakeJudgeClient("not valid json at all")

    judged, failures = judge_dataset(pairs, client)

    assert len(judged) == 0
    assert len(failures) == 1
    assert failures[0]["source_entry_id"] == "entry_1"


def test_judge_pair_recovers_scores_via_regex_when_embedded_quote_breaks_json():
    broken_json = (
        '{"issues_found": ["استخدام كلمة "أضائع" غير صحيح"], '
        '"language_correctness": 3, "curriculum_fidelity": 5, '
        '"structural_adherence": 5, "level_calibration": 5, "usability": 5, '
        '"notes": "خطأ لغوي"}'
    )
    client = FakeJudgeClient(broken_json)
    result = judge_pair(make_pair(), client)

    assert result.scores["language_correctness"] == 3.0
    assert result.scores["usability"] == 5.0
    assert result.average == 4.6
