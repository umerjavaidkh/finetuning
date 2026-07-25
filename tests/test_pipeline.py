import json

from datagen.pipeline import (
    candidate_to_dict,
    run_dataset_generation,
    run_extraction,
    run_judge_stage,
    write_jsonl,
)
from datagen.schema_adapter import SFTPromptCandidate, TaskType

LONG_CONTENT = "نص تعليمي تجريبي يحتوي على معلومات حقيقية عن الدرس. " * 20


def _quality(passed=True, confidence=1.0, hard_failures=None, soft_warnings=None):
    return {
        "passed": passed,
        "confidence": confidence,
        "hard_failures": hard_failures or [],
        "soft_warnings": soft_warnings or [],
        "metrics": {},
    }


def _entry(**overrides):
    entry = {
        "id": "doc-1_1",
        "DocId": "doc-1",
        "Page": 1,
        "ContentType": None,
        "Content": LONG_CONTENT,
        "LearningObjectives": ["أن يتعرّف التلميذ على معنى الكلمة الجديدة."],
        "Instructions": None,
        "Unit": None,
        "Lesson": None,
        "UnitTitle": None,
        "LessonTitle": None,
        "Topics": [],
        "Skills": [],
        "Difficulty": None,
        "Materials": [],
        "TimeEstimate": None,
        "Characters": [],
        "Answers": [],
        "_extraction_quality": _quality(),
    }
    entry.update(overrides)
    return entry


class FakeClient:
    def generate(self, system: str, user: str) -> str:
        return "مخرج تجريبي."


def test_run_extraction_aggregates_quality_and_builds_candidates_only_from_usable_pages(monkeypatch):
    entries = [
        _entry(id="doc-1_1", Page=1),
        _entry(
            id="doc-1_2",
            Page=2,
            Content="xvi",
            LearningObjectives=[],
            _extraction_quality=_quality(passed=False, confidence=0.0, hard_failures=["too_short"]),
        ),
        _entry(
            id="doc-1_3",
            Page=3,
            LearningObjectives=[],
            _extraction_quality=_quality(passed=True, confidence=0.85, soft_warnings=["excessive_repetition"]),
        ),
    ]
    monkeypatch.setattr("datagen.pipeline.extract_pdf", lambda pdf_path, doc_id=None: entries)

    returned_entries, candidates, report = run_extraction("fake.pdf", grade_level=8)

    assert returned_entries == entries
    assert report.total_pages == 3
    assert report.usable_pages == 2
    assert report.failed_pages == 1
    assert report.hard_failure_counts == {"too_short": 1}
    assert report.soft_warning_counts == {"excessive_repetition": 1}
    assert report.doc_id == "doc-1"
    # only page 1 (has LearningObjectives) yields an indicator_questions candidate;
    # page 3 is usable but has no learning objectives and is short of the exam threshold
    assert report.candidate_count == len(candidates)
    assert all(c.source_entry_id != "doc-1_2" for c in candidates)


def test_run_dataset_generation_respects_limit_and_reports_counts():
    candidates = [
        SFTPromptCandidate(
            task_type=TaskType.EXAM,
            context_block="<السياق>\nنص\n</السياق>",
            source_entry_id=f"entry_{i}",
            doc_id="doc-1",
            unit=None,
            lesson=None,
            lesson_title=None,
            content_type=None,
        )
        for i in range(5)
    ]

    results, failures, report = run_dataset_generation(candidates, FakeClient(), limit=2)

    assert len(results) == 2
    assert len(failures) == 0
    assert report.candidates_submitted == 2
    assert report.pairs_generated == 2
    assert report.pairs_failed == 0
    assert report.generated_at


def test_write_jsonl_roundtrips_pairs_with_unescaped_arabic(tmp_path):
    pairs = [
        {"messages": [{"role": "assistant", "content": "نص عربي تجريبي"}], "task_type": "exam"},
        {"messages": [{"role": "assistant", "content": "نص آخر"}], "task_type": "vocab_activity"},
    ]
    output_path = tmp_path / "dataset.jsonl"

    write_jsonl(pairs, str(output_path))

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "نص عربي تجريبي" in lines[0]
    assert json.loads(lines[0]) == pairs[0]
    assert json.loads(lines[1]) == pairs[1]


def test_candidate_to_dict_is_json_serializable_with_string_task_type(tmp_path):
    candidate = SFTPromptCandidate(
        task_type=TaskType.EXAM,
        context_block="<السياق>\nنص\n</السياق>",
        source_entry_id="entry_1",
        doc_id="doc-1",
        unit=1,
        lesson=3,
        lesson_title="درس تجريبي",
        content_type="objective",
        oversample_weight=2.0,
    )

    d = candidate_to_dict(candidate)

    assert d["task_type"] == "exam"
    assert d["source_entry_id"] == "entry_1"
    output_path = tmp_path / "candidates.jsonl"
    write_jsonl([d], str(output_path))
    roundtripped = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert roundtripped == d


class FakeJudgeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return response


def _sft_pair(source_entry_id="entry_1"):
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<السياق>\nنص\n</السياق>"},
            {"role": "assistant", "content": "مخرج تجريبي."},
        ],
        "task_type": "exam",
        "source_entry_id": source_entry_id,
        "doc_id": "doc-1",
        "unit": 1,
        "lesson": 3,
        "lesson_title": "درس تجريبي",
        "oversample_weight": 1.0,
    }


def test_run_judge_stage_splits_accepted_and_rejected_and_reports_funnel():
    good = json.dumps(
        {
            "language_correctness": 5, "curriculum_fidelity": 5,
            "structural_adherence": 5, "level_calibration": 5, "usability": 5,
            "notes": "ممتاز",
        },
        ensure_ascii=False,
    )
    bad = json.dumps(
        {
            "language_correctness": 2, "curriculum_fidelity": 2,
            "structural_adherence": 2, "level_calibration": 2, "usability": 2,
            "notes": "ضعيف",
        },
        ensure_ascii=False,
    )
    pairs = [_sft_pair("entry_1"), _sft_pair("entry_2"), _sft_pair("entry_3")]
    client = FakeJudgeClient([good, bad, "not valid json"])

    judged, failures, report = run_judge_stage(pairs, client)

    assert len(judged) == 2
    assert len(failures) == 1
    assert report.pairs_submitted == 3
    assert report.pairs_judged == 2
    assert report.pairs_accepted == 1
    assert report.pairs_rejected == 1
    assert report.pairs_failed == 1
    assert report.acceptance_rate == round(1 / 3, 3)
