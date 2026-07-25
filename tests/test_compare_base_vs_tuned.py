import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_base_vs_tuned import compare

GOOD_SCORES = {
    "language_correctness": 5, "curriculum_fidelity": 5,
    "structural_adherence": 5, "level_calibration": 5, "usability": 5,
    "notes": "ممتاز",
}
WEAK_SCORES = {
    "language_correctness": 2, "curriculum_fidelity": 2,
    "structural_adherence": 2, "level_calibration": 2, "usability": 2,
    "notes": "ضعيف",
}


class FakeClient:
    def __init__(self, responses_by_call_order):
        self.responses = responses_by_call_order
        self.calls = 0

    def generate(self, system, user):
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return json.dumps(response, ensure_ascii=False)


def _row(entry_id, task_type="exam"):
    return {
        "source_entry_id": entry_id,
        "task_type": task_type,
        "context": "<السياق>\nنص\n</السياق>",
        "generated": "مخرج تجريبي",
    }


def test_compare_counts_tuned_win_when_tuned_scores_higher():
    base_rows = [_row("e1")]
    tuned_rows = [_row("e1")]
    client = FakeClient([WEAK_SCORES, GOOD_SCORES])

    result = compare(base_rows, tuned_rows, client)

    assert result["summary"]["pairs_compared"] == 1
    assert result["summary"]["wins_tuned"] == 1
    assert result["summary"]["losses_tuned"] == 0
    assert result["summary"]["ties"] == 0


def test_compare_counts_tie_when_scores_equal():
    base_rows = [_row("e1")]
    tuned_rows = [_row("e1")]
    client = FakeClient([GOOD_SCORES, GOOD_SCORES])

    result = compare(base_rows, tuned_rows, client)

    assert result["summary"]["ties"] == 1


def test_compare_skips_unmatched_entries():
    base_rows = [_row("e1"), _row("e2")]
    tuned_rows = [_row("e1")]
    client = FakeClient([GOOD_SCORES, GOOD_SCORES])

    result = compare(base_rows, tuned_rows, client)

    assert result["summary"]["pairs_compared"] == 1


def test_compare_reports_per_criterion_averages():
    base_rows = [_row("e1")]
    tuned_rows = [_row("e1")]
    client = FakeClient([WEAK_SCORES, GOOD_SCORES])

    result = compare(base_rows, tuned_rows, client)

    per_crit = result["summary"]["per_criterion"]["language_correctness"]
    assert per_crit["base_avg"] == 2.0
    assert per_crit["tuned_avg"] == 5.0
