import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from compare_closed_book import compare


def _row(split, overlap):
    return {"split": split, "overlap_score": overlap}


def test_compare_reports_per_split_averages():
    base_rows = [_row("train_seen", 0.1), _row("val_heldout", 0.05)]
    tuned_rows = [_row("train_seen", 0.1), _row("val_heldout", 0.05)]

    result = compare(base_rows, tuned_rows)

    assert result["base_avg_overlap"]["train_seen"] == 0.1
    assert result["tuned_avg_overlap"]["val_heldout"] == 0.05


def test_compare_flags_memorization_gap_when_tuned_overlaps_more_on_seen_lessons():
    base_rows = [_row("train_seen", 0.1)]
    tuned_rows = [_row("train_seen", 0.6)]

    result = compare(base_rows, tuned_rows)

    assert result["memorization_gap_train_seen"] == 0.5


def test_compare_generalization_gap_near_zero_is_expected():
    base_rows = [_row("val_heldout", 0.08)]
    tuned_rows = [_row("val_heldout", 0.09)]

    result = compare(base_rows, tuned_rows)

    assert result["generalization_gap_val_heldout"] == 0.01


def test_compare_handles_missing_split_gracefully():
    base_rows = [_row("train_seen", 0.1)]
    tuned_rows = [_row("val_heldout", 0.1)]

    result = compare(base_rows, tuned_rows)

    assert result["memorization_gap_train_seen"] is None
    assert result["generalization_gap_val_heldout"] is None
