import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize(rows: list[dict]) -> dict:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row["overlap_score"])
    return {
        split: round(sum(scores) / len(scores), 4) if scores else None
        for split, scores in by_split.items()
    }


def compare(base_rows: list[dict], tuned_rows: list[dict]) -> dict:
    base_summary = summarize(base_rows)
    tuned_summary = summarize(tuned_rows)

    memorization_gap = None
    if base_summary.get("train_seen") is not None and tuned_summary.get("train_seen") is not None:
        memorization_gap = round(tuned_summary["train_seen"] - base_summary["train_seen"], 4)

    generalization_gap = None
    if base_summary.get("val_heldout") is not None and tuned_summary.get("val_heldout") is not None:
        generalization_gap = round(tuned_summary["val_heldout"] - base_summary["val_heldout"], 4)

    return {
        "base_avg_overlap": base_summary,
        "tuned_avg_overlap": tuned_summary,
        # tuned - base, on lessons the tuned model WAS trained on.
        # Large positive = tuned model recalls specific content it was never given -> memorization.
        "memorization_gap_train_seen": memorization_gap,
        # tuned - base, on held-out lessons NEVER seen in training.
        # Should stay near zero: fine-tuning isn't meant to inject new knowledge (RAG's job).
        "generalization_gap_val_heldout": generalization_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize closed-book eval: memorization vs generalization, tuned vs base"
    )
    parser.add_argument("--base", required=True, help="Path to base model closed-book outputs JSONL")
    parser.add_argument("--tuned", required=True, help="Path to fine-tuned model closed-book outputs JSONL")
    parser.add_argument("--output", default="data/extracted/closed_book_report.json")
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    tuned_rows = load_jsonl(args.tuned)
    result = compare(base_rows, tuned_rows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
