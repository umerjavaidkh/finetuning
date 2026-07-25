import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datagen.generate import OpenAIChatClient
from datagen.judge import RUBRIC_CRITERIA, judge_pair


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _as_judge_pair(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": row["context"]},
            {"role": "assistant", "content": row["generated"]},
        ],
        "task_type": row["task_type"],
    }


def compare(base_rows: list[dict], tuned_rows: list[dict], client) -> dict:
    tuned_by_id = {r["source_entry_id"]: r for r in tuned_rows}
    per_pair = []
    base_totals = {k: [] for k in RUBRIC_CRITERIA}
    tuned_totals = {k: [] for k in RUBRIC_CRITERIA}
    wins = ties = losses = 0

    for base_row in base_rows:
        tuned_row = tuned_by_id.get(base_row["source_entry_id"])
        if tuned_row is None:
            continue

        base_result = judge_pair(_as_judge_pair(base_row), client)
        tuned_result = judge_pair(_as_judge_pair(tuned_row), client)

        for k in RUBRIC_CRITERIA:
            base_totals[k].append(base_result.scores[k])
            tuned_totals[k].append(tuned_result.scores[k])

        if tuned_result.average > base_result.average:
            wins += 1
        elif tuned_result.average < base_result.average:
            losses += 1
        else:
            ties += 1

        per_pair.append(
            {
                "source_entry_id": base_row["source_entry_id"],
                "task_type": base_row["task_type"],
                "base_average": base_result.average,
                "tuned_average": tuned_result.average,
            }
        )

    summary = {
        "pairs_compared": len(per_pair),
        "wins_tuned": wins,
        "ties": ties,
        "losses_tuned": losses,
        "per_criterion": {
            k: {
                "base_avg": round(sum(base_totals[k]) / len(base_totals[k]), 2) if base_totals[k] else None,
                "tuned_avg": round(sum(tuned_totals[k]) / len(tuned_totals[k]), 2) if tuned_totals[k] else None,
            }
            for k in RUBRIC_CRITERIA
        },
    }
    return {"summary": summary, "per_pair": per_pair}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge-based base vs fine-tuned comparison")
    parser.add_argument("--base", required=True, help="Path to base model outputs JSONL")
    parser.add_argument("--tuned", required=True, help="Path to fine-tuned model outputs JSONL")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="data/extracted/base_vs_tuned_report.json")
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    tuned_rows = load_jsonl(args.tuned)
    client = OpenAIChatClient(model=args.model)

    result = compare(base_rows, tuned_rows, client)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved full report to {args.output}")


if __name__ == "__main__":
    main()
