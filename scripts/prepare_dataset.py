import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datagen.generate import OpenAIChatClient
from datagen.pipeline import (
    candidate_to_dict,
    run_dataset_generation,
    run_extraction,
    run_judge_stage,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arabic curriculum dataset preparation pipeline")
    parser.add_argument("--pdf", required=True, help="Path to a source book PDF")
    parser.add_argument("--grade-level", type=int, default=None)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also call the teacher model to generate SFT pairs (real API cost)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Also score generated pairs with the judge rubric (real API cost). "
        "Judges the pairs just generated, or an existing *_sft.jsonl if --generate wasn't passed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of candidates/pairs sent to the teacher/judge model",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out-dir", default="data/extracted")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.pdf))[0]

    entries, candidates, extraction_report = run_extraction(
        args.pdf, doc_id=args.doc_id, grade_level=args.grade_level
    )

    report_path = os.path.join(args.out_dir, f"{base_name}_extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(extraction_report.to_dict(), f, ensure_ascii=False, indent=2)

    pages_path = os.path.join(args.out_dir, f"{base_name}_pages.jsonl")
    write_jsonl(entries, pages_path)

    usable_entries = [e for e in entries if e["_extraction_quality"]["passed"]]
    usable_pages_path = os.path.join(args.out_dir, f"{base_name}_pages_usable.jsonl")
    write_jsonl(usable_entries, usable_pages_path)

    candidates_path = os.path.join(args.out_dir, f"{base_name}_candidates.jsonl")
    write_jsonl([candidate_to_dict(c) for c in candidates], candidates_path)

    print("=== Extraction report ===")
    print(json.dumps(extraction_report.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nSaved extraction report to {report_path}")
    print(f"Saved {len(entries)} extracted pages (all) to {pages_path}")
    print(f"Saved {len(usable_entries)} usable pages to {usable_pages_path}")
    print(f"Saved {len(candidates)} candidates to {candidates_path}")

    dataset_path = os.path.join(args.out_dir, f"{base_name}_sft.jsonl")

    if not args.generate and not args.judge:
        print(
            "\nExtraction-only run (no teacher-model calls, no cost). "
            "Re-run with --generate to produce the SFT dataset, or --judge to score an existing one."
        )
        return

    results = None
    if args.generate:
        client = OpenAIChatClient(model=args.model)
        results, failures, generation_report = run_dataset_generation(
            candidates, client, limit=args.limit
        )
        write_jsonl(results, dataset_path)

        failures_path = os.path.join(args.out_dir, f"{base_name}_failures.json")
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

        print("\n=== Dataset generation report ===")
        print(json.dumps(generation_report.to_dict(), ensure_ascii=False, indent=2))
        print(f"\nSaved {len(results)} SFT pairs to {dataset_path}")
        if failures:
            print(f"Saved {len(failures)} failures to {failures_path}")

    if not args.judge:
        return

    if results is None:
        if not os.path.exists(dataset_path):
            print(f"\nNo pairs to judge: {dataset_path} does not exist. Run with --generate first.")
            return
        with open(dataset_path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

    judge_client = OpenAIChatClient(model=args.model)
    judged, judge_failures, judge_report = run_judge_stage(results, judge_client, limit=args.limit)
    accepted = [p for p in judged if p["judge"]["passed"]]
    rejected = [p for p in judged if not p["judge"]["passed"]]

    judged_path = os.path.join(args.out_dir, f"{base_name}_judged.jsonl")
    write_jsonl(judged, judged_path)

    accepted_path = os.path.join(args.out_dir, f"{base_name}_accepted.jsonl")
    write_jsonl(accepted, accepted_path)

    judge_failures_path = os.path.join(args.out_dir, f"{base_name}_judge_failures.json")
    with open(judge_failures_path, "w", encoding="utf-8") as f:
        json.dump(judge_failures, f, ensure_ascii=False, indent=2)

    print("\n=== Judge report ===")
    print(json.dumps(judge_report.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nSaved {len(judged)} judged pairs to {judged_path}")
    print(f"Saved {len(accepted)} accepted (training-ready) pairs to {accepted_path}")
    print(f"Rejected {len(rejected)} pairs below threshold")
    if judge_failures:
        print(f"Saved {len(judge_failures)} judge failures to {judge_failures_path}")


if __name__ == "__main__":
    main()
