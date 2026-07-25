import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datagen.dataset_split import combine_accepted_datasets, held_out_lesson_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-book accepted datasets into train/val splits")
    parser.add_argument(
        "--book",
        action="append",
        nargs=2,
        metavar=("NAME", "ACCEPTED_JSONL_PATH"),
        required=True,
        help="Repeatable: a book name and its *_accepted.jsonl path",
    )
    parser.add_argument(
        "--held-out",
        action="append",
        nargs=3,
        metavar=("BOOK", "UNIT", "LESSON"),
        default=[],
        help="Repeatable: (book, unit, lesson) to hold out entirely for eval",
    )
    parser.add_argument("--out-dir", default="data/extracted")
    args = parser.parse_args()

    book_paths = {name: path for name, path in args.book}
    held_out_lessons = [(book, int(unit), int(lesson)) for book, unit, lesson in args.held_out]

    combined = combine_accepted_datasets(book_paths)
    train, val = held_out_lesson_split(combined, held_out_lessons)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for pair in train:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for pair in val:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Combined {len(combined)} pairs from {len(book_paths)} book(s)")
    print(f"Held out {len(held_out_lessons)} lesson(s): {held_out_lessons}")
    print(f"Saved {len(train)} train pairs to {train_path}")
    print(f"Saved {len(val)} val (held-out) pairs to {val_path}")


if __name__ == "__main__":
    main()
