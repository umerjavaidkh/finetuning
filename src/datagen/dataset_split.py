from __future__ import annotations

import json


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def combine_accepted_datasets(book_paths: dict[str, str]) -> list[dict]:
    combined: list[dict] = []
    for book_name, path in book_paths.items():
        pairs = load_jsonl(path)
        for pair in pairs:
            tagged = dict(pair)
            tagged["book"] = book_name
            combined.append(tagged)
    return combined


def held_out_lesson_split(
    pairs: list[dict], held_out_lessons: list[tuple[str, int, int]]
) -> tuple[list[dict], list[dict]]:
    held_out_keys = {(book, unit, lesson) for book, unit, lesson in held_out_lessons}

    train: list[dict] = []
    val: list[dict] = []
    for pair in pairs:
        key = (pair.get("book"), pair.get("unit"), pair.get("lesson"))
        if key in held_out_keys:
            val.append(pair)
        else:
            train.append(pair)
    return train, val
