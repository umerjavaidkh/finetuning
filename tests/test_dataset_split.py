import json

from datagen.dataset_split import combine_accepted_datasets, held_out_lesson_split, load_jsonl


def _write_jsonl(tmp_path, name, rows):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _pair(unit, lesson, task_type="exam"):
    return {
        "messages": [{"role": "assistant", "content": "نص"}],
        "task_type": task_type,
        "unit": unit,
        "lesson": lesson,
    }


def test_load_jsonl_roundtrips(tmp_path):
    path = _write_jsonl(tmp_path, "a.jsonl", [_pair(1, 1), _pair(1, 2)])
    rows = load_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["unit"] == 1


def test_combine_accepted_datasets_tags_each_pair_with_its_book(tmp_path):
    path_a = _write_jsonl(tmp_path, "book_a.jsonl", [_pair(1, 1)])
    path_b = _write_jsonl(tmp_path, "book_b.jsonl", [_pair(2, 3), _pair(2, 4)])

    combined = combine_accepted_datasets({"BookA": path_a, "BookB": path_b})

    assert len(combined) == 3
    assert combined[0]["book"] == "BookA"
    assert combined[1]["book"] == "BookB"
    assert combined[2]["book"] == "BookB"


def test_held_out_lesson_split_excludes_specified_lessons_entirely():
    pairs = [
        {**_pair(1, 1), "book": "BookA"},
        {**_pair(1, 1), "book": "BookA"},
        {**_pair(1, 2), "book": "BookA"},
        {**_pair(2, 1), "book": "BookB"},
    ]

    train, val = held_out_lesson_split(pairs, held_out_lessons=[("BookA", 1, 1)])

    assert len(train) == 2
    assert len(val) == 2
    assert all((p["book"], p["unit"], p["lesson"]) != ("BookA", 1, 1) for p in train)
    assert all((p["book"], p["unit"], p["lesson"]) == ("BookA", 1, 1) for p in val)


def test_held_out_lesson_split_handles_missing_unit_lesson_gracefully():
    pairs = [
        {**_pair(None, None), "book": "BookA"},
        {**_pair(1, 1), "book": "BookA"},
    ]
    train, val = held_out_lesson_split(pairs, held_out_lessons=[("BookA", 1, 1)])
    assert len(train) == 1
    assert len(val) == 1
