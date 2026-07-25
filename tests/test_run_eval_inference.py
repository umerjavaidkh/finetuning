import json

from training.run_eval_inference import load_val_prompts


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "val.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def test_load_val_prompts_reads_all_rows(tmp_path):
    rows = [{"task_type": "exam"}, {"task_type": "worksheet"}]
    path = _write_jsonl(tmp_path, rows)
    loaded = load_val_prompts(path)
    assert len(loaded) == 2
    assert loaded[0]["task_type"] == "exam"


def test_load_val_prompts_respects_limit(tmp_path):
    rows = [{"task_type": f"t{i}"} for i in range(10)]
    path = _write_jsonl(tmp_path, rows)
    loaded = load_val_prompts(path, limit=3)
    assert len(loaded) == 3
    assert loaded[-1]["task_type"] == "t2"


def test_load_val_prompts_no_limit_returns_everything(tmp_path):
    rows = [{"task_type": f"t{i}"} for i in range(5)]
    path = _write_jsonl(tmp_path, rows)
    loaded = load_val_prompts(path, limit=None)
    assert len(loaded) == 5
