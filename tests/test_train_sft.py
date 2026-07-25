import json
from pathlib import Path

from training.train_sft import load_config, load_expanded_dataset

CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "sft.yaml")


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def test_load_config_reads_real_sft_yaml():
    config = load_config(CONFIG_PATH)
    assert config["model_name"] == "Qwen/Qwen3-8B"
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert set(config["lora"]["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    assert config["training"]["num_train_epochs"] == 2
    assert config["training"]["per_device_train_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 8


def test_load_expanded_dataset_duplicates_by_oversample_weight(tmp_path):
    rows = [
        {"messages": [{"role": "assistant", "content": "a"}], "oversample_weight": 1.0},
        {"messages": [{"role": "assistant", "content": "b"}], "oversample_weight": 2.0},
    ]
    path = _write_jsonl(tmp_path, rows)

    expanded = load_expanded_dataset(path)

    assert len(expanded) == 3
    assert sum(1 for r in expanded if r["messages"][0]["content"] == "a") == 1
    assert sum(1 for r in expanded if r["messages"][0]["content"] == "b") == 2


def test_load_expanded_dataset_defaults_missing_weight_to_one(tmp_path):
    rows = [{"messages": [{"role": "assistant", "content": "a"}]}]
    path = _write_jsonl(tmp_path, rows)
    expanded = load_expanded_dataset(path)
    assert len(expanded) == 1
