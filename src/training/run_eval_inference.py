import argparse
import json
import os

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_val_prompts(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def _generate_one(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_eval_inference(
    config: dict, model_path: str, val_rows: list[dict], max_new_tokens: int
) -> list[dict]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config["max_seq_length"],
        dtype=config.get("dtype"),
        load_in_4bit=config["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)

    results = []
    for row in val_rows:
        system_msg, user_msg = row["messages"][0], row["messages"][1]
        generated = _generate_one(model, tokenizer, [system_msg, user_msg], max_new_tokens)
        results.append(
            {
                "source_entry_id": row.get("source_entry_id"),
                "task_type": row.get("task_type"),
                "unit": row.get("unit"),
                "lesson": row.get("lesson"),
                "context": user_msg["content"],
                "reference": row["messages"][2]["content"] if len(row["messages"]) > 2 else None,
                "generated": generated,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base or fine-tuned outputs on the held-out val set")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF repo id for the base model, or a local adapter directory for the fine-tuned model",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    args = parser.parse_args()

    config = load_config(args.config)
    val_rows = load_val_prompts(config["data"]["val_path"], limit=args.limit)
    results = run_eval_inference(config, args.model_path, val_rows, args.max_new_tokens)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} generations from {args.model_path} to {args.output}")


if __name__ == "__main__":
    main()
