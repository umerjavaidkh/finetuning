import argparse
import json
import os
import re

import yaml

from datagen.generate import _TASK_INSTRUCTIONS
from datagen.schema_adapter import TaskType


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rows(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def build_closed_book_prompt(row: dict) -> str:
    """Same task instruction as training, but only the unit/lesson label — no
    learning objectives, no content text. Tests whether the model can produce
    lesson-specific detail it was never given in the prompt."""
    task_type = TaskType(row["task_type"])
    instruction = _TASK_INSTRUCTIONS[task_type]
    lines = ["<السياق>"]
    if row.get("unit") is not None:
        lines.append(f"الوحدة: {row['unit']}")
    if row.get("lesson") is not None:
        lesson_line = f"الدرس: {row['lesson']}"
        if row.get("lesson_title"):
            lesson_line += f" — {row['lesson_title']}"
        lines.append(lesson_line)
    lines.append("</السياق>")
    return f"{instruction}\n\n" + "\n".join(lines)


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2}


def overlap_score(generated: str, reference: str) -> float:
    """Fraction of the generated text's distinct words that also appear in the
    reference (the real lesson content/target). A crude memorization proxy:
    closed-book output that overlaps heavily with content it was never given
    is more likely recalled from training than invented."""
    gen_words = _words(generated)
    ref_words = _words(reference)
    if not gen_words or not ref_words:
        return 0.0
    return len(gen_words & ref_words) / len(gen_words)


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


def run_closed_book_eval(
    config: dict,
    model_path: str,
    rows_by_split: dict[str, list[dict]],
    max_new_tokens: int,
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

    system_msg = {"role": "system", "content": ""}
    results = []
    for split_label, rows in rows_by_split.items():
        for row in rows:
            closed_book_prompt = build_closed_book_prompt(row)
            user_msg = {"role": "user", "content": closed_book_prompt}
            generated = _generate_one(model, tokenizer, [system_msg, user_msg], max_new_tokens)
            reference = row["messages"][2]["content"] if len(row["messages"]) > 2 else ""
            results.append(
                {
                    "source_entry_id": row.get("source_entry_id"),
                    "task_type": row.get("task_type"),
                    "unit": row.get("unit"),
                    "lesson": row.get("lesson"),
                    "split": split_label,
                    "closed_book_prompt": closed_book_prompt,
                    "generated": generated,
                    "reference": reference,
                    "overlap_score": round(overlap_score(generated, reference), 4),
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-book eval: ask for a specific lesson's content with no "
            "curriculum context given, to check hallucination vs memorization"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF repo id for the base model, or a local adapter directory for the fine-tuned model",
    )
    parser.add_argument("--train-path", required=True, help="Rows the model WAS trained on")
    parser.add_argument("--val-path", required=True, help="Held-out rows the model NEVER saw")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    args = parser.parse_args()

    config = load_config(args.config)
    rows_by_split = {
        "train_seen": load_rows(args.train_path, limit=args.sample_size),
        "val_heldout": load_rows(args.val_path, limit=args.sample_size),
    }
    results = run_closed_book_eval(config, args.model_path, rows_by_split, args.max_new_tokens)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} closed-book generations from {args.model_path} to {args.output}")


if __name__ == "__main__":
    main()
