import argparse
import json
import os

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_expanded_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            repeats = max(1, round(row.get("oversample_weight", 1.0)))
            rows.extend([row] * repeats)
    return rows


def _build_text_dataset(rows, tokenizer):
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    return Dataset.from_dict({"text": texts})


def run_training(config: dict, resume_from_checkpoint: str | None = None) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=config.get("dtype"),
        load_in_4bit=config["load_in_4bit"],
        token=os.environ.get("HF_TOKEN"),
    )

    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
    )

    train_rows = load_expanded_dataset(config["data"]["train_path"])
    val_rows = load_expanded_dataset(config["data"]["val_path"]) if config["data"].get("val_path") else []

    train_dataset = _build_text_dataset(train_rows, tokenizer)
    eval_dataset = _build_text_dataset(val_rows, tokenizer) if val_rows else None

    t = config["training"]
    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if wandb_cfg.get("enabled") else "none"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "bilarabi-finetune"))

    import torch

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = SFTConfig(
        output_dir=t["output_dir"],
        dataset_text_field="text",
        max_length=config["max_seq_length"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", t["per_device_train_batch_size"]),
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_train_epochs=t["num_train_epochs"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        gradient_checkpointing=t["gradient_checkpointing"],
        save_strategy=t["save_strategy"],
        logging_steps=t["logging_steps"],
        seed=t["seed"],
        report_to=report_to,
        run_name=wandb_cfg.get("run_name") or os.path.basename(t["output_dir"]),
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=bf16_supported,
        fp16=not bf16_supported,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="BilArabi QLoRA SFT training (Unsloth + TRL)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
