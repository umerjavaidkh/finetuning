# BilArabi Fine-Tuning

Production-grade LLM specialization pipeline for Arabic curriculum content generation. Part of a portfolio triad: RAG ([agentic_graph_rag](https://github.com/umerjavaidkh)) · Agents · **Fine-Tuning (this repo)**.

RAG retrieves curriculum context; this fine-tuned model generates in-style, indicator-aligned educational artifacts (exams, worksheets, lesson plans, vocabulary activities, grammar explanations) grounded in that context. See [`bilarabi-finetuning-blueprint.md`](bilarabi-finetuning-blueprint.md) for the full design.

## Status

- [x] Dataset-prep pipeline: PDF extraction → quality gating → SFT candidate building → teacher-model generation → LLM-judge scoring, 75 unit tests
- [x] Validated end-to-end on 2 real books → 593 quality-gated training pairs (69.5% and 66.1% judge-acceptance rates)
- [x] Training: QLoRA fine-tune of Qwen3-8B on Kaggle (T4×2), 2 epochs, 848 training examples — see [Training Run](#training-run) below
- [ ] Evaluation (base vs. fine-tuned comparison) — in progress
- [ ] Model + adapter published to Hugging Face

## Training Run

QLoRA (r=16) fine-tune of `Qwen/Qwen3-8B`, run on a Kaggle T4×2 kernel per `configs/sft.yaml`. 106 steps over 2 epochs, `train_loss` 1.15, held-out `eval_loss` improved epoch-over-epoch (1.209 → 1.192).

| Start | End |
|---|---|
| ![Training start](training_images/training_start.png) | ![Training end](training_images/training_end.png) |

Checkpointed every 20 steps for resumability; final adapter saved to `outputs/sft_run/`.

## Pipeline

```bash
# Extract + generate + judge one book
python scripts/prepare_dataset.py --pdf <book>.pdf --grade-level <N> --generate --judge

# Combine accepted pairs from multiple books, held-out-by-lesson split for eval
python scripts/build_training_dataset.py \
  --book NAME1 path/to/book1_accepted.jsonl \
  --book NAME2 path/to/book2_accepted.jsonl \
  --held-out BOOK UNIT LESSON

# Train (run on a GPU, e.g. a Kaggle Script kernel)
python src/training/train_sft.py --config configs/sft.yaml
```

## Tests

```bash
python -m pytest tests/ -v
```

## IP note

Source curriculum PDFs and all extracted/generated data are excluded from this repo (see `.gitignore`). Only pipeline code is public — no book text, no generated training pairs.
