# BilArabi Fine-Tuning

Production-grade LLM specialization pipeline for Arabic curriculum content generation. Part of a portfolio triad: RAG ([agentic_graph_rag](https://github.com/umerjavaidkh)) · Agents · **Fine-Tuning (this repo)**.

RAG retrieves curriculum context; this fine-tuned model generates in-style, indicator-aligned educational artifacts (exams, worksheets, lesson plans, vocabulary activities, grammar explanations) grounded in that context. See [`bilarabi-finetuning-blueprint.md`](bilarabi-finetuning-blueprint.md) for the full design.

## Status

- [x] Dataset-prep pipeline: PDF extraction → quality gating → SFT candidate building → teacher-model generation → LLM-judge scoring, 75 unit tests
- [x] Validated end-to-end on 2 real books → 593 quality-gated training pairs (69.5% and 66.1% judge-acceptance rates)
- [ ] Training (QLoRA config ready in `configs/sft.yaml`, not yet run — needs a GPU)
- [ ] Evaluation (base vs. fine-tuned comparison)
- [ ] Model + adapter published to Hugging Face

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
