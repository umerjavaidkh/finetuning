# BilArabi Fine-Tuning

Production-grade LLM specialization pipeline for Arabic curriculum content generation. Part of a portfolio triad: RAG ([agentic_graph_rag](https://github.com/umerjavaidkh)) · Agents · **Fine-Tuning (this repo)**.

RAG retrieves curriculum context; this fine-tuned model generates in-style, indicator-aligned educational artifacts (exams, worksheets, lesson plans, vocabulary activities, grammar explanations) grounded in that context. See [`bilarabi-finetuning-blueprint.md`](bilarabi-finetuning-blueprint.md) for the full design.

## Status

- [x] Dataset-prep pipeline: PDF extraction → quality gating → SFT candidate building → teacher-model generation → LLM-judge scoring, 75 unit tests
- [x] Validated end-to-end on 2 real books → 593 quality-gated training pairs (69.5% and 66.1% judge-acceptance rates)
- [x] Training: QLoRA fine-tune of Qwen3-8B on Kaggle (T4×2), 2 epochs, 848 training examples — see [Training Run](#training-run) below
- [x] Evaluation: base vs. fine-tuned, judge-scored on 20 held-out prompts — see [Evaluation](#evaluation) below
- [ ] Model + adapter published to Hugging Face

## Training Run

QLoRA (r=16) fine-tune of `Qwen/Qwen3-8B`, run on a Kaggle T4×2 kernel per `configs/sft.yaml`. 106 steps over 2 epochs, `train_loss` 1.15, held-out `eval_loss` improved epoch-over-epoch (1.209 → 1.192).

| Start | End |
|---|---|
| ![Training start](training_images/training_start.png) | ![Training end](training_images/training_end.png) |

Checkpointed every 20 steps for resumability; final adapter saved to `outputs/sft_run/`.

## Evaluation

Base `Qwen/Qwen3-8B` vs. the fine-tuned adapter, both generating on the same 20 held-out prompts (excluded from training via the held-out-by-lesson split), judged blind on the same 5-criterion rubric used to filter the training data (`eval/rubric.md`).

| Criterion | Base | Fine-tuned | Δ |
|---|---|---|---|
| Language correctness | 4.00 | 3.35 | −0.65 |
| Curriculum fidelity | 4.50 | 5.00 | +0.50 |
| Structural adherence | 3.25 | 3.80 | +0.55 |
| Level calibration | 4.65 | 4.50 | −0.15 |
| Usability | 4.15 | 4.30 | +0.15 |

**Win / tie / loss** (fine-tuned vs. base, by overall average): 9 wins, 3 ties, 8 losses.

Mixed, honest result — not a clean sweep, which is the expected shape for a 2-epoch fine-tune on fewer than 900 examples. The real, structural story: base `Qwen3-8B` emits a `<think>...</think>` reasoning block by default, which frequently consumed most of the generation budget and left the actual answer truncated mid-sentence. Fine-tuning (on targets that never contained visible reasoning) taught the model to emit an empty `<think></think>` and go straight to a complete, correctly-structured answer — this is exactly what shows up as the `structural_adherence` and `curriculum_fidelity` gains. The `language_correctness` regression is a real, unresolved caveat, most likely from the small training set introducing occasional grammatical rough edges; worth investigating before scaling up training data or claiming a win.

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
