# Sample Data

Small, publicly-safe excerpts from the dataset-prep and evaluation pipeline, included so the pipeline's actual inputs/outputs are visible without publishing the source curriculum book or the full generated dataset (see the repo-level [IP note](../../README.md#ip-note)).

## Contents

- **`sft_pairs_sample.jsonl`** — 6 SFT training pairs, one per task type (`exam`, `worksheet`, `indicator_questions`, `lesson_plan`, `grammar_explanation`, `vocab_activity`), picked as the shortest example of each type from the full training set for readability. Each row is a `messages`-format chat triple (system/user/assistant) plus `task_type`, `unit`, `lesson` metadata — the exact schema fed to the SFT trainer.
- **`eval_outputs_sample.jsonl`** — 3 matched base-vs-fine-tuned generation pairs from the held-out evaluation set (`exam`, `worksheet`, `lesson_plan`), each with the shared context, `base_generated`, and `tuned_generated` fields. These illustrate the `<think>` truncation effect described in the [Evaluation](../../README.md#evaluation) section: the base model's reasoning block frequently eats the generation budget, and the tuned model answers directly instead.

## Schema

`sft_pairs_sample.jsonl`:
```
{
  "task_type": str,       # one of the 6 task types this pipeline generates
  "unit": int, "lesson": int,
  "messages": [
    {"role": "system", "content": str},
    {"role": "user", "content": str},      # curriculum context + task instruction
    {"role": "assistant", "content": str}  # teacher-model (gpt-4o-mini) generation, judge-accepted
  ]
}
```

`eval_outputs_sample.jsonl`:
```
{
  "source_entry_id": str, "task_type": str, "unit": int, "lesson": int,
  "context": str,           # held-out prompt (excluded from training)
  "base_generated": str,    # Qwen/Qwen3-8B, zero-shot
  "tuned_generated": str    # same base model + QLoRA adapter
}
```

## Filter statistics (one representative book, `TG07`)

| Metric | Value |
|---|---|
| Total pages | 239 |
| Usable pages (passed extraction quality gate) | 186 |
| Failed pages (front matter, rubric boilerplate, OCR/encoding errors, too short) | 53 |
| Avg. extraction confidence | 0.769 |
| SFT candidates generated | 501 |
| Candidates by task type | exam 185, indicator_questions 131, lesson_plan 101, worksheet 62, vocab_activity 21, grammar_explanation 1 |

Across both books used end-to-end, judge-acceptance rates were 69.5% and 66.1% (see repo README).

## Known limitations

- Context blocks are extracted directly from scanned/exported curriculum PDF text and can carry residual OCR/bidi artifacts (broken tokenization, reversed unit/lesson numerals) — the extraction-quality gate filters the worst of these out, but not all.
- `grammar_explanation` is underrepresented (1 candidate in this book) — task-type distribution is driven by what each book's content naturally supports, not a fixed target.
- These 6+3 rows are illustrative, not statistically representative of the full ~850-example training set or 20-example eval set.

## License / provenance note

The `assistant` content in `sft_pairs_sample.jsonl` is synthetic, LLM-generated (teacher model: `gpt-4o-mini`) text conditioned on curriculum context, not verbatim book content. The `context`/`user` fields are short excerpts (paraphrased structure, indicator lists, lesson metadata) from copyrighted curriculum material, included in small quantity for demonstration purposes only. No full pages, chapters, or the source PDF are included in this repo.
