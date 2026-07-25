# Judge Rubric

Used by `src/datagen/judge.py` to score each generated SFT pair before it's accepted into the training set. Five criteria, each scored 1-5. A pair is accepted when the average across all five is **≥ 4.0**.

| Criterion | Key | What it measures |
|---|---|---|
| صحة اللغة | `language_correctness` | Grammatical correctness, MSA register |
| الالتزام بالمنهاج | `curriculum_fidelity` | Faithful to the provided context; no invented curriculum facts |
| الالتزام بالبنية | `structural_adherence` | Structure appropriate to the task type (exam/worksheet/etc.), clear numbering, answer key where required |
| الملاءمة للمستوى | `level_calibration` | Difficulty and time allocation calibrated to the stated grade level |
| قابلية الاستخدام | `usability` | A real teacher could use this as-is |

The judge model is given the original curriculum context and the generated output side by side, and returns a JSON object with a 1-5 score per criterion plus a short note. See `_JUDGE_INSTRUCTION_TEMPLATE` in `src/datagen/judge.py` for the exact prompt.

Report the funnel in the README as evidence of the quality gate actually doing something, e.g.:

> 377 generated → N accepted (X% acceptance)
