# BilArabi Fine-Tuning Blueprint
### Production-Grade LLM Specialization for Arabic Curriculum Content — 0 → 100 Plan

> **Purpose of this document:** A grounded, end-to-end execution blueprint for building a public, recruiter-facing fine-tuning repository using BilArabi curriculum data — with minimal (near-zero) compute cost, industry-grade practices, and clean IP separation.
>
> **Portfolio triad:** RAG (agentic_graph_rag) · Agents · **Fine-Tuning (this repo)**
>
> **Status:** Blueprint v1.0 — July 2026

---

## 0. Project Framing

### 0.1 What fine-tuning is for (and not for)

| Fine-tuning IS for | Fine-tuning is NOT for |
|---|---|
| Task specialization (exam/worksheet generation) | Knowledge injection (facts of the book) |
| Output format compliance (BilArabi house style) | Replacing retrieval |
| Register/style (MSA educational Arabic) | Cross-chapter linking (that's the graph's job) |
| Consistent structure (answer keys, timings, indicators) | Up-to-date content |

**Core thesis of the repo:** *RAG retrieves the curriculum context; the fine-tuned model generates in-style, indicator-aligned educational artifacts.* FT and RAG are complements — state this explicitly in the README; it is a senior-level framing.

### 0.2 The model's job (task definition)

> Given curriculum context (lesson content, indicator text, grammar topic, level), generate BilArabi-style outputs:
> 1. Exams with answer keys (امتحان + مفتاح الإجابة)
> 2. Worksheets (أوراق عمل)
> 3. Indicator-aligned question sets (أسئلة مرتبطة بمؤشر تعلّم)
> 4. Lesson plans (خطط شرح الدروس)
> 5. Grade-calibrated grammar explanations (شرح قواعد بمستوى الصف)
> 6. Vocabulary activities (أنشطة مفردات: جذور، اشتقاقات، سياق)

### 0.3 IP / copyright guardrails (public repo hygiene)

- **Never publish** raw book pages, OCR dumps, or datasets that reproduce book text.
- Public repo contains: pipeline **code**, a **small synthetic sample** (50–100 pairs, fully synthetic or heavily transformed), the **LoRA adapter**, eval reports, model card.
- Real dataset lives in a **private** repo / private HF dataset.
- README explicitly documents this separation → reads as enterprise data-governance awareness, which is a selling point, not a limitation.

### 0.4 Model selection

| Choice | Model | Why |
|---|---|---|
| **Primary** | Qwen3-8B (Instruct) | Strong Arabic, Apache 2.0, Unsloth support, big ecosystem |
| **Ablation** | Qwen3-4B | Size ablation table; runs on weaker hardware |
| Alternatives (mention in README) | ALLaM, Fanar, AceGPT | Arabic-native options; ALLaM = Saudi signal |

### 0.5 Success criteria (define BEFORE building)

- ≥ +1.0 absolute improvement on 5-point judge rubric vs. base model (same prompts, same context)
- ≥ 95% format compliance (parseable exam structure, answer key present)
- No regression > 2 points on Arabic general benchmark slice
- Model + adapter published on HF Hub with complete model card
- Fully reproducible: `make dataset && make train && make eval && make serve`

---

## 1. Phase 1 — Dataset Engineering (Week 1–1.5) ★ 60% of credibility

### 1.1 Source data

Reuse the structured JSON extraction from the RAG pipeline (one extraction feeds both repos):

```json
{
  "level": 8, "unit": 1, "lesson": 1,
  "lesson_title": "كلّنا متشابهون",
  "section": "ورشة المفردات",
  "indicator_code": "2.أ",
  "indicator_text": "يكتشف معاني المفردات مستخدمًا سياق الجمل والفقرات...",
  "activity_no": 1, "duration_min": 15,
  "objective": "...", "content": "...", "answers": "...",
  "grammar_topics": ["المثنى"], "vocab": [{"word": "الترحال", "root": "رحل"}]
}
```

### 1.2 Synthetic instruction generation

- **Teacher model:** Claude Sonnet / GPT-4o via API (~$10–30 total) or Qwen3-235B free tiers.
- **Volume target:** 3,000–5,000 pairs across the 6 task types (roughly balanced; oversample exam + indicator-questions as flagship tasks).
- **Structure of every pair (ChatML):**

```
system: أنت مساعد تربوي متخصص في منهاج "بالعربي". تُنتج مواد تعليمية
        بالفصحى ملتزمة ببنية المنهاج ومؤشرات التعلّم.
user:   [INSTRUCTION in Arabic]
        <السياق>
        [curriculum context JSON/text — level, lesson, indicator, content]
        </السياق>
assistant: [Target output — exam / worksheet / plan ...]
```

**Rule: context is ALWAYS in the prompt.** This teaches context-grounding (pairs with RAG at inference) and avoids training the weights to reproduce book text as targets.

- **Diversity axes to vary during generation:** phrasing of instruction (formal/casual teacher voice), level (1–9), output length, number of questions, with/without tashkeel in query, dialect-tinged queries → MSA output.

### 1.3 Quality pipeline (the "industry-grade" part)

```
raw generations (≈12k)
  → schema/format validator (parseable structure, answer key present)
  → LLM-as-judge scoring (rubric below, drop score < 4)
  → dedup (MinHash + embedding cosine > 0.92 near-dup removal)
  → Arabic checks (no code-switching, MSA register, tashkeel consistency)
  → decontamination (no overlap with eval set — exact + fuzzy)
  → final dataset (≈3–5k)
```

**Judge rubric (1–5 each, average, threshold ≥ 4.0):**
1. **صحة اللغة** — grammatical correctness, MSA register
2. **الالتزام بالمنهاج** — faithful to provided context/indicator, no invented curriculum facts
3. **الالتزام بالبنية** — exam/worksheet structure, answer key, numbering
4. **الملاءمة للمستوى** — difficulty calibrated to the stated grade level
5. **قابلية الاستخدام** — a real teacher could use it as-is

**README gold number:** report the funnel, e.g. *"12,400 generated → 4,100 after quality gates (33% acceptance)"*.

### 1.4 Splits & anti-memorization design

- Train / val split: 95 / 5
- **Held-out-by-lesson eval:** exclude 2–3 ENTIRE lessons from training; eval prompts reference only those lessons → proves generalization, not memorization. Call this out in README.
- Decontamination check committed as a script: `scripts/check_contamination.py`

### 1.5 Artifacts

- `data/sample/` — 50–100 public synthetic pairs (JSONL)
- `data/README.md` — **datacard**: schema, generation prompts (link), filter statistics, known limitations, license note
- Versioned JSONL naming: `bilarabi_sft_v{N}_{date}.jsonl` + config hash

---

## 2. Phase 2 — Training (Week 2, 2–3 days of runs)

### 2.1 Infrastructure (free)

| Resource | Details |
|---|---|
| **Kaggle** | 30 GPU-hrs/week, T4×2 or P100 — primary |
| Colab free | Backup / quick experiments |
| Framework | **Unsloth** + TRL SFTTrainer, QLoRA 4-bit |

Qwen3-8B in 4-bit QLoRA fits in ~16GB VRAM at seq 4096 with grad checkpointing.

### 2.2 Reference config

```yaml
model: Qwen/Qwen3-8B-Instruct        # + 4B ablation
quantization: 4bit-nf4 (QLoRA)
lora:
  r: 16                               # ablate: 8 / 16 / 32
  alpha: 32                           # α = 2r
  target: all-linear                  # q,k,v,o,gate,up,down
  dropout: 0.05
training:
  lr: 2.0e-4
  scheduler: cosine, warmup 3%
  epochs: 2                           # ablate: 1 / 2 / 3
  effective_batch: 16                 # per-device 2 × grad_accum 8
  max_seq_len: 4096                   # exams are long
  bf16: true (or fp16 on T4)
  grad_checkpointing: true
  seed: fixed + logged
```

### 2.3 Experiment tracking — W&B (free tier)

Log per run: full config, dataset version hash, loss curves, val loss, judge score on 30 fixed val prompts, GPU hours.

**Ablation matrix (the README centerpiece — worth more than the model):**

| Run | Model | r | Epochs | Context in prompt | Val judge score |
|---|---|---|---|---|---|
| A1 | 8B | 16 | 2 | ✅ | — |
| A2 | 8B | 32 | 2 | ✅ | — |
| A3 | 8B | 16 | 1 | ✅ | — |
| A4 | 8B | 16 | 2 | ❌ | — |
| A5 | 4B | 16 | 2 | ✅ | — |

### 2.4 Checkpoint selection

Pick by **val-set judge score**, not loss alone. One README sentence: *"Final checkpoint selected by rubric score on validation prompts; loss and generation quality diverged after epoch 2"* — signals you know loss ≠ quality.

---

## 3. Phase 3 — Evaluation (Week 2–3, 2–3 days) ★ second-biggest credibility source

### 3.1 Three-layer eval

**Layer 1 — Task eval (headline):**
- 100–150 held-out prompts (incl. held-out lessons), same prompts to **base vs. fine-tuned**
- LLM-as-judge with the written rubric (§1.3), judge blinded to which model produced which output (randomize A/B order)
- Report: per-criterion delta table + per-task-type breakdown + win/tie/loss counts

**Layer 2 — Regression / forgetting check:**
- lm-evaluation-harness on an ArabicMMLU subset (or similar), base vs. tuned
- Even a ~0 delta is the point — *checking* is the signal

**Layer 3 — Human spot-check:**
- 20 samples reviewed manually, notes committed to `eval/human_review.md`

### 3.2 Tooling

- `scripts/run_eval.py` → JSON report (mirror of agentic_graph_rag's eval harness — consistent personal brand)
- `eval/rubric.md` — the rubric as a standalone doc
- `eval/results/` — versioned JSON reports per checkpoint

### 3.3 Headline README table (target shape)

| Criterion | Base Qwen3-8B | Fine-tuned | Δ |
|---|---|---|---|
| Format compliance | ~60% | ≥95% | +35 |
| Curriculum alignment | — | — | — |
| Arabic quality | — | — | — |
| Level calibration | — | — | — |
| **Avg rubric (1–5)** | — | — | **≥ +1.0** |

---

## 4. Phase 4 — Serving (Week 3, ~2 days)

### 4.1 Two serving paths (show both)

**Path A — Local / edge (Ollama):**
```
merge adapter → quantize GGUF Q4_K_M → Modelfile → ollama run bilarabi-teacher
```

**Path B — Production (vLLM + hot-swappable LoRA):**
```
vllm serve Qwen/Qwen3-8B-Instruct --enable-lora --lora-modules bilarabi=./adapter
```
Serving base + swappable adapters is the current production pattern — demonstrating it is a differentiator.

### 4.2 App layer

- Thin **FastAPI** wrapper: `/generate` with streaming, task-type param, context injection slot (RAG hookup point)
- **Gradio** demo UI (~50 lines): task dropdown, level selector, context box, generate
- Optional: one endpoint that calls the RAG repo for context then this model → the full-loop demo

### 4.3 Publish to HuggingFace Hub

- LoRA adapter + GGUF quant
- **Model card** (rare in portfolios — instantly professional): intended use, training data description (synthetic, curriculum-derived, private), eval results, limitations (level coverage, dialects, hallucination scope), bias/safety notes, license

---

## 5. Phase 5 — Engineering Wrapper (Week 3, ~2 days)

What separates "notebook" from "repo":

```
bilarabi-finetune/
├── Makefile                    # dataset / train / eval / serve / all
├── configs/                    # YAML per run (pydantic-settings or Hydra)
├── src/
│   ├── datagen/                # synthetic generation + judge + filters
│   │   ├── generate.py
│   │   ├── judge.py
│   │   ├── filters.py          # dedup, Arabic checks, decontamination
│   │   └── prompts/            # per-task generation prompts (versioned)
│   ├── training/
│   │   ├── train_sft.py        # Unsloth QLoRA
│   │   └── merge_quantize.py   # merge → GGUF
│   ├── eval/
│   │   ├── run_eval.py         # judge-based A/B eval → JSON report
│   │   ├── regression.py       # lm-eval-harness wrapper
│   │   └── rubric.md
│   └── serving/
│       ├── api.py              # FastAPI streaming
│       ├── app_gradio.py
│       └── Modelfile           # Ollama
├── data/
│   ├── sample/                 # public synthetic sample only
│   └── README.md               # datacard
├── eval/results/               # versioned reports
├── notebooks/                  # Kaggle training notebook (thin, calls src/)
├── .github/workflows/ci.yml    # lint + eval validators + contamination check on PR
├── Dockerfile                  # serving path
├── MODEL_CARD.md
└── README.md
```

**README must contain:** architecture diagram (data → train → eval → serve), the before/after table, ablation table, dataset funnel numbers, cost breakdown, 2-min demo video link, FT+RAG complementarity paragraph, IP-separation note.

**Cost breakdown line for README:** *"Compute: $0 (Kaggle free tier). API (data gen + judging): ~$18. Total: under $20."*

---

## 6. Phase 6 (Optional Standout) — DPO Round

- Sample 2 outputs per prompt from the SFT model (temp 0.8) on 300–500 prompts
- Judge picks winner → preference pairs
- DPO (or ORPO) on top of SFT adapter, β = 0.1, 1 epoch, lr 5e-6
- Report SFT vs. SFT+DPO delta on the same eval
- SFT→DPO pipeline knowledge = current interview differentiator; even a modest delta with honest reporting reads senior

---

## 7. Timeline & Budget Summary

| Phase | Effort | Cost |
|---|---|---|
| 0 Framing + repo scaffold | 0.5 day | $0 |
| 1 Dataset engineering | 5–7 days | $10–25 API |
| 2 Training + ablations | 2–3 days | $0 (Kaggle) |
| 3 Evaluation | 2–3 days | $5–10 API (judging) |
| 4 Serving + HF publish | 2 days | $0 |
| 5 Engineering wrapper | 2 days | $0 |
| 6 DPO (optional) | 2–3 days | $3–5 API |
| **Total** | **~3 weeks part-time** | **< $40** |

---

## 8. Execution Checklist

**Phase 0**
- [ ] Repo scaffolded with Makefile + configs
- [ ] Success criteria written into README (before building)
- [ ] IP-separation policy documented

**Phase 1**
- [ ] Extraction JSON reused from RAG pipeline
- [ ] Generation prompts written & versioned per task type
- [ ] Judge rubric written (`eval/rubric.md`)
- [ ] Quality funnel implemented (validator → judge → dedup → Arabic checks → decontamination)
- [ ] Held-out-by-lesson split created
- [ ] Datacard written; public sample exported

**Phase 2**
- [ ] Kaggle notebook runs end-to-end on sample
- [ ] W&B project live; seeds fixed
- [ ] 5-run ablation matrix completed
- [ ] Checkpoint selected by judge score

**Phase 3**
- [ ] 100–150 prompt A/B eval, judge blinded
- [ ] Regression benchmark run (base vs. tuned)
- [ ] Human review of 20 samples committed
- [ ] Headline delta table in README

**Phase 4**
- [ ] GGUF + Ollama Modelfile working
- [ ] vLLM LoRA serving config committed
- [ ] FastAPI + Gradio demo
- [ ] HF Hub: adapter + GGUF + model card

**Phase 5**
- [ ] CI: lint + eval validators + contamination check
- [ ] Dockerfile for serving
- [ ] README complete (diagram, tables, video, cost line)

**Phase 6 (optional)**
- [ ] Preference pairs collected
- [ ] DPO run + delta reported

---

## 9. Recruiter Narrative (copy-paste seed)

> *"Complete LLM specialization pipeline for Arabic education: synthetic data engineering with quality gates (12k → 4k acceptance funnel), QLoRA training with ablation studies, blinded judge-based evaluation with regression testing against catastrophic forgetting, and dual serving paths (Ollama GGUF + vLLM hot-swappable LoRA) — with proprietary curriculum data kept fully separated from the public repository. Total cost: under $40."*

One sentence. Hits: data engineering, training rigor, eval rigor, deployment, IP governance, cost discipline.

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU limits mid-run | Checkpoint every epoch; resume-from-checkpoint in train script |
| Judge model bias inflates scores | Blinded A/B, randomized order; human spot-check layer |
| Memorization instead of generalization | Held-out-by-lesson eval; context-always-in-prompt design |
| Copyright exposure | No raw text in public repo; synthetic sample only; datacard states policy |
| Arabic quality drift (code-switching) | Dedicated filter in quality pipeline + rubric criterion |
| T4 fp16 instability | Use Unsloth's fp16-safe defaults; monitor loss spikes; lower lr if needed |
