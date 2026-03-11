# Do Intermediate Reasoning Steps Matter? Evaluating Generalization Under Data Corruption

**Authors:** Noam Barlin, Dan Ayzik, Ido Friedman
**Course:** NLP Final Project — Tel Aviv University, 2025

> We design a controlled experimental framework to test whether LLMs genuinely rely on their
> chain-of-thought (CoT) steps to derive final answers, or whether CoT can function as a superficial pattern.

---

## Table of Contents
- [Motivation](#motivation)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Project Structure](#project-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Dependencies](#dependencies)

---

## Motivation

Large language models produce higher-quality answers when prompted to "think step by step." But is the intermediate reasoning *causally* responsible for the correct answer, or does the model simply recall the answer and generate plausible-sounding steps after the fact?

Prior work (Turpin et al., 2023; Lanham et al., 2023) has suggested CoT may act as *rationalization* rather than genuine reasoning. We probe this directly using a **data-corruption approach**: fine-tune the same base model on training sets that decouple reasoning steps from final answers, then measure whether each component independently drives model behaviour.

---

## Experimental Design

### Dataset — GSM8K
[GSM8K](https://huggingface.co/datasets/openai/gsm8k) is a benchmark of ~8,500 grade-school math word problems with step-by-step solutions. We use 7,473 training examples and a disjoint 500-example test set for final evaluation.

### Data Variants

| Model | Training condition | Steps correct? | Answer correct? |
|:-----:|-------------------|:--------------:|:---------------:|
| **1** | CS-CA: correct steps + correct answer | ✅ | ✅ |
| **2** | WS-CA: wrong steps + correct answer   | ❌ | ✅ |
| **3** | CS-WA: correct steps + wrong answer   | ✅ | ❌ |
| **4** | Base model (no fine-tuning)           | —  | — |

**Corruption methods:**
- *Wrong steps (WS)* — intermediate steps are shuffled, numbers perturbed (±5), and operators swapped (`+`→`-`, `*`→`/`).
- *Wrong answer (WA)* — the final numeric answer is replaced with a distorted value (digit-swap, offset, ×/÷ by a small constant).

### Model — Phi-3.5-mini-instruct + QLoRA

- **Base model:** `microsoft/Phi-3.5-mini-instruct` (3.8B parameters)
- **Fine-tuning:** QLoRA (4-bit NF4, double quantization, `paged_adamw_32bit`, cosine LR)
- **LoRA targets:** `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`
- **Hyperparameters:** r=16, α=32, dropout=0.05, lr=7×10⁻⁵, 1 epoch, batch 1 + grad-accum 8

### Verifier — ThinkPRM-7B

Generated solutions are scored step-by-step by [`launch/ThinkPRM-7B`](https://huggingface.co/launch/ThinkPRM-7B), a process-reward model trained on PRM800K. This provides a *reasoning quality* signal independent of final-answer accuracy.

---

## Results

Evaluated on 500 held-out GSM8K test examples:

| Model | Training condition | EM | StepScore | AllOK | Spearman | SelAcc@0.7 | Cov@0.7 |
|:-----:|-------------------|----|-----------|-------|----------|------------|---------|
| 1 | CS-CA (clean SFT)  | **0.762** | 0.927 | 0.786 | +0.508 | 0.794 | 0.922 |
| 2 | WS-CA (wrong steps)| 0.134 | 0.308 | 0.190 | -0.060 | 0.093 | 0.258 |
| 3 | CS-WA (wrong answer)| 0.010 | 0.904 | 0.754 | -0.073 | 0.009 | 0.888 |
| 4 | Base model         | **0.830** | 0.965 | 0.912 | +0.463 | 0.860 | 0.960 |

**Metrics:** EM = Exact Match accuracy. StepScore = average step-level correctness (0–1). AllOK = proportion of examples where all steps scored ≥0.5. Spearman = rank correlation between step scores and answer correctness. SelAcc@0.7 = accuracy when retaining only examples with StepScore ≥0.7. Cov@0.7 = proportion retained.

### Key findings

1. **CoT can be a superficial pattern.**
   Model 3 (CS-WA) maintains high step quality (0.904) but achieves near-zero accuracy (1.0%). Its Spearman of −0.073 confirms that step correctness and answer correctness are completely decoupled — the model produces plausible-looking reasoning but outputs the memorised wrong answer regardless.

2. **Step quality strongly predicts accuracy — under aligned training.**
   Models 1 and 4 both show positive Spearman correlations (0.508 / 0.463) and high SelAcc@0.7, meaning step scores are a reliable confidence signal when training data is consistent.

3. **Corrupting steps devastates accuracy.**
   Model 2 (WS-CA) drops from 76.2% to 13.4% even though its training labels had correct answers. This shows that the *form* of reasoning shapes final predictions, not just the answer token.

4. **Misaligned training destroys calibration.**
   Models 2 and 3 have near-zero / negative Spearman and SelAcc@0.7 near 0, making their confidence signals useless.

5. **One epoch of SFT on this scale slightly hurts the base model.**
   Model 1 (76.2%) underperforms the unmodified base (83.0%), likely due to minor overfitting on a relatively small dataset.

---

## Project Structure

```
.
├── data/
│   ├── original_data/          # Raw GSM8K JSONL files
│   └── parsed_data/            # Processed & corrupted datasets
│       ├── correct_steps_correct_answer_train.jsonl
│       ├── wrong_steps_correct_answer_train.jsonl
│       ├── correct_steps_wrong_answer_train.jsonl
│       ├── evaluation_set.jsonl        # 500-example validation set
│       └── questions_for_inference.jsonl  # 500-example test set
│
├── model_scripts/
│   ├── sft_phi.py              # QLoRA fine-tuning (training)
│   ├── generate.py             # CoT solution generation (inference)
│   └── verify.py               # Step-level verification with ThinkPRM
│
├── parsing_scripts/
│   ├── grade_school_data_parsing.py   # Dataset corruption & normalization
│   ├── parse_critique.py              # Parse ThinkPRM outputs -> step scores
│   └── clean_base_model_outputs.py    # Normalize raw model outputs
│
├── model_adapters/             # Saved LoRA adapter weights (.rar)
├── model_outputs/              # Raw JSONL outputs per model
├── verifier_outputs/           # ThinkPRM critique outputs
├── parsed_verifier_critiques/  # Structured step scores (final results)
│
├── analyze_results.py          # Full metrics table + findings
├── model_mappings.txt          # Model index -> training condition
├── NLP_project.pdf             # Full project report
└── pyproject.toml              # Project metadata & tooling config
```

---

## Reproducing the Experiments

### 1. Prepare the data

```bash
python parsing_scripts/grade_school_data_parsing.py
```

Reads `data/original_data/` and writes all variants to `data/parsed_data/`.

### 2. Fine-tune models 1–3

```bash
# Model 1 — clean data
python model_scripts/sft_phi.py \
  --train_path data/parsed_data/correct_steps_correct_answer_train.jsonl \
  --output_dir model_adapters/model_1

# Model 2 — corrupted steps
python model_scripts/sft_phi.py \
  --train_path data/parsed_data/wrong_steps_correct_answer_train.jsonl \
  --output_dir model_adapters/model_2

# Model 3 — corrupted answer
python model_scripts/sft_phi.py \
  --train_path data/parsed_data/correct_steps_wrong_answer_train.jsonl \
  --output_dir model_adapters/model_3
```

Requires a GPU with ≥12 GB VRAM (tested on A100 40 GB via Google Colab Pro).

### 3. Generate solutions

```bash
python model_scripts/generate.py --adapter_dir model_adapters/model_1 --output_file model_outputs/model_1_output.jsonl
# repeat for models 2, 3, 4 (omit --adapter_dir for model 4)
```

### 4. Verify with ThinkPRM

```bash
python model_scripts/verify.py   # edit INPUT_PATH / DRIVE_OUTPUT_PATH per model
```

### 5. Parse critiques

```bash
python parsing_scripts/parse_critique.py
```

### 6. Analyze results

```bash
python analyze_results.py
```

---

## Dependencies

```
torch>=2.1
transformers>=4.43
peft>=0.11
trl>=0.9
bitsandbytes>=0.43
datasets>=2.19
huggingface_hub
tqdm
```

Install with:

```bash
pip install torch transformers peft trl bitsandbytes vllm datasets huggingface_hub tqdm
```

> **Note:** `bitsandbytes` and `vllm` require a CUDA-capable GPU and Linux. Training was performed on Google Colab Pro (A100).

---

## References

- Wei et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS.
- Cobbe et al. (2021). *Training verifiers to solve math word problems.* EMNLP.
- Turpin et al. (2023). *Language models don't always say what they think.* ACL.
- Lanham et al. (2023). *Measuring faithfulness in chain-of-thought reasoning.* EMNLP.
- Lightman et al. (2023). *PRM800K: A process supervision dataset.* NeurIPS.
