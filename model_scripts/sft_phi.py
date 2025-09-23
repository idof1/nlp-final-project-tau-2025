# sft.py
# QLoRA 4-bit SFT for Phi-3.5-mini-instruct using chat roles + SFTConfig

# =========================
# Global configuration
# =========================
BASE_MODEL   = "microsoft/Phi-3.5-mini-instruct"

TRAIN_PATH   = "data/parsed_data/wrong_steps_correct_answer_train.jsonl"
DEV_PATH     = "data/parsed_data/evaluation_set.jsonl"

OUTPUT_DIR   = "phi_model"

SEED         = 42
MAX_SEQ_LEN  = 1024
EPOCHS       = 1
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LEARNING_RATE= 7e-5
WARMUP_STEPS = 100

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

MAX_TRAIN_EXAMPLES = None
MAX_EVAL_EXAMPLES  = None
# =========================

import os, random, argparse
from typing import Optional
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="QLoRA SFT with optional sampling of train/dev examples.")
    p.add_argument("--train_path", type=str, default=TRAIN_PATH)
    p.add_argument("--dev_path", type=str, default=DEV_PATH)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--max_train_examples", type=lambda x: None if x=="None" else int(x),
                   default=MAX_TRAIN_EXAMPLES)
    p.add_argument("--max_eval_examples", type=lambda x: None if x=="None" else int(x),
                   default=MAX_EVAL_EXAMPLES)

    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)

    p.add_argument("--lora_r", type=int, default=LORA_R)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    return p.parse_args()

# ---------- IO ----------
def load_jsonl(path: str):
    return datasets.load_dataset("json", data_files=path, split="train")

def limit_dataset(ds: datasets.Dataset, n: Optional[int], seed: int) -> datasets.Dataset:
    """Shuffle with seed and take first n (if provided)."""
    if n is None:
        return ds
    n = max(0, min(n, len(ds)))
    if n == len(ds):
        return ds.shuffle(seed=seed)
    return ds.shuffle(seed=seed).select(range(n))

# ---------- Formatting ----------
def format_dataset(ds: datasets.Dataset, tokenizer, max_seq_len: int):
    """
    Converts {"question", "solution"} -> {"input_ids", "labels"} (token IDs) for SFTTrainer.
    Labels are masked for the prompt part.
    """
    def map_fn(example):
        prompt = f"""You are an expert mathematician. Solve the following problem step by step, numbering each step like "Step 1:", "Step 2:", etc.

Problem:
{example['question']}
"""
        # Combine prompt + solution
        full_text = prompt + example["solution"]
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Labels: mask prompt with -100 so loss is computed only on solution
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_seq_len)["input_ids"])
        labels = [-100]*prompt_len + input_ids[prompt_len:]

        # Ensure labels same length as input_ids
        labels = labels[:len(input_ids)]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return ds.map(map_fn)


# ---------- Main ----------
def main():
    args = parse_args()
    random.seed(args.seed)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    # --- Datasets ---
    train_ds = load_jsonl(args.train_path)
    train_ds = limit_dataset(train_ds, args.max_train_examples, args.seed)
    train_ds = format_dataset(train_ds, tokenizer, args.max_seq_len)

    eval_ds = None
    if args.dev_path and os.path.exists(args.dev_path):
        eval_ds = load_jsonl(args.dev_path)
        eval_ds = limit_dataset(eval_ds, args.max_eval_examples, args.seed)
        eval_ds = format_dataset(eval_ds, tokenizer, args.max_seq_len)

    print(f"Train examples: {len(train_ds)}")
    if eval_ds is not None:
        print(f"Eval examples:  {len(eval_ds)}")

    # --- 4-bit QLoRA base ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # --- Training config ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to="none",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        dataloader_pin_memory=False,
        packing=False,
        seed=args.seed,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"✓ Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()
