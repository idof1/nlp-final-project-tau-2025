"""Generate chain-of-thought solutions using a Phi LoRA adapter."""

import argparse
import json
import logging
import os
from typing import Final

import torch
from huggingface_hub import login
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

HF_TOKEN_ENV: Final[str] = "HF_TOKEN"

# ========================
# Defaults
# ========================
BASE_MODEL: Final[str] = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_DIR: Final[str] = "phi_model"
INPUT_FILE: Final[str] = "data/parsed_data/questions_for_inference.jsonl"
OUTPUT_FILE: Final[str] = "outputs_phi.jsonl"
MAX_NEW_TOKENS: Final[int] = 1024
TEMPERATURE: Final[float] = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CoT solutions with a Phi LoRA adapter.")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--adapter_dir", default=ADAPTER_DIR)
    p.add_argument("--input_file", default=INPUT_FILE)
    p.add_argument("--output_file", default=OUTPUT_FILE)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    return p.parse_args()


def _hf_login() -> None:
    token = os.getenv(HF_TOKEN_ENV, "")
    if token:
        login(token=token)
    else:
        logger.warning(
            "Environment variable %s not set; proceeding without Hugging Face login.",
            HF_TOKEN_ENV,
        )


def load_model_and_tokenizer(
    base_model: str, adapter_dir: str
) -> tuple[PeftModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def make_prompt(problem: str) -> str:
    return (
        'You are an expert mathematician. Solve the following problem step by step, '
        'numbering each step like "Step 1:", "Step 2:", etc. Show all reasoning clearly.\n\n'
        f"question:\n{problem}\n"
    )


def generate_solution(
    problem: str,
    model: PeftModel,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = make_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    _hf_login()
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_dir)

    results: list[dict[str, str]] = []
    not_completed_cnt = 0

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating"):
            obj = json.loads(line)
            problem = obj["question"]
            solution = generate_solution(
                problem, model, tokenizer, args.max_new_tokens, args.temperature
            )
            if not solution:
                not_completed_cnt += 1
            results.append({"question": problem, "solution": solution})

    logger.info("Times not completed a full answer: %d", not_completed_cnt)

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for result in results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("Finished generation. Outputs saved to %s", args.output_file)


if __name__ == "__main__":
    main()
