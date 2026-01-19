"""Generate chain-of-thought solutions using a Phi LoRA adapter."""

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

token = os.getenv(HF_TOKEN_ENV, "")
if token:
    login(token=token)
else:
    logger.warning(
        "Environment variable %s not set; proceeding without Hugging Face login.", HF_TOKEN_ENV
    )
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_DIR = "phi_model"

# ========================
# Paths & settings
# ========================
INPUT_FILE: Final[str] = "data/parsed_data/questions_for_inference.jsonl"
OUTPUT_FILE: Final[str] = "outputs_phi.jsonl"
MAX_NEW_TOKENS: Final[int] = 1024
TEMPERATURE: Final[float] = 0.1

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()
base_model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()


# ========================
# Prompt & generation
# ========================
def make_prompt(problem: str) -> str:
    return f"""You are an expert mathematician. Solve the following problem step by step, numbering each step like "Step 1:", "Step 2:", etc. Show all reasoning clearly.

question:
{problem}
"""


def generate_solution(problem: str) -> str:
    prompt = make_prompt(problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            use_cache=False,  # disable the cache to avoid DynamicCache error
        )

    # Decode generated tokens after the prompt
    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    solution = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return solution


def main() -> None:
    """Read questions, generate solutions, and save them as JSONL."""
    logging.basicConfig(level=logging.INFO)

    results: list[dict[str, str]] = []
    not_completed_cnt = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating"):
            obj = json.loads(line)
            problem = obj["question"]
            solution = generate_solution(problem)
            if not solution:
                not_completed_cnt += 1
            results.append({"question": problem, "solution": solution})

    logger.info("Times not completed a full answer: %d", not_completed_cnt)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for result in results:
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("Finished generation. Outputs saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
