import os
import json
import time
import multiprocessing as mp

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ====== USER CONFIG ======
INPUT_PATH = "model_outputs/model_1_output.jsonl.jsonl"
DRIVE_OUTPUT_PATH = "thinkprm_outputs.jsonl"
MODEL_ID = "launch/ThinkPRM-7B"
SAVE_EVERY = 50
# =========================

def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            cnt += 1
    return cnt

def flush_buffer_to_file(buffer: list, out_path: str):
    mode = 'a' if os.path.exists(out_path) else 'w'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode, encoding='utf-8') as of:
        for item in buffer:
            of.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Flushed {len(buffer)} items -> {out_path}")
    buffer.clear()

def extract_text_from_vllm_outputs(outputs) -> str:
    try:
        out0 = outputs[0]
        if hasattr(out0, 'outputs') and len(out0.outputs) > 0:
            return getattr(out0.outputs[0], 'text', '') or ''
        return str(outputs)
    except Exception:
        return str(outputs)

def main():


    # Resume support
    existing = count_lines(DRIVE_OUTPUT_PATH)
    if existing > 0:
        print(f"Detected {existing} existing output lines. Will skip that many input records.")

    # Multiprocessing safety
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    print("Loading tokenizer and vLLM model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, max_model_len=16384)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
        stop=None
    )

    buffer = []
    processed_count = 0

    with open(INPUT_PATH, 'r', encoding='utf-8') as inf:
        for idx, raw_line in enumerate(inf):
            if not raw_line.strip():
                continue
            if idx < existing:
                continue

            try:
                item = json.loads(raw_line)
            except Exception as e:
                print(f"Skipping line {idx}: JSON parse error: {e}")
                continue

            question = item.get('question', '')
            solution = item.get('solution', '')

            prompt_text = f"""You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{question}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. For each step, give a score between 0 and 1
"""

            prompt = tokenizer.apply_chat_template(
                [{'role': "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True
            ) + "Let's verify step by step:"

            try:
                outputs = llm.generate(prompt, sampling_params)
                verification_cot = extract_text_from_vllm_outputs(outputs)
            except Exception as e:
                verification_cot = f"<<ERROR during generation: {repr(e)}>>"
                print(f"Generation error for input idx {idx}: {e}")

            result = {
                "question": question,
                "solution": solution,
                "critique": verification_cot
            }
            buffer.append(result)
            processed_count += 1

            if len(buffer) >= SAVE_EVERY:
                flush_buffer_to_file(buffer, DRIVE_OUTPUT_PATH)

            if processed_count % 10 == 0:
                print(f"Processed {processed_count} examples (current input index {idx}).")
            time.sleep(0.05)

    if buffer:
        flush_buffer_to_file(buffer, DRIVE_OUTPUT_PATH)

    print(f"Done. Total new processed: {processed_count}. Output appended to: {DRIVE_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
