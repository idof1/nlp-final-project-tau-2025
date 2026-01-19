"""Run a vLLM verifier model to critique and score GSM8K-style solutions."""

import json
import logging
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# ====== USER CONFIG ======
INPUT_PATH = "model_outputs/model_1_output.jsonl.jsonl"
DRIVE_OUTPUT_PATH = "thinkprm_outputs.jsonl"
MODEL_ID = "launch/ThinkPRM-7B"
SAVE_EVERY = 50
# =========================


def count_lines(path: str) -> int:
    """Return the number of lines in a file, or 0 if it does not exist."""
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def flush_buffer_to_file(buffer: List[Dict[str, Any]], out_path: str) -> None:
    """Append buffered JSON objects to a file and clear the buffer."""
    mode = "a" if os.path.exists(out_path) else "w"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode, encoding="utf-8") as of:
        for item in buffer:
            of.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Flushed %d items to %s", len(buffer), out_path)
    buffer.clear()


def extract_text_from_vllm_outputs(outputs: Any) -> str:
    """Extract the text field from vLLM outputs, falling back to repr on error."""
    try:
        out0 = outputs[0]
        if hasattr(out0, "outputs") and len(out0.outputs) > 0:
            return getattr(out0.outputs[0], "text", "") or ""
        return str(outputs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse vLLM outputs: %r", exc)
        return str(outputs)


def main() -> None:
    """Stream inputs through the verifier model and store critiques incrementally."""
    logging.basicConfig(level=logging.INFO)

    # Resume support
    existing = count_lines(DRIVE_OUTPUT_PATH)
    if existing > 0:
        logger.info("Detected %d existing output lines; skipping those inputs.", existing)

    # Multiprocessing safety
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # Already set; safe to continue.
        pass

    logger.info("Loading tokenizer and vLLM model '%s'...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, max_model_len=16384)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop=None)

    buffer: List[Dict[str, str]] = []
    processed_count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as inf:
        for idx, raw_line in enumerate(inf):
            if not raw_line.strip():
                continue
            if idx < existing:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d due to JSON parse error: %r", idx, exc)
                continue

            question = item.get("question", "")
            solution = item.get("solution", "")

            prompt_text = f"""You are given a math problem and a proposed step-by-step solution:

[Math Problem]

{question}

[Solution]

{solution}

Review and critique each step in the proposed solution to determine whether each step is correct. For each step, give a score between 0 and 1
"""

            prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                + "Let's verify step by step:"
            )

            try:
                outputs = llm.generate(prompt, sampling_params)
                verification_cot = extract_text_from_vllm_outputs(outputs)
            except Exception as exc:  # noqa: BLE001
                verification_cot = f"<<ERROR during generation: {repr(exc)}>>"
                logger.error("Generation error for input idx %d: %r", idx, exc)

            result: Dict[str, str] = {
                "question": str(question),
                "solution": str(solution),
                "critique": verification_cot,
            }
            buffer.append(result)
            processed_count += 1

            if len(buffer) >= SAVE_EVERY:
                flush_buffer_to_file(buffer, DRIVE_OUTPUT_PATH)

            if processed_count % 10 == 0:
                logger.info(
                    "Processed %d examples (current input index %d).",
                    processed_count,
                    idx,
                )
            time.sleep(0.05)

    if buffer:
        flush_buffer_to_file(buffer, DRIVE_OUTPUT_PATH)

    logger.info(
        "Verification run complete. Total new processed: %d. Output appended to: %s",
        processed_count,
        DRIVE_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
