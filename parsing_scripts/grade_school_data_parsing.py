"""Parsing and corruption utilities for the GSM8K-style grade school data."""

import json
import logging
import random
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


def index_solution(input_file: str, output_file: str) -> None:
    """Normalize raw answers into numbered step-by-step solutions."""
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data: Dict[str, object] = json.loads(line)

            # Replace 'answer' with 'solution'
            answer = data.pop("answer", "")

            # Replace '####' with 'final answer: '
            answer = str(answer).replace("####", "final answer:")

            # Split steps by newline
            steps = answer.strip().split("\n")

            numbered_steps: List[str] = []
            for step in steps:
                step = step.strip()
                if step.startswith("final answer:"):
                    # Keep final answer without a step number
                    numbered_steps.append(step)
                elif step:  # skip empty lines
                    # Add step numbers
                    numbered_steps.append(f"step {len(numbered_steps) + 1}: {step}")

            # Assign to 'solution' key
            data["solution"] = "\n".join(numbered_steps)

            # Write to output file
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info("Indexed solutions from %s into %s", input_file, output_file)


def distort_number(n: int) -> int:
    """Apply a random numeric distortion to corrupt an otherwise-correct answer."""
    distortions = []

    # 1. Add/subtract small
    distortions.append(lambda x: x + random.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10]))
    # 2. Multiply/divide
    distortions.append(lambda x: int(x * random.choice([0.5, 1.5, 2, 3])))
    # 3. Round to nearest 10
    distortions.append(lambda x: round(x, -1))
    # 4. Swap digits if >9
    distortions.append(lambda x: int(str(x)[::-1]) if x > 9 else x + 1)
    # 5. Random replacement in range
    distortions.append(lambda x: random.randint(max(1, x - 50), x + 50))

    fn = random.choice(distortions)
    distorted = fn(n)

    if distorted == n:
        return n + 1
    return distorted


def ruin_step(step_text: str) -> str:
    """Corrupt a single solution step by perturbing numbers and operators."""
    # Ruin numbers
    step_text = re.sub(r"\d+", lambda x: str(int(x.group()) + random.randint(-5, 5)), step_text)
    # Swap some arithmetic operators
    step_text = step_text.replace("+", "-").replace("*", "/")
    return step_text


def ruin_steps(input_file: str, output_file: str) -> None:
    """Shuffle and corrupt intermediate steps while preserving the final answer."""
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data: Dict[str, object] = json.loads(line)
            solution = str(data.get("solution", ""))
            steps = solution.strip().split("\n")

            ruined_steps: List[str] = []
            final_answer = None
            intermediate_steps: List[str] = []

            # Separate final answer from intermediate steps
            for step in steps:
                if step.startswith("final answer:"):
                    final_answer = step
                else:
                    # Remove the 'step N: ' prefix
                    match = re.match(r"step \d+:\s*(.*)", step)
                    if match:
                        intermediate_steps.append(match.group(1))
                    else:
                        intermediate_steps.append(step)

            # Shuffle intermediate steps
            random.shuffle(intermediate_steps)

            # Ruin each step and re-add numbering
            for i, step_text in enumerate(intermediate_steps, 1):
                ruined_steps.append(f"step {i}: {ruin_step(step_text)}")

            # Append final answer at the end
            if final_answer:
                ruined_steps.append(final_answer)

            data["solution"] = "\n".join(ruined_steps)
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info("Ruined steps from %s into %s", input_file, output_file)


def distort_answers(input_file: str, output_file: str) -> None:
    """Corrupt only the final numeric answers while keeping reasoning intact."""
    with (
        open(input_file, "r", encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            obj: Dict[str, object] = json.loads(line)
            solution = str(obj.get("solution", ""))

            if "final answer:" in solution:
                parts = solution.rsplit("final answer:", 1)
                number_str = parts[1].strip()

                # handle commas
                number_str_clean = number_str.replace(",", "")

                # parse
                try:
                    correct_val: float | int
                    correct_val = int(number_str_clean)
                except ValueError:
                    correct_val = float(number_str_clean)

                distorted = distort_number(int(correct_val))

                # reformat with commas if original had them
                if "," in number_str:
                    distorted_str = f"{distorted:,}"
                else:
                    distorted_str = str(distorted)

                new_solution = parts[0] + f"final answer: {distorted_str}"
                obj["solution"] = new_solution

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("Distorted final answers from %s into %s", input_file, output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_file = "data/original_data/original_train.jsonl"
    output_file = "data/parsed_data/correct_steps_correct_answer_train.jsonl"
    index_solution(input_file, output_file)

    input_file = "data/original_data/test.jsonl"
    output_file = "data/parsed_data/evaluation_set.jsonl"
    index_solution(input_file, output_file)

    input_file = "data/parsed_data/correct_steps_correct_answer_train.jsonl"
    output_file = "../data/parsed_data/wrong_steps_correct_answer_train.jsonl"
    ruin_steps(input_file, output_file)

    input_file = "data/parsed_data/correct_steps_correct_answer_train.jsonl"
    output_file = "data/parsed_data/correct_steps_wrong_answer_train.jsonl"
    distort_answers(input_file, output_file)
