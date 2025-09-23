import json
from typing import Dict, List, Tuple, Optional
import re


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts the number after 'final answer:' (case-insensitive).
    Allows commas inside numbers, and supports negatives.
    Returns the number as a string with commas removed.
    """
    match = re.search(r"final answer:\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    print("No final answer found.")
    return None


def split_steps(text: str) -> Tuple[Dict[int, str], List[str], int]:

    trimmed = text.lstrip("\n\r\t ")
    step_pat = re.compile(r"\bStep\s+(\d+)\b", re.IGNORECASE)
    matches = [(int(m.group(1)), m.start(), m.end()) for m in step_pat.finditer(text)]

    if not matches or not trimmed.lower().startswith("step 1"):
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        steps = {i + 1: chunk for i, chunk in enumerate(chunks)}
        max_step = len(steps)
        errors = [
            f"No 'Step X' markers found or text does not start with 'Step 1'. Split into {max_step} chunks by \\n\\n."]
        return steps, errors, max_step

    matches.sort(key=lambda t: t[1])  # sort by position
    max_step = max(m[0] for m in matches)

    steps: Dict[int, str] = {}
    errors: List[str] = []
    last_pos = 0

    for i in range(1, max_step + 1):
        # find first "Step i" at or after last_pos
        start_match = next((m for m in matches if m[0] == i and m[1] >= last_pos), None)
        if not start_match:
            msg = f"⚠️ Step {i} not found after position {last_pos}."
            errors.append(msg)
            continue

        start = start_match[1]

        # prefer the next Step i+1 that comes after this start
        candidate = None
        if i < max_step:
            candidate = next((m for m in matches if m[0] == i + 1 and m[1] > start), None)

            # fallback: earliest Step k with k > i that occurs after this start
            if candidate is None:
                candidate = next((m for m in matches if m[0] > i and m[1] > start), None)

        # determine end
        if candidate:
            end = candidate[1]
        else:
            if i == max_step:
                end = len(text)  # last step goes to end
            else:
                msg = f"⚠️ Couldn't find Step {i+1} (or any Step > {i}) after Step {i} at pos {start}. Taking until end."
                errors.append(msg)
                end = len(text)

        steps[i] = text[start:end].strip()
        last_pos = end

    return steps, errors, max_step


def fetch_score(step_text: str):
    # patterns to match
    patterns = [
        (r"\*\*Score:\*\*\s*([-+]?\d*\.?\d+)", None),      # **Score:** 3.5
        (r"boxed\{([-+]?\d*\.?\d+)\}", None),              # boxed{3.5}
        (r"Score:\*\*\s*([-+]?\d*\.?\d+)", None),          # Score:** 3.5
        (r"Score:\s*([-+]?\d*\.?\d+)", None),              # Score: 3.5
        (r"the score for this step is\s*([-+]?\d*\.?\d+)", None),  # the score for this step is 3.5
        (r"score is\s*([-+]?\d*\.?\d+)", None),            # score is 3.5
        (r"\\boxed\{correct\}", 1.0),                      # \boxed{correct} → score 1
    ]

    for pat, fixed_value in patterns:
        match = re.search(pat, step_text, re.IGNORECASE)
        if match:
            return float(fixed_value) if fixed_value is not None else float(match.group(1))

    # fallback: check for words "correct"/"incorrect"
    lowered = step_text.lower()
    if "correct" in lowered:
        return 1.0
    elif "incorrect" in lowered:
        return 0.0
    elif "mistake" in lowered:
        return 0.0
    elif "issue" in lowered:
        return 0.0
    elif "not" in lowered:
        return 0.0
    elif "inaccurate" in lowered:
        return 0.0
    elif "accurate" in lowered:
        return 1.0
    # if nothing matched
    print("⚠️ Warning score found in step:\n", step_text)
    return 1


def main(input_file, output_file, test_file):
    # load ground truth answers
    ground_truths = []
    with open(test_file, "r", encoding="utf-8") as tf:
        for line in tf:
            obj = json.loads(line)
            sol = obj.get("solution", "")
            ground_truths.append(extract_final_answer(sol))

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            obj = json.loads(line)
            critique = obj.get("critique", "")
            # slice before "final answer"
            idx = critique.lower().find("final answer")
            if idx != -1:
                critique = critique[:idx]

            steps, errors, max_step = split_steps(critique)

            scores = []
            for j in range(1, max_step + 1):
                score = fetch_score(steps[j])

                scores.append(score)

            # extract answers
            model_answer = extract_final_answer(line)   # from input_file entry
            ground_truth = ground_truths[i] if i < len(ground_truths) else None

            # compare (numeric if possible)
            is_correct = 0
            if model_answer is not None and ground_truth is not None:
                try:
                    is_correct = int(float(model_answer) == float(ground_truth))
                except ValueError:
                    is_correct = int(model_answer == ground_truth)

            new_obj = {
                "num_steps": max_step,
                "step_scores": scores,
                "model_answer": model_answer,
                "ground_truth_answer": ground_truth,
                "is_correct": is_correct,
            }

            outfile.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    for i in range(1, 5):
        input_f = f"verifier_outputs/model_{i}_outputs_verified.jsonl"
        output = f"parsed_verifier_critiques/model_{i}_results.jsonl"
        test_f = "data/parsed_data/questions_for_inference.jsonl"
        main(input_f, output, test_f)








