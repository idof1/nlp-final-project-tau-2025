"""Utility script to clean raw base model outputs into a normalized JSONL format."""

import json
import logging
import re
from typing import Final, Pattern

INPUT_FILE: Final[str] = "../outputs_phi_base.jsonl"
OUTPUT_FILE: Final[str] = "../clean_base_outputs.jsonl"

logger = logging.getLogger(__name__)


def main() -> None:
    """Clean raw solutions by trimming headers, final answers, and collapsing newlines."""
    logging.basicConfig(level=logging.INFO)
    final_answer_regex: Pattern[str] = re.compile(r"(final answer:\s*[\d,]+)")

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(OUTPUT_FILE, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            data = json.loads(line)
            solution = data.get("solution", "")

            # Step 1: Remove leading "answer:\n"
            if solution.startswith("answer:\n"):
                solution = solution[len("answer:\n") :]

            # Step 2: Keep only up to and including first "final answer: <number>"
            match = final_answer_regex.search(solution)
            if match:
                solution = solution[: match.end()]

            # Step 3: Collapse multiple newlines into one
            solution = re.sub(r"\n{2,}", "\n", solution)

            data["solution"] = solution
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info("Finished cleaning base model outputs from %s to %s", INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
