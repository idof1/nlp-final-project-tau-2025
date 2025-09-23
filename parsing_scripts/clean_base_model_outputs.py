import json
import re

input_file = "../outputs_phi_base.jsonl"
output_file = "../clean_base_outputs.jsonl"

def main():
    final_answer_regex = re.compile(r"(final answer:\s*[\d,]+)")

    with open(input_file, "r", encoding="utf-8") as infile, \
            open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line)
            solution = data.get("solution", "")

            # Step 1: Remove leading "answer:\n"
            if solution.startswith("answer:\n"):
                solution = solution[len("answer:\n"):]

            # Step 2: Keep only up to and including first "final answer: <number>"
            match = final_answer_regex.search(solution)
            if match:
                solution = solution[:match.end()]

            # Step 3: Collapse multiple newlines into one
            solution = re.sub(r"\n{2,}", "\n", solution)

            data["solution"] = solution
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()