import json
import random
import re

def index_solution(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)

            # Replace 'answer' with 'solution'
            answer = data.pop("answer", "")

            # Replace '####' with 'final answer: '
            answer = answer.replace("####", "final answer:")

            # Split steps by newline
            steps = answer.strip().split("\n")

            numbered_steps = []
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

def distort_number(n: int) -> int:
    distortions = []

    # 1. Add/subtract small
    distortions.append(lambda x: x + random.choice([-10,-5,-3,-2,-1,1,2,3,5,10]))
    # 2. Multiply/divide
    distortions.append(lambda x: int(x * random.choice([0.5, 1.5, 2, 3])))
    # 3. Round to nearest 10
    distortions.append(lambda x: round(x, -1))
    # 4. Swap digits if >9
    distortions.append(lambda x: int(str(x)[::-1]) if x > 9 else x+1)
    # 5. Random replacement in range
    distortions.append(lambda x: random.randint(max(1, x-50), x+50))

    fn = random.choice(distortions)
    distorted = fn(n)

    if distorted == n:
        return n + 1
    return distorted

def ruin_step(step_text):
    # Ruin numbers
    step_text = re.sub(r'\d+', lambda x: str(int(x.group()) + random.randint(-5, 5)), step_text)
    # Swap some arithmetic operators
    step_text = step_text.replace("+", "-").replace("*", "/")
    return step_text

def ruin_steps(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            solution = data.get("solution", "")
            steps = solution.strip().split("\n")

            ruined_steps = []
            final_answer = None
            intermediate_steps = []

            # Separate final answer from intermediate steps
            for step in steps:
                if step.startswith("final answer:"):
                    final_answer = step
                else:
                    # Remove the 'step N: ' prefix
                    m = re.match(r'step \d+:\s*(.*)', step)
                    if m:
                        intermediate_steps.append(m.group(1))
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

def distort_answers(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            solution = obj["solution"]

            if "final answer:" in solution:
                parts = solution.rsplit("final answer:", 1)
                number_str = parts[1].strip()

                # handle commas
                number_str_clean = number_str.replace(",", "")

                # parse
                try:
                    correct = int(number_str_clean)
                except ValueError:
                    correct = float(number_str_clean)

                distorted = distort_number(correct)

                # reformat with commas if original had them
                if "," in number_str:
                    distorted_str = f"{distorted:,}"
                else:
                    distorted_str = str(distorted)

                new_solution = parts[0] + f"final answer: {distorted_str}"
                obj["solution"] = new_solution

            fout.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
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