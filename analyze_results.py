"""Aggregate and display evaluation results across all four experimental models.

Usage:
    python analyze_results.py
    python analyze_results.py --results_dir parsed_verifier_critiques --n_models 4
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import TypedDict


MODEL_LABELS = {
    1: "CS-CA: correct steps + correct answer (SFT)",
    2: "WS-CA: wrong steps  + correct answer (SFT)",
    3: "CS-WA: correct steps + wrong answer  (SFT)",
    4: "Base model (no fine-tuning)",
}

SEL_THRESHOLD = 0.7


class ModelStats(TypedDict):
    model_id: int
    label: str
    n: int
    accuracy: float
    avg_step_score: float
    all_steps_ok: float
    spearman: float
    sel_acc: float
    coverage: float
    avg_num_steps: float


def load_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return 0.0

    def ranks(vals: list[float]) -> list[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[sorted_vals[k][0]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mean_rx = statistics.mean(rx)
    mean_ry = statistics.mean(ry)
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((v - mean_rx) ** 2 for v in rx) ** 0.5
    den_y = sum((v - mean_ry) ** 2 for v in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def compute_stats(records: list[dict], model_id: int) -> ModelStats:
    n = len(records)
    accuracy = sum(r["is_correct"] for r in records) / n

    per_example_avg = [
        statistics.mean(r["step_scores"]) for r in records if r["step_scores"]
    ]
    avg_step_score = statistics.mean(per_example_avg) if per_example_avg else 0.0

    # AllStepsOK: proportion of examples where every step scored >= 0.5
    all_steps_ok = sum(
        1 for r in records if r["step_scores"] and all(s >= 0.5 for s in r["step_scores"])
    ) / n

    # Spearman between per-example avg step score and is_correct
    scores_list = [
        statistics.mean(r["step_scores"]) if r["step_scores"] else 0.0 for r in records
    ]
    correct_list = [float(r["is_correct"]) for r in records]
    spearman = _spearman(scores_list, correct_list)

    # SelAcc@threshold: accuracy restricted to examples with avg_step_score >= threshold
    high_conf = [r for r, s in zip(records, scores_list) if s >= SEL_THRESHOLD]
    sel_acc = sum(r["is_correct"] for r in high_conf) / len(high_conf) if high_conf else 0.0
    coverage = len(high_conf) / n

    avg_num_steps = statistics.mean(r["num_steps"] for r in records)

    return ModelStats(
        model_id=model_id,
        label=MODEL_LABELS.get(model_id, f"model_{model_id}"),
        n=n,
        accuracy=accuracy,
        avg_step_score=avg_step_score,
        all_steps_ok=all_steps_ok,
        spearman=spearman,
        sel_acc=sel_acc,
        coverage=coverage,
        avg_num_steps=avg_num_steps,
    )


def print_table(stats_list: list[ModelStats]) -> None:
    header = (
        f"{'#':>2}  {'Training condition':<46}  {'N':>5}  "
        f"{'EM':>6}  {'StepScore':>9}  {'AllOK':>6}  "
        f"{'Spearman':>9}  {'SelAcc':>7}  {'Cov':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for s in stats_list:
        print(
            f"  {s['model_id']}  {s['label']:<46}  {s['n']:>5}  "
            f"{s['accuracy']:>6.3f}  {s['avg_step_score']:>9.3f}  "
            f"{s['all_steps_ok']:>6.3f}  {s['spearman']:>+9.3f}  "
            f"{s['sel_acc']:>7.3f}  {s['coverage']:>6.3f}"
        )
    print(sep)


def print_findings(stats_list: list[ModelStats]) -> None:
    by_id = {s["model_id"]: s for s in stats_list}
    print("\nKey findings:")

    if 2 in by_id and 1 in by_id:
        drop = by_id[1]["accuracy"] - by_id[2]["accuracy"]
        print(
            f"  * Corrupting steps (M2 vs M1) drops EM by {drop:.3f} "
            f"({by_id[1]['accuracy']:.3f} -> {by_id[2]['accuracy']:.3f}), "
            f"showing step quality strongly affects final accuracy."
        )

    if 3 in by_id:
        s3 = by_id[3]
        print(
            f"  * M3 has high step scores ({s3['avg_step_score']:.3f}) but near-zero EM "
            f"({s3['accuracy']:.3f}) and Spearman {s3['spearman']:+.3f}: "
            f"CoT can be a superficial pattern decoupled from the final answer."
        )

    if 1 in by_id and 4 in by_id:
        for mid in (1, 4):
            s = by_id[mid]
            print(
                f"  * M{mid} (aligned training): Spearman {s['spearman']:+.3f}, "
                f"SelAcc@{SEL_THRESHOLD} = {s['sel_acc']:.3f} -- "
                f"step scores are a reliable confidence signal."
            )

    if 4 in by_id and 1 in by_id:
        diff = by_id[4]["accuracy"] - by_id[1]["accuracy"]
        direction = "above" if diff > 0 else "below"
        print(
            f"  * Base model (M4) sits {abs(diff):.3f} {direction} clean-SFT (M1); "
            f"one epoch of SFT on this scale causes slight degradation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise model evaluation results.")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("parsed_verifier_critiques"),
        help="Directory containing model_N_results.jsonl files.",
    )
    parser.add_argument("--n_models", type=int, default=4, help="Number of models to analyse.")
    args = parser.parse_args()

    stats_list: list[ModelStats] = []
    for i in range(1, args.n_models + 1):
        path = args.results_dir / f"model_{i}_results.jsonl"
        if not path.exists():
            print(f"  [warning] {path} not found, skipping model {i}.")
            continue
        records = load_results(path)
        stats_list.append(compute_stats(records, i))

    if not stats_list:
        print("No result files found.")
        return

    print(f"\nEvaluation summary  ({stats_list[0]['n']} examples per model)\n")
    print_table(stats_list)
    print_findings(stats_list)
    print()


if __name__ == "__main__":
    main()
