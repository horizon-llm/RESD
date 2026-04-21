"""Calculate val generation metrics: acc and score (mean@k, best@k).

Usage:
    python calc_val_metrics.py <val_generations_dir>
    python calc_val_metrics.py <single_file.jsonl>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def calc_metrics(filepath, k):
    """Calculate metrics for a single jsonl file. Returns dict of metrics."""
    problems = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            problems[entry["input"]].append(entry)

    acc_means, acc_bests = [], []
    score_means, score_bests = [], []

    for trials in problems.values():
        assert len(trials) >= k, (
            f"Expected at least {k} trials, got {len(trials)} in {filepath}"
        )
        trials = trials[:k]
        accs = [t["acc"] for t in trials]
        scores = [t["score"] for t in trials]

        acc_means.append(sum(accs) / len(accs))
        acc_bests.append(max(accs))
        score_means.append(sum(scores) / len(scores))
        score_bests.append(max(scores))

    n = len(problems)
    return {
        "n_problems": n,
        "acc/mean": sum(acc_means) / n,
        "acc/best": sum(acc_bests) / n,
        "score/mean": sum(score_means) / n,
        "score/best": sum(score_bests) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to val_generations directory or a single jsonl file",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of trials per problem")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to save results as CSV (default: <path>/val_metrics.csv)")
    args = parser.parse_args()

    path = Path(args.path)

    if args.csv is None:
        if path.is_file():
            args.csv = str(path.parent / "val_metrics.csv")
        else:
            args.csv = str(path / "val_metrics.csv")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.jsonl"), key=lambda p: int(p.stem))

    if not files:
        print(f"No jsonl files found in {path}")
        return

    # Header
    print(f"{'step':>6}  {'#problems':>9}  {'acc/mean@'+str(args.k):>12}  "
          f"{'acc/best@'+str(args.k):>12}  {'score/mean@'+str(args.k):>14}  "
          f"{'score/best@'+str(args.k):>14}")
    print("-" * 80)

    rows = []
    for f in files:
        step = f.stem
        m = calc_metrics(f, args.k)
        rows.append({"step": int(step), **m})
        print(f"{step:>6}  {m['n_problems']:>9}  {m['acc/mean']:>12.4f}  "
              f"{m['acc/best']:>12.4f}  {m['score/mean']:>14.4f}  "
              f"{m['score/best']:>14.4f}")

    if args.csv and rows:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
