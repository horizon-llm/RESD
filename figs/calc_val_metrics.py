"""Calculate val generation metrics: acc and score (mean@k, best@k).

Usage:
    python calc_val_metrics.py <val_generations_dir>
    python calc_val_metrics.py <single_file.jsonl>
"""

import argparse
import csv
import json
import statistics
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

    def _std(values, mean):
        if n < 2:
            return 0.0
        return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5

    acc_mean = sum(acc_means) / n
    acc_best = sum(acc_bests) / n
    score_mean = sum(score_means) / n
    score_best = sum(score_bests) / n
    return {
        "n_problems": n,
        "acc/mean": acc_mean,
        "acc/best": acc_best,
        "score/mean": score_mean,
        "score/best": score_best,
        "acc/mean_std": _std(acc_means, acc_mean),
        "acc/best_std": _std(acc_bests, acc_best),
        "score/mean_std": _std(score_means, score_mean),
        "score/best_std": _std(score_bests, score_best),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="Path to val_generations directory or a single jsonl file",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of trials per problem")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Only consider steps <= this value when computing the summary "
                             "(final/best/overall/mean/std). The CSV still contains all steps.")
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
    print(f"{'step':>6}  {'#problems':>9}  {'acc/mean@'+str(args.k):>18}  "
          f"{'acc/best@'+str(args.k):>18}  {'score/mean@'+str(args.k):>20}  "
          f"{'score/best@'+str(args.k):>20}")
    print("-" * 104)

    rows = []
    for f in files:
        step = f.stem
        m = calc_metrics(f, args.k)
        rows.append({"step": int(step), **m})
        print(f"{step:>6}  {m['n_problems']:>9}  "
              f"{m['acc/mean']:>10.4f}±{m['acc/mean_std']:<7.4f}  "
              f"{m['acc/best']:>10.4f}±{m['acc/best_std']:<7.4f}  "
              f"{m['score/mean']:>12.4f}±{m['score/mean_std']:<7.4f}  "
              f"{m['score/best']:>12.4f}±{m['score/best_std']:<7.4f}")

    if rows:
        metric_keys = ["acc/mean", "acc/best", "score/mean", "score/best"]
        summary_rows = (
            [r for r in rows if r["step"] <= args.max_steps]
            if args.max_steps is not None else rows
        )
        if not summary_rows:
            print(f"\nNo steps <= {args.max_steps}; skipping summary.")
            return
        if args.max_steps is not None:
            print(f"\n(summary truncated to steps <= {args.max_steps}: "
                  f"{len(summary_rows)}/{len(rows)} steps)")
        final = summary_rows[-1]
        best = {k: max(r[k] for r in summary_rows) for k in metric_keys}
        best_steps = {k: max(summary_rows, key=lambda r: r[k])["step"] for k in metric_keys}
        base = next((r for r in summary_rows if r["step"] == 0), None)
        mean_over_steps = {k: statistics.fmean(r[k] for r in summary_rows) for k in metric_keys}
        std_over_steps = {
            k: statistics.stdev(r[k] for r in summary_rows) if len(summary_rows) > 1 else 0.0
            for k in metric_keys
        }

        # Rank-based overall best: for each metric, rank steps (1 = worst, higher = better,
        # ties averaged), then pick the step with the highest average rank across metrics.
        rank_totals = {r["step"]: 0.0 for r in summary_rows}
        for k in metric_keys:
            sorted_rows = sorted(summary_rows, key=lambda r: r[k])
            i = 0
            while i < len(sorted_rows):
                j = i
                while j + 1 < len(sorted_rows) and sorted_rows[j + 1][k] == sorted_rows[i][k]:
                    j += 1
                avg_rank = (i + j) / 2 + 1
                for r in sorted_rows[i:j + 1]:
                    rank_totals[r["step"]] += avg_rank
                i = j + 1
        best_overall_step = max(rank_totals, key=rank_totals.get)
        best_overall_row = next(r for r in summary_rows if r["step"] == best_overall_step)

        print()
        header = f"{'':>6}  {'':>9}  {'acc/mean':>12}  {'acc/best':>12}  {'score/mean':>14}  {'score/best':>14}"
        print(header)
        if base is not None:
            print(f"{'base':>6}  {'step=0':>9}  {base['acc/mean']:>12.4f}  "
                  f"{base['acc/best']:>12.4f}  {base['score/mean']:>14.4f}  "
                  f"{base['score/best']:>14.4f}")
        step_tag = "step=" + str(final["step"])
        print(f"{'final':>6}  {step_tag:>9}  {final['acc/mean']:>12.4f}  "
              f"{final['acc/best']:>12.4f}  {final['score/mean']:>14.4f}  "
              f"{final['score/best']:>14.4f}")
        best_step_tags = [str(best_steps[k]) for k in metric_keys]
        print(f"{'best':>6}  {'step=':>9}  {best_step_tags[0]:>12}  {best_step_tags[1]:>12}  "
              f"{best_step_tags[2]:>14}  {best_step_tags[3]:>14}")
        print(f"{'':>6}  {'':>9}  {best['acc/mean']:>12.4f}  "
              f"{best['acc/best']:>12.4f}  {best['score/mean']:>14.4f}  "
              f"{best['score/best']:>14.4f}")
        overall_tag = "step=" + str(best_overall_row["step"])
        print(f"{'overall':>6}  {overall_tag:>9}  {best_overall_row['acc/mean']:>12.4f}  "
              f"{best_overall_row['acc/best']:>12.4f}  {best_overall_row['score/mean']:>14.4f}  "
              f"{best_overall_row['score/best']:>14.4f}")
        print(f"{'mean':>6}  {'':>9}  {mean_over_steps['acc/mean']:>12.4f}  "
              f"{mean_over_steps['acc/best']:>12.4f}  {mean_over_steps['score/mean']:>14.4f}  "
              f"{mean_over_steps['score/best']:>14.4f}")
        print(f"{'std':>6}  {'':>9}  {std_over_steps['acc/mean']:>12.4f}  "
              f"{std_over_steps['acc/best']:>12.4f}  {std_over_steps['score/mean']:>14.4f}  "
              f"{std_over_steps['score/best']:>14.4f}")

    if args.csv and rows:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved to {csv_path}")

        summary = {
            "final": {"step": final["step"], **{k: final[k] for k in metric_keys}},
            "best": {k: {"step": best_steps[k], "value": best[k]} for k in metric_keys},
            "best_overall": {
                "step": best_overall_row["step"],
                **{k: best_overall_row[k] for k in metric_keys},
            },
            "over_steps": {
                k: {"mean": mean_over_steps[k], "std": std_over_steps[k]}
                for k in metric_keys
            },
        }
        if base is not None:
            summary["base"] = {"step": 0, **{k: base[k] for k in metric_keys}}
        summary_path = csv_path.with_name(csv_path.stem + "_summary.json")
        with open(summary_path, "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
