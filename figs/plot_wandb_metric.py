"""Plot a wandb metric from downloaded history.jsonl, comparing across runs.

Usage:
    python figs/plot_wandb_metric.py actor/pg_loss run_dir1 run_dir2
    python figs/plot_wandb_metric.py actor/pg_loss run_dir1 run_dir2 --labels "GRPO" "SDPO"
    python figs/plot_wandb_metric.py timing_s/step run_dir1 --delta  # apply timing delta correction
    python figs/plot_wandb_metric.py actor/pg_loss run_dir1 --smooth 10
    python figs/plot_wandb_metric.py actor/pg_loss run_dir1 -o figs/output.pdf
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metric(run_dir: Path, metric: str, delta: bool, updates_per_batch: Optional[int]):
    history_path = run_dir / "history.jsonl"
    config_path = run_dir / "config.json"

    if not history_path.exists():
        raise FileNotFoundError(f"history.jsonl not found in {run_dir}")

    with open(history_path) as f:
        data = [json.loads(line) for line in f]

    if delta:
        if updates_per_batch is None and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            updates_per_batch = config.get("trainer", {}).get("max_updates_per_batch", 1)
        elif updates_per_batch is None:
            updates_per_batch = 1

    steps = []
    values = []
    prev = 0.0
    timing_step_count = 0

    for d in data:
        step = d.get("_step")
        v = d.get(metric)
        if step is None or v is None:
            continue

        if delta:
            is_batch_start = (timing_step_count % updates_per_batch == 0)
            timing_step_count += 1
            if is_batch_start:
                val = v
            else:
                val = v - prev
            prev = v
            v = val

        steps.append(step)
        values.append(v)

    return np.array(steps), np.array(values)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def get_run_label(run_dir: Path) -> str:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("name"):
            return meta["name"]
    return run_dir.name


def main():
    parser = argparse.ArgumentParser(description="Plot a wandb metric comparing across runs.")
    parser.add_argument("metric", help="Metric key to plot (e.g., actor/pg_loss)")
    parser.add_argument("run_dirs", nargs="+", help="Paths to downloaded wandb run directories")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels for each run")
    parser.add_argument("--delta", action="store_true",
                        help="Apply timing delta correction (for cumulative timing_s/* metrics)")
    parser.add_argument("--updates-per-batch", type=int, default=None,
                        help="Override updates_per_batch for delta correction")
    parser.add_argument("--smooth", type=int, default=1, help="Smoothing window size (default: 1, no smoothing)")
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate x-axis at this step")
    parser.add_argument("--colors", nargs="+", default=None, help="Colors for each run")
    parser.add_argument("--ylabel", type=str, default=None, help="Y-axis label (default: metric name)")
    parser.add_argument("--xlabel", type=str, default="Step", help="X-axis label")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--figsize", nargs=2, type=float, default=[6, 4], help="Figure size (width height)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: figs/<metric_name>.pdf)")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    labels = args.labels or [get_run_label(d) for d in run_dirs]
    assert len(labels) == len(run_dirs), "Number of labels must match number of run directories"

    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    colors = args.colors or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        steps, values = load_metric(run_dir, args.metric, args.delta, args.updates_per_batch)

        if args.max_steps is not None:
            mask = steps <= args.max_steps
            steps = steps[mask]
            values = values[mask]

        values_smoothed = smooth(values, args.smooth)
        color = colors[i % len(colors)]

        if args.smooth > 1:
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            ax.plot(steps, values_smoothed, color=color, label=label, linewidth=1.5)
        else:
            ax.plot(steps, values, color=color, label=label, linewidth=1.5)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel or args.metric)
    if args.title:
        ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if args.output:
        out_path = Path(args.output)
    else:
        safe_name = args.metric.replace("/", "_")
        out_path = Path("figs") / f"{safe_name}.pdf"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
