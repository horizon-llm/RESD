"""Plot per-prompt reward std distribution from val_generations JSONL files.

Groups samples by input text, computes reward std within each group,
and plots the distribution of those std values.

Modes:
  hist    - overlaid histograms (good for comparing a few steps)
  heatmap - 2D heatmap (x=std bins, y=step, color=density; good for many steps)
  stacked - stacked bar chart showing fraction of all-correct, all-wrong, and
            mixed groups per step (best for binary fields like acc)
  line    - line plot tracking fraction of each category and mean std over steps

Usage:
    python plot_reward_std.py <val_generations_dir>
    python plot_reward_std.py <single_file.jsonl>
    python plot_reward_std.py <dir> --steps 0 100 200
    python plot_reward_std.py <dir> --field acc --mode stacked
    python plot_reward_std.py <dir> --field acc --mode line
    python plot_reward_std.py <dir> --mode heatmap
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_groups(filepath, field="score"):
    """Load per-prompt grouped values from a single JSONL file."""
    groups = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            groups[entry["input"]].append(entry[field])
    return groups


def compute_group_stds(filepath, field="score"):
    """Compute per-prompt reward std from a single JSONL file."""
    groups = load_groups(filepath, field)
    stds = []
    for vals in groups.values():
        if len(vals) > 1:
            stds.append(np.std(vals, ddof=0))
    return np.array(stds)


def plot_hist(ax, files, field, bins):
    for f in files:
        step = int(f.stem)
        stds = compute_group_stds(f, field=field)
        if len(stds) == 0:
            continue
        ax.hist(stds, bins=bins, alpha=0.5, label=f"step {step} (mean={stds.mean():.4f})")

    ax.set_xlabel(f"Per-prompt {field} std")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of per-prompt {field} std across groups")
    ax.legend()


def plot_heatmap(ax, files, field, bins):
    all_stds = {}
    for f in files:
        step = int(f.stem)
        stds = compute_group_stds(f, field=field)
        if len(stds) > 0:
            all_stds[step] = stds

    if not all_stds:
        return

    steps = sorted(all_stds.keys())
    global_min = min(s.min() for s in all_stds.values())
    global_max = max(s.max() for s in all_stds.values())
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    density_matrix = np.zeros((len(steps), bins))
    for i, step in enumerate(steps):
        counts, _ = np.histogram(all_stds[step], bins=bin_edges, density=True)
        density_matrix[i] = counts

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    im = ax.imshow(
        density_matrix,
        aspect="auto",
        origin="lower",
        extent=[bin_centers[0], bin_centers[-1], -0.5, len(steps) - 0.5],
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps)
    ax.set_xlabel(f"Per-prompt {field} std")
    ax.set_ylabel("Step")
    ax.set_title(f"Per-prompt {field} std distribution over training")
    plt.colorbar(im, ax=ax, label="Density")


def plot_stacked(ax, files, field):
    steps = []
    # For group size k, possible number of correct: 0, 1, ..., k
    # We detect k from the first file with data
    k = None
    all_fracs = []  # list of dicts: {n_correct: fraction}

    for f in files:
        step = int(f.stem)
        groups = load_groups(f, field)
        if not groups:
            continue

        counts = defaultdict(int)
        for vals in groups.values():
            if len(vals) < 2:
                continue
            if k is None:
                k = len(vals)
            n_correct = int(round(sum(vals)))
            counts[n_correct] += 1

        total = sum(counts.values())
        if total == 0:
            continue
        steps.append(step)
        all_fracs.append({nc: counts[nc] / total for nc in counts})

    if not steps or k is None:
        return

    x = np.arange(len(steps))
    width = 0.8

    # Color gradient: red (0/k) -> orange -> yellow -> light green -> green (k/k)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, k + 1))

    bottom = np.zeros(len(steps))
    for nc in range(k + 1):
        fracs = np.array([d.get(nc, 0.0) for d in all_fracs])
        ax.bar(x, fracs, width, bottom=bottom, label=f"{nc}/{k} correct", color=colors[nc])
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of groups")
    ax.set_title(f"Per-prompt {field} group composition over training")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim(0, 1)


def plot_line(ax, files, field):
    steps = []
    frac_all_correct = []
    frac_all_wrong = []
    frac_mixed = []
    mean_stds = []

    for f in files:
        step = int(f.stem)
        groups = load_groups(f, field)
        if not groups:
            continue

        n_all_correct = 0
        n_all_wrong = 0
        n_mixed = 0
        stds = []
        for vals in groups.values():
            if len(vals) < 2:
                continue
            mean_val = np.mean(vals)
            if mean_val == 1.0:
                n_all_correct += 1
            elif mean_val == 0.0:
                n_all_wrong += 1
            else:
                n_mixed += 1
            stds.append(np.std(vals, ddof=0))

        total = n_all_correct + n_all_wrong + n_mixed
        if total == 0:
            continue
        steps.append(step)
        frac_all_correct.append(n_all_correct / total)
        frac_all_wrong.append(n_all_wrong / total)
        frac_mixed.append(n_mixed / total)
        mean_stds.append(np.mean(stds))

    if not steps:
        return

    ax.plot(steps, frac_all_correct, "o-", color="#2ecc71", label="All correct", linewidth=2, markersize=5)
    ax.plot(steps, frac_mixed, "s-", color="#f39c12", label="Mixed", linewidth=2, markersize=5)
    ax.plot(steps, frac_all_wrong, "^-", color="#e74c3c", label="All wrong", linewidth=2, markersize=5)

    ax2 = ax.twinx()
    ax2.plot(steps, mean_stds, "D--", color="#3498db", label="Mean std", linewidth=2, markersize=5)
    ax2.set_ylabel(f"Mean {field} std", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of groups")
    ax.set_title(f"Per-prompt {field} diversity over training")
    ax.set_ylim(0, 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to val_generations dir or a single JSONL file")
    parser.add_argument("--field", type=str, default="score", help="Field to compute std over (default: score)")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Only plot specific steps")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    parser.add_argument("--mode", type=str, default="heatmap", choices=["hist", "heatmap", "stacked", "line"],
                        help="Plot mode (default: heatmap)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for figures (default: same dir as input path)")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.jsonl"), key=lambda p: int(p.stem))

    if not files:
        print(f"No JSONL files found in {path}")
        return

    if args.steps is not None:
        step_set = set(args.steps)
        files = [f for f in files if int(f.stem) in step_set]

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    if args.mode == "hist":
        plot_hist(ax, files, args.field, args.bins)
    elif args.mode == "stacked":
        plot_stacked(ax, files, args.field)
    elif args.mode == "line":
        plot_line(ax, files, args.field)
    else:
        plot_heatmap(ax, files, args.field, args.bins)

    plt.tight_layout()

    out_dir = Path(args.output) if args.output else (path if path.is_dir() else path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):
        out_path = out_dir / f"reward_std_{args.field}_{args.mode}.{ext}"
        fig.savefig(out_path)
        print(f"Saved to {out_path}")
    plt.close(fig)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "input_path": str(path.resolve()),
        "field": args.field,
        "mode": args.mode,
        "steps": [int(f.stem) for f in files],
        "bins": args.bins,
        "output_dir": str(out_dir.resolve()),
    }
    meta_path = out_dir / "plot_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
