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
    python plot_reward_std.py <dir> --fields acc --mode stacked
    python plot_reward_std.py <dir> --fields acc --mode line
    python plot_reward_std.py <dir> --mode heatmap
    python plot_reward_std.py <dir> --fields score acc --mode line  # side-by-side
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
    plt.colorbar(im, ax=ax, label="Density")


def plot_stacked(ax, files, field, n_bins=5):
    """Stacked bars of per-step group composition.

    Binary field (all values in {0, 1}): bucket by n/k correct.
    Continuous field (e.g. fraction of test cases passed): bucket the group mean
    into n_bins equal-width bins over [0, 1].
    """
    steps = []
    all_groups_per_step = []  # list of list-of-group-values (only multi-sample groups)

    is_binary = True
    k = None

    for f in files:
        step = int(f.stem)
        groups = load_groups(f, field)
        if not groups:
            continue
        kept = []
        for vals in groups.values():
            if len(vals) < 2:
                continue
            if k is None:
                k = len(vals)
            if is_binary and any(v not in (0, 0.0, 1, 1.0) for v in vals):
                is_binary = False
            kept.append(vals)
        if not kept:
            continue
        steps.append(step)
        all_groups_per_step.append(kept)

    if not steps or k is None:
        return

    x = np.arange(len(steps))
    width = 0.8

    if is_binary:
        n_cats = k + 1
        labels = [f"{nc}/{k} correct" for nc in range(n_cats)]
        def bucket(vals):
            return int(round(sum(vals)))
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}{']' if i == n_bins - 1 else ')'}"
                  for i in range(n_bins)]
        n_cats = n_bins
        def bucket(vals):
            m = float(np.mean(vals))
            # Place m=1.0 in the last bin rather than out of range
            idx = int(m * n_bins)
            if idx >= n_bins:
                idx = n_bins - 1
            return idx

    all_fracs = []
    for kept in all_groups_per_step:
        counts = defaultdict(int)
        for vals in kept:
            counts[bucket(vals)] += 1
        total = sum(counts.values())
        all_fracs.append({c: counts[c] / total for c in counts})

    colors = plt.cm.RdYlGn(np.linspace(0, 1, n_cats))

    bottom = np.zeros(len(steps))
    for c in range(n_cats):
        fracs = np.array([d.get(c, 0.0) for d in all_fracs])
        ax.bar(x, fracs, width, bottom=bottom, label=labels[c], color=colors[c])
        bottom += fracs

    max_ticks = 8
    stride = max(1, int(np.ceil(len(steps) / max_ticks)))
    tick_idx = np.arange(0, len(steps), stride)
    ax.set_xticks(x[tick_idx])
    ax.set_xticklabels([steps[i] for i in tick_idx])
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of groups")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
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
    ax.set_ylim(0, 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)


def render(ax, files, field, mode, bins):
    if mode == "hist":
        plot_hist(ax, files, field, bins)
    elif mode == "stacked":
        plot_stacked(ax, files, field)
    elif mode == "line":
        plot_line(ax, files, field)
    else:
        plot_heatmap(ax, files, field, bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to val_generations dir or a single JSONL file")
    parser.add_argument("--fields", type=str, nargs="+", default=["score", "acc"],
                        help="Fields to compute std over (default: score acc). Pass multiple for side-by-side plot.")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="Only plot specific steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Only plot steps <= this value (truncate curves beyond this point)")
    parser.add_argument("--step-stride", type=int, default=1,
                        help="Keep every N-th step file (e.g. 2 to plot every 8 steps when eval is every 4)")
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

    if args.max_steps is not None:
        files = [f for f in files if int(f.stem) <= args.max_steps]

    if args.step_stride > 1:
        files = files[::args.step_stride]

    plt.rcParams.update({
        "font.size": 22,
        "axes.labelsize": 26,
        "axes.titlesize": 24,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    out_dir = Path(args.output) if args.output else (path if path.is_dir() else path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots, one per field
    for field in args.fields:
        fig, ax = plt.subplots(figsize=(10, 6))
        render(ax, files, field, args.mode, args.bins)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            out_path = out_dir / f"reward_std_{field}_{args.mode}.{ext}"
            fig.savefig(out_path)
            print(f"Saved to {out_path}")
        plt.close(fig)

    # Combined side-by-side when multiple fields
    if len(args.fields) > 1:
        combined_rc = {
            "font.size": 24,
            "axes.labelsize": 28,
            "axes.titlesize": 26,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
        }
        with plt.rc_context(combined_rc):
            fig, axes = plt.subplots(1, len(args.fields), figsize=(10 * len(args.fields), 6))
            if len(args.fields) == 1:
                axes = [axes]
            for ax, field in zip(axes, args.fields):
                render(ax, files, field, args.mode, args.bins)
            fig.tight_layout()
            for ext in ("pdf", "png"):
                out_path = out_dir / f"reward_std_{args.mode}_combined.{ext}"
                fig.savefig(out_path)
                print(f"Saved to {out_path}")
            plt.close(fig)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "input_path": str(path.resolve()),
        "fields": args.fields,
        "mode": args.mode,
        "steps": [int(f.stem) for f in files],
        "max_steps": args.max_steps,
        "bins": args.bins,
        "output_dir": str(out_dir.resolve()),
    }
    meta_path = out_dir / "plot_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
