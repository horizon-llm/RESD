"""Plot val metrics curves from CSV files produced by calc_val_metrics.py.

Usage:
    python plot_val_metrics.py <csv_file_or_dir> [<csv_file_or_dir> ...]
    python plot_val_metrics.py results1.csv results2.csv --labels "Method A" "Method B"
    python plot_val_metrics.py /path/to/val_generations/  # looks for val_metrics.csv inside
    python plot_val_metrics.py a.csv b.csv --labels A B --colors red blue
    python plot_val_metrics.py a.csv --groups score          # only plot score
    python plot_val_metrics.py a.csv --combined-order acc score  # acc on left
    python plot_val_metrics.py a.csv --ylabels "Accuracy" "Score"  # y-axis labels per group

A plot_metadata.json file is saved alongside the figures to log which
experiments and options were used.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


METRIC_GROUPS = {
    "acc": ["acc/mean", "acc/best"],
    "score": ["score/mean", "score/best"],
}

METRIC_LABELS = {
    "acc/mean": "mean@k",
    "acc/best": "best@k",
    "score/mean": "mean@k",
    "score/best": "best@k",
}

# Line styles per metric type: same color per method, different style per metric
METRIC_LINESTYLES = {
    "mean": "-",
    "best": "--",
}

METRIC_MARKERS = {
    "mean": "o",
    "best": "s",
}


def resolve_csv(path: Path) -> Path:
    if path.is_file() and path.suffix == ".csv":
        return path
    if path.is_dir():
        candidate = path / "val_metrics.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No CSV found at {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="CSV files or directories containing val_metrics.csv")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend labels for each CSV")
    parser.add_argument("--groups", nargs="+", default=list(METRIC_GROUPS.keys()),
                        choices=list(METRIC_GROUPS.keys()),
                        help="Which metric groups to plot (default: all)")
    parser.add_argument("--combined-order", nargs="+", default=["score", "acc"],
                        choices=list(METRIC_GROUPS.keys()),
                        help="Left-to-right order for the combined plot (default: score acc)")
    parser.add_argument("--colors", nargs="+", default=None,
                        help="Colors for each method (e.g., red blue '#1f77b4' tab:orange)")
    parser.add_argument("--ylabels", nargs="+", default=None,
                        help="Y-axis labels for each metric group, in the order of --groups (e.g., --ylabels 'Accuracy' 'Score')")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for figures (default: same dir as first CSV)")
    args = parser.parse_args()

    csv_paths = [resolve_csv(Path(p)) for p in args.paths]
    labels = args.labels or [p.parent.name for p in csv_paths]
    assert len(labels) == len(csv_paths), "Number of labels must match number of paths"

    def read_csv(path):
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {k: [float(r[k]) for r in rows] for k in rows[0]}

    all_data = {label: read_csv(p) for label, p in zip(labels, csv_paths)}

    # Paper-ready style
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

    out_dir = Path(args.output) if args.output else csv_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ylabels = {}
    if args.ylabels:
        for i, group_name in enumerate(args.groups):
            if i < len(args.ylabels):
                ylabels[group_name] = args.ylabels[i]

    def plot_group(ax, group_name, all_data, show_legend=True):
        metrics = METRIC_GROUPS[group_name]
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for method_idx, (label, data) in enumerate(all_data.items()):
            if args.colors and method_idx < len(args.colors):
                color = args.colors[method_idx]
            else:
                color = default_colors[method_idx % len(default_colors)]
            for metric in metrics:
                metric_type = metric.split("/")[-1]  # "mean" or "best"
                style_label = METRIC_LABELS[metric]
                if len(all_data) > 1:
                    style_label = f"{label} / {style_label}"
                ax.plot(data["step"], data[metric], label=style_label,
                        color=color,
                        linestyle=METRIC_LINESTYLES.get(metric_type, "-"),
                        marker=METRIC_MARKERS.get(metric_type, "o"),
                        linewidth=2.5, markersize=5)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabels.get(group_name, ""))
        ax.grid(True, alpha=0.3, linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_legend:
            ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                      ncol=len(metrics), borderaxespad=0)

    # Individual plots
    for group_name in args.groups:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_group(ax, group_name, all_data)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out_path = out_dir / f"val_{group_name}.{ext}"
            fig.savefig(out_path)
            print(f"Saved to {out_path}")
        plt.close(fig)

    # Combined side-by-side: score (left), acc (right)
    if len(args.groups) > 1:
        combined_groups = [g for g in args.combined_order if g in args.groups]
        fig, axes = plt.subplots(1, len(combined_groups), figsize=(5.5 * len(combined_groups), 4.5))
        if len(combined_groups) == 1:
            axes = [axes]
        for ax, group_name in zip(axes, combined_groups):
            plot_group(ax, group_name, all_data, show_legend=False)
        # Single shared legend across the top
        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, 0.92),
                   ncol=len(handles))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.85, wspace=0.2)
        for ext in ("pdf", "png"):
            out_path = out_dir / f"val_combined.{ext}"
            fig.savefig(out_path)
            print(f"Saved to {out_path}")
        plt.close(fig)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "experiments": [
            {"label": label, "csv_path": str(p.resolve())}
            for label, p in zip(labels, csv_paths)
        ],
        "groups": args.groups,
        "combined_order": args.combined_order,
        "output_dir": str(out_dir.resolve()),
    }
    meta_path = out_dir / "plot_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
