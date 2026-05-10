"""Plot training curves in publication-quality style.

Produces clean figures like the "Sample-Efficient Online Improvement" panel:
- Smooth curves with distinct line styles per method
- Optional shaded confidence bands
- X-axis with "k" suffix formatting (10k, 20k, ...)
- Annotation text box for key takeaways
- No top/right spines, light grid, clean legend

Usage:
    python plot_training_curves.py                          # demo with synthetic data
    python plot_training_curves.py a.csv b.csv c.csv       # multiple CSVs (one per method)
    python plot_training_curves.py a.csv b.csv --labels "GRPO" "SDPO"
    python plot_training_curves.py a.csv b.csv --metric score/mean
    python plot_training_curves.py --single data.csv       # single CSV with all methods as columns
    python plot_training_curves.py a.csv b.csv --colors red blue --linestyles dashed solid

Each CSV is expected to have a "step" column plus one or more metric columns.
When using multiple CSVs (positional args), use --metric to pick which column to plot.
When using --single, each non-step column becomes a separate curve.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------- Style configuration ----------

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}

# Default method styling: (color, linestyle, linewidth, zorder)
# Methods not listed here get auto-assigned from the color cycle.
METHOD_STYLES = {
    "GRPO": {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.2, "zorder": 2},
    "SDPO": {"color": "#4b0082", "linestyle": "-.", "linewidth": 2.2, "zorder": 2},
    "RESD (ours)": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.8, "zorder": 3},
}


# ---------- Helpers ----------

def format_k(x, _pos):
    """Format axis tick as '10k', '20k', etc."""
    if x == 0:
        return "0"
    if x >= 1000:
        return f"{int(x / 1000)}k"
    return str(int(x))


def smooth_curve(y, window=5):
    """Simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def generate_demo_data(n_steps=100):
    """Generate synthetic training curves for demonstration."""
    steps = np.linspace(0, 100000, n_steps)

    np.random.seed(42)
    resd = 1.0 - np.exp(-steps / 20000) * 0.95
    resd += np.random.normal(0, 0.012, n_steps)
    resd = np.clip(resd, 0, 1)

    grpo = 1.0 - np.exp(-steps / 30000) * 0.98
    grpo += np.random.normal(0, 0.015, n_steps)
    grpo = np.clip(grpo, 0, 1)
    grpo *= 0.92

    sdpo = 1.0 - np.exp(-steps / 50000) * 0.99
    sdpo += np.random.normal(0, 0.015, n_steps)
    sdpo = np.clip(sdpo, 0, 1)
    sdpo *= 0.60

    resd_std = 0.03 * np.exp(-steps / 40000) + 0.01

    return {
        "steps": steps,
        "curves": {"GRPO": grpo, "SDPO": sdpo, "RESD (ours)": resd},
        "stds": {"RESD (ours)": resd_std},
    }


# ---------- Main plotting function ----------

def plot_training_curves(
    all_steps: dict,
    all_curves: dict,
    stds: dict = None,
    title: str = "Sample-Efficient Online Improvement",
    xlabel: str = "Training Steps",
    ylabel: str = "Validation Performance",
    annotations: list = None,
    ylim: tuple = None,
    figsize: tuple = (7, 5.5),
    output: str = None,
    smooth: int = 0,
    custom_colors: list = None,
    custom_linestyles: list = None,
    custom_linewidths: list = None,
    max_steps: int = None,
    show_title: bool = True,
    show_legend: bool = True,
    label_rollouts: dict = None,
):
    """
    Plot training curves with publication styling.

    Args:
        all_steps: dict of {method_name: step_array} — allows different x-axes per method
        all_curves: dict of {method_name: y_values}
        stds: dict of {method_name: std_values} for confidence bands
        title: plot title (use None to hide)
        xlabel, ylabel: axis labels
        annotations: list of bullet-point strings for text box
        ylim: (ymin, ymax) or None for auto
        figsize: figure size in inches
        output: output path (without extension); saves .pdf and .png
        smooth: moving-average window size (0 = no smoothing)
        custom_colors: list of colors per method
        custom_linestyles: list of linestyles per method
        custom_linewidths: list of linewidths per method
        max_steps: truncate curves beyond this step value
    """
    stds = stds or {}

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (method, y) in enumerate(all_curves.items()):
            steps = all_steps[method]

            # Truncate if needed
            if max_steps is not None:
                mask = steps <= max_steps
                steps = steps[mask]
                y = y[mask]

            style = METHOD_STYLES.get(method, {
                "color": default_colors[i % len(default_colors)],
                "linestyle": "-",
                "linewidth": 2.0,
                "zorder": 2,
            })

            color = (custom_colors[i] if custom_colors and i < len(custom_colors)
                     else style["color"])
            linestyle = (custom_linestyles[i] if custom_linestyles and i < len(custom_linestyles)
                         else style["linestyle"])
            linewidth = (custom_linewidths[i] if custom_linewidths and i < len(custom_linewidths)
                         else style["linewidth"])

            y_plot = smooth_curve(y, smooth) if smooth > 1 else y

            # Build legend label with optional rollout emphasis
            plot_label = method
            if label_rollouts and method in label_rollouts:
                r = label_rollouts[method]
                plot_label = f"{method} (sample×{r})"

            ax.plot(
                steps, y_plot,
                label=plot_label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=style["zorder"],
            )

            if method in stds:
                std = stds[method]
                if max_steps is not None:
                    std = std[mask]
                ax.fill_between(
                    steps,
                    y_plot - std,
                    y_plot + std,
                    alpha=0.15,
                    color=color,
                    zorder=style["zorder"] - 1,
                    linewidth=0,
                )

        # Axes formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))

        if ylim:
            ax.set_ylim(ylim)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Subtle grid
        ax.grid(True, alpha=0.25, linewidth=0.8, color="#cccccc")
        ax.set_axisbelow(True)

        # Legend
        if show_legend:
            legend = ax.legend(
                frameon=True,
                framealpha=0.95,
                edgecolor="none",
                loc="upper left",
                borderpad=0.6,
                handlelength=2.5,
            )
            legend.set_zorder(10)

        # Title (bold, left-aligned like the reference)
        if show_title and title:
            ax.set_title(title, fontweight="bold", loc="left", pad=12)

        # Annotation text box (bullet points)
        if annotations:
            text = "\n".join(f"•  {a}" for a in annotations)
            bbox_props = dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="#cccccc",
                linewidth=1.0,
                alpha=0.95,
            )
            ax.text(
                0.98, 0.25,
                text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=bbox_props,
                linespacing=1.6,
            )

        fig.tight_layout()

        # Save
        if output:
            out_dir = Path(output)
            out_dir.mkdir(parents=True, exist_ok=True)
            for ext in ("pdf", "png"):
                path = out_dir / f"out.{ext}"
                fig.savefig(path)
                print(f"Saved: {path}")
        else:
            plt.show()

        plt.close(fig)


# ---------- Data loading ----------

def read_csv(path):
    """Read CSV into dict of {column_name: [values]}."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def resolve_csv(path: Path) -> Path:
    """Resolve path: if directory, look for val_metrics.csv inside."""
    if path.is_file() and path.suffix == ".csv":
        return path
    if path.is_dir():
        candidate = path / "val_metrics.csv"
        if candidate.exists():
            return candidate
        # Try val_generations subdirectory
        candidate = path / "val_generations" / "val_metrics.csv"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No CSV found at {path}")


def infer_k(csv_path: Path):
    """Infer k from the first JSONL sibling of the CSV file."""
    from collections import defaultdict as _dd
    jsonl = next(csv_path.parent.glob("*.jsonl"), None)
    if jsonl is None:
        return None
    problems = _dd(int)
    with open(jsonl) as f:
        for line in f:
            entry = json.loads(line)
            problems[entry["input"]] += 1
    return min(problems.values())


def default_label(csv_path: Path) -> str:
    """Derive a label from the CSV path — use grandparent if parent is val_generations."""
    if csv_path.parent.name == "val_generations":
        return csv_path.parent.parent.name
    return csv_path.parent.name


def load_multiple_csvs(paths, labels, metric):
    """
    Load multiple CSVs (one per method). Each must have a 'step' column.
    Returns (all_steps, all_curves, stds) dicts keyed by label.
    """
    all_steps = {}
    all_curves = {}
    stds = {}

    for label, path in zip(labels, paths):
        data = read_csv(path)
        steps = np.array(data["step"])

        # Available metric columns (exclude step, n_problems, and _std columns)
        available = [k for k in data
                     if k not in ("step", "n_problems") and not k.endswith("_std")]
        if metric:
            if metric not in data:
                raise KeyError(f"Metric '{metric}' not found in {path}. Available: {available}")
            metric_key = metric
        else:
            # Default to score/mean if available
            metric_key = "score/mean" if "score/mean" in data else available[0]
            if len(paths) > 1:
                print(f"  Using metric '{metric_key}' (available: {available})")

        y = np.array(data[metric_key])
        all_steps[label] = steps
        all_curves[label] = y

        # Check for std column
        std_key = f"{metric_key}_std"
        if std_key in data:
            stds[label] = np.array(data[std_key])

    return all_steps, all_curves, stds


def load_single_csv(path):
    """Load single CSV where each non-step column is a method."""
    data = read_csv(path)
    step_col = list(data.keys())[0]
    steps = np.array(data[step_col])

    all_steps = {}
    all_curves = {}
    stds = {}

    for col in data:
        if col == step_col or col.endswith("_std"):
            continue
        all_steps[col] = steps
        all_curves[col] = np.array(data[col])

    for col in data:
        if col.endswith("_std"):
            method = col[:-4]
            if method in all_curves:
                stds[method] = np.array(data[col])

    return all_steps, all_curves, stds


def load_wandb_json(path):
    """Load wandb export JSON (list of dicts with _step and metric keys)."""
    with open(path) as f:
        data = json.load(f)
    steps = np.array([d["_step"] for d in data])
    all_steps = {}
    all_curves = {}
    for key in data[0]:
        if key.startswith("_"):
            continue
        all_steps[key] = steps
        all_curves[key] = np.array([d.get(key, np.nan) for d in data])
    return all_steps, all_curves, {}


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves (publication style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python plot_training_curves.py a.csv b.csv --labels GRPO SDPO --metric acc/mean
  python plot_training_curves.py --single combined.csv
  python plot_training_curves.py a.csv b.csv --colors red blue --linestyles dashed solid
""",
    )
    parser.add_argument("paths", nargs="*",
                        help="CSV files or directories (one per method)")
    parser.add_argument("--single", type=str, default=None,
                        help="Single CSV with all methods as columns")
    parser.add_argument("--wandb-json", type=str, default=None,
                        help="Wandb JSON export file")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend labels for each CSV")
    parser.add_argument("--metric", type=str, default=None,
                        help="Which metric column to plot from each CSV (default: first non-step column)")
    parser.add_argument("--title", type=str, default="Sample-Efficient Online Improvement")
    parser.add_argument("--xlabel", type=str, default="Training Steps")
    parser.add_argument("--ylabel", type=str, default="Validation Performance")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--rollouts", nargs="+", type=int, default=None,
                        help="Rollouts per prompt for each method (converts x-axis to total samples)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Truncate curves beyond this x-axis value")
    parser.add_argument("--annotations", nargs="+", default=None,
                        help="Bullet-point annotations for text box")
    parser.add_argument("--colors", nargs="+", default=None,
                        help="Colors per method (e.g., '#1f77b4' red darkgreen)")
    parser.add_argument("--linestyles", nargs="+", default=None,
                        help="Line styles per method (dashed, dashdot, dotted, solid)")
    parser.add_argument("--linewidths", nargs="+", type=float, default=None,
                        help="Line widths per method (e.g., 2.2 2.2 2.8)")
    parser.add_argument("--no-std", action="store_true",
                        help="Disable shaded std bands")
    parser.add_argument("--no-title", action="store_true",
                        help="Hide the plot title")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide the legend")
    parser.add_argument("--label-rollouts", nargs="+", type=int, default=None,
                        help="Rollout counts to show in legend labels (e.g., 1 8 1 -> 'GRPO (sample×8)')")
    parser.add_argument("--smooth", type=int, default=0, help="Smoothing window (0=none)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[7, 5.5])
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory (e.g., figs/my_exp); saves out.pdf and out.png inside")
    args = parser.parse_args()
    ylabel_explicit = "--ylabel" in sys.argv or "-ylabel" in sys.argv

    if args.paths:
        csv_paths = [resolve_csv(Path(p)) for p in args.paths]
        labels = args.labels or [default_label(p) for p in csv_paths]
        assert len(labels) == len(csv_paths), "Number of --labels must match number of paths"
        all_steps, all_curves, stds = load_multiple_csvs(csv_paths, labels, args.metric)
        # Auto-set ylabel to metric@k only if user didn't explicitly pass --ylabel
        if not ylabel_explicit and args.metric:
            k = infer_k(csv_paths[0])
            k_str = str(k) if k else "k"
            args.ylabel = f"{args.metric}@{k_str}"
    elif args.single:
        all_steps, all_curves, stds = load_single_csv(args.single)
    elif args.wandb_json:
        all_steps, all_curves, stds = load_wandb_json(args.wandb_json)
    else:
        print("No data file provided — using demo data.")
        demo = generate_demo_data()
        all_steps = {k: demo["steps"] for k in demo["curves"]}
        all_curves = demo["curves"]
        stds = demo["stds"]
        if args.output is None:
            args.output = "figs/demo_training_curves"
        if args.annotations is None:
            args.annotations = [
                "Early-stage gain",
                "1 rollout / prompt",
                "Better bootstrapping under\n   limited interaction budget",
            ]

    if args.no_std:
        stds = {}

    # Scale x-axis by rollouts (steps -> total samples) and interpolate onto shared grid
    if args.rollouts:
        methods = list(all_steps.keys())
        assert len(args.rollouts) == len(methods), \
            f"Number of --rollouts ({len(args.rollouts)}) must match number of methods ({len(methods)})"
        for method, r in zip(methods, args.rollouts):
            all_steps[method] = all_steps[method] * r

        # Determine shared x-grid max: use --max-steps if given, else max across all methods
        grid_max = args.max_steps if args.max_steps else max(all_steps[m][-1] for m in methods)
        n_grid = int(grid_max) + 1
        grid = np.linspace(0, grid_max, min(n_grid, 500))

        for method in methods:
            x_orig = all_steps[method]
            y_orig = all_curves[method]
            # Interpolate onto shared grid (only within this method's data range)
            valid = grid <= x_orig[-1]
            y_interp = np.interp(grid[valid], x_orig, y_orig)
            all_steps[method] = grid[valid]
            all_curves[method] = y_interp
            if method in stds:
                stds[method] = np.interp(grid[valid], x_orig, stds[method])

        # If --max-steps was used, it's already handled above; clear it so it's not applied again
        args.max_steps = None
        if args.xlabel == "Training Steps":
            args.xlabel = "Number of Samples"

    # Build label_rollouts dict from --label-rollouts
    label_rollouts_dict = None
    if args.label_rollouts:
        methods = list(all_steps.keys())
        assert len(args.label_rollouts) == len(methods), \
            f"Number of --label-rollouts must match number of methods"
        label_rollouts_dict = {
            m: r for m, r in zip(methods, args.label_rollouts) if r > 1
        }

    plot_training_curves(
        all_steps=all_steps,
        all_curves=all_curves,
        stds=stds,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        ylim=tuple(args.ylim) if args.ylim else None,
        max_steps=args.max_steps,
        annotations=args.annotations,
        smooth=args.smooth,
        figsize=tuple(args.figsize),
        output=args.output,
        custom_colors=args.colors,
        custom_linestyles=args.linestyles,
        custom_linewidths=args.linewidths,
        show_title=not args.no_title,
        show_legend=not args.no_legend,
        label_rollouts=label_rollouts_dict,
    )


if __name__ == "__main__":
    main()
