"""Visualize sokoban rollouts from training dumps.

Parses rollout jsonl files and renders:
1. The initial puzzle grid (with colored symbols)
2. Step-by-step simulation of the predicted moves
3. Step-by-step simulation of the ground-truth moves
4. Summary stats (score, feedback)

Usage:
    python scripts/visualize_sokoban_rollouts.py <rollouts_dir> [--step 1] [--sample 0] [--no-gt]
    python scripts/visualize_sokoban_rollouts.py <rollouts_dir> --step 1 --sample 0,1,2
    python scripts/visualize_sokoban_rollouts.py <rollouts_dir> --step 1 --all
    python scripts/visualize_sokoban_rollouts.py <rollouts_dir> --step 1 --all --save
"""

import argparse
import json
import re
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np

# Import sokoban engine directly to avoid package-level __init__ issues
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "selfevolve" / "sdpo_fewshot" / "feedback" / "reasoning_gym_games"))
from sokoban import Game, _get_board_string, _is_solved

# ── ANSI colors ──────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"

SYMBOL_COLORS = {
    "+": DIM,         # wall
    "@": YELLOW,      # box
    "*": CYAN,        # player
    "%": GREEN,       # player on goal
    "X": RED,         # goal
    "$": GREEN,       # box on goal
    "-": DIM,         # empty
}


def colorize_grid(board_str: str) -> str:
    """Add ANSI colors to a sokoban board string."""
    out = []
    for ch in board_str:
        color = SYMBOL_COLORS.get(ch, "")
        if color:
            out.append(f"{color}{ch}{RESET}")
        else:
            out.append(ch)
    return "".join(out)


def extract_grid_from_input(input_text: str) -> list[list[str]]:
    """Extract the sokoban grid from the prompt text."""
    lines = input_text.split("\n")
    grid_lines = []
    in_grid = False
    for line in lines:
        stripped = line.strip()
        # Grid lines start with '+' (wall) and contain sokoban symbols
        if stripped and stripped[0] == "+" and all(c in "+ @*%X$-\t " for c in stripped):
            in_grid = True
            # Split by whitespace to get individual cells
            cells = stripped.split()
            grid_lines.append(cells)
        elif in_grid:
            # Check if this is still a grid line (interior rows start with '+')
            if stripped and stripped[0] == "+" and all(c in "+ @*%X$-\t " for c in stripped):
                cells = stripped.split()
                grid_lines.append(cells)
            else:
                break
    return grid_lines


def simulate_moves(grid_list: list[list[str]], moves: str) -> list[tuple[str, np.ndarray, str]]:
    """Simulate moves and return list of (move, matrix, state) after each step."""
    matrix = np.array(grid_list)
    h, w = matrix.shape
    game = Game(height=h, width=w)
    game.load_puzzle_matrix(matrix)

    snapshots = []
    # Initial state
    snapshots.append(("START", game.get_matrix().copy(), game.get_curr_state()))

    for move_char in moves:
        game.player.update(key=move_char)
        snapshots.append((move_char, game.get_matrix().copy(), game.get_curr_state()))

    return snapshots


def print_header(text: str):
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")


def print_subheader(text: str):
    print(f"\n{BOLD}── {text} ──{RESET}")


def print_grid(matrix: np.ndarray, label: str = ""):
    """Print a colored grid with optional label."""
    if label:
        print(f"  {DIM}{label}{RESET}")
    board_str = _get_board_string(matrix)
    colored = colorize_grid(board_str)
    for line in colored.split("\n"):
        print(f"    {line}")


def print_side_by_side(snapshots: list[tuple[str, np.ndarray, str]], title: str, max_cols: int = 4):
    """Print grid snapshots side-by-side in groups."""
    print_subheader(title)

    for group_start in range(0, len(snapshots), max_cols):
        group = snapshots[group_start : group_start + max_cols]

        # Header row with move labels
        labels = []
        for i, (move, matrix, state) in enumerate(group):
            step_idx = group_start + i
            if move == "START":
                label = f"{'Initial':^{matrix.shape[1] * 2 - 1}}"
            else:
                solved_marker = f" {GREEN}✓{RESET}" if _is_solved(state) else ""
                label = f"{'Step ' + str(step_idx) + ': ' + move:^{matrix.shape[1] * 2 - 1}}{solved_marker}"
            labels.append(label)
        print("    " + "   ".join(labels))

        # Grid rows
        height = group[0][1].shape[0]
        for r in range(height):
            row_parts = []
            for move, matrix, state in group:
                row_str = " ".join(matrix[r])
                row_parts.append(colorize_grid(row_str))
            print("    " + "   ".join(row_parts))
        print()


def visualize_sample(sample: dict, sample_idx: int, show_gt: bool = True):
    """Visualize a single rollout sample."""
    input_text = sample["input"]
    pred = sample.get("pred", "")
    gts = sample.get("gts", "")
    score = sample.get("score", 0)
    feedback = sample.get("feedback", "")
    step = sample.get("step", "?")
    acc = sample.get("acc", "?")

    # Extract grid
    grid_list = extract_grid_from_input(input_text)
    if not grid_list:
        print(f"  {RED}Could not extract grid from input.{RESET}")
        return

    print_header(f"Sample {sample_idx} | Step {step} | Score: {score:.2f} | Acc: {acc}")

    # Legend
    print(f"\n  {DIM}Legend:{RESET} "
          f"{DIM}+{RESET}=wall  "
          f"{YELLOW}@{RESET}=box  "
          f"{CYAN}*{RESET}=player  "
          f"{GREEN}%{RESET}=player+goal  "
          f"{RED}X{RESET}=goal  "
          f"{GREEN}${RESET}=box+goal  "
          f"{DIM}-{RESET}=empty")

    # Simulate predicted moves
    if pred:
        try:
            pred_snapshots = simulate_moves(grid_list, pred)
            print_side_by_side(pred_snapshots, f"Predicted: {YELLOW}{pred}{RESET} ({len(pred)} moves)")
        except Exception as e:
            print(f"  {RED}Error simulating predicted moves: {e}{RESET}")

    # Simulate ground-truth moves
    if show_gt and gts:
        try:
            gt_snapshots = simulate_moves(grid_list, gts)
            print_side_by_side(gt_snapshots, f"Ground Truth: {GREEN}{gts}{RESET} ({len(gts)} moves)")
        except Exception as e:
            print(f"  {RED}Error simulating GT moves: {e}{RESET}")

    # Feedback
    if feedback:
        print_subheader("Feedback")
        for line in feedback.split("\n"):
            print(f"    {line}")

    print()



# ── HTML output ──────────────────────────────────────────────────────────────

HTML_SYMBOL_COLORS = {
    "+": "#888",       # wall - gray
    "@": "#e6a817",    # box - yellow
    "*": "#00bcd4",    # player - cyan
    "%": "#4caf50",    # player on goal - green
    "X": "#e74c3c",    # goal - red
    "$": "#4caf50",    # box on goal - green
    "-": "#555",       # empty - dim
}

HTML_HEADER = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sokoban Rollout Visualization</title>
<style>
body { background: #1e1e1e; color: #ddd; font-family: 'Consolas', 'Monaco', monospace; padding: 20px; }
h1 { color: #bb86fc; border-bottom: 2px solid #bb86fc; padding-bottom: 8px; }
h2 { color: #03dac6; margin-top: 30px; }
h3 { color: #cf6679; }
.summary { background: #2d2d2d; padding: 10px 16px; border-radius: 6px; margin: 8px 0; display: inline-block; }
.legend { color: #aaa; margin: 8px 0; }
.grids-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0; }
.grid-box { background: #2a2a2a; border-radius: 6px; padding: 8px 12px; text-align: center; }
.grid-box .label { font-size: 12px; color: #aaa; margin-bottom: 4px; }
.grid-box .label .solved { color: #4caf50; font-weight: bold; }
.grid-box pre { margin: 0; line-height: 1.4; font-size: 14px; }
.feedback { background: #2d2d2d; padding: 10px 16px; border-radius: 6px; margin: 8px 0; white-space: pre-wrap; color: #ffab91; }
.sample-divider { border-top: 1px solid #444; margin: 24px 0; }
.score-good { color: #4caf50; }
.score-mid { color: #e6a817; }
.score-bad { color: #e74c3c; }
</style></head><body>
"""

HTML_FOOTER = "</body></html>\n"


def _html_colorize_cell(ch: str) -> str:
    color = HTML_SYMBOL_COLORS.get(ch)
    if color:
        return f'<span style="color:{color}">{ch}</span>'
    return ch


def _html_grid(matrix: np.ndarray) -> str:
    rows = []
    for r in range(matrix.shape[0]):
        cells = []
        for c in range(matrix.shape[1]):
            cells.append(_html_colorize_cell(matrix[r, c]))
        rows.append(" ".join(cells))
    return "<pre>" + "\n".join(rows) + "</pre>"


def _html_snapshots(snapshots: list[tuple[str, np.ndarray, str]], title: str, max_cols: int = 6) -> str:
    parts = [f"<h3>{title}</h3>"]
    for group_start in range(0, len(snapshots), max_cols):
        group = snapshots[group_start : group_start + max_cols]
        parts.append('<div class="grids-row">')
        for i, (move, matrix, state) in enumerate(group):
            step_idx = group_start + i
            if move == "START":
                label = "Initial"
            else:
                solved = ' <span class="solved">&#10003;</span>' if _is_solved(state) else ""
                label = f"Step {step_idx}: {move}{solved}"
            parts.append(f'<div class="grid-box"><div class="label">{label}</div>{_html_grid(matrix)}</div>')
        parts.append("</div>")
    return "\n".join(parts)


def _score_css_class(score: float) -> str:
    if score >= 0.8:
        return "score-good"
    elif score >= 0.3:
        return "score-mid"
    return "score-bad"


def html_visualize_sample(sample: dict, sample_idx: int, show_gt: bool = True, max_cols: int = 6) -> str:
    """Generate HTML for a single rollout sample."""
    input_text = sample["input"]
    pred = sample.get("pred", "")
    gts = sample.get("gts", "")
    score = sample.get("score", 0)
    feedback = sample.get("feedback", "")
    step = sample.get("step", "?")
    acc = sample.get("acc", "?")

    grid_list = extract_grid_from_input(input_text)
    if not grid_list:
        return f'<div class="sample-divider"></div><p>Sample {sample_idx}: could not extract grid.</p>'

    score_cls = _score_css_class(score)
    parts = [
        '<div class="sample-divider"></div>',
        f'<h2>Sample {sample_idx} &mdash; Step {step} &mdash; '
        f'Score: <span class="{score_cls}">{score:.2f}</span> &mdash; Acc: {acc}</h2>',
        '<div class="legend">'
        '<span style="color:#888">+</span>=wall &nbsp; '
        '<span style="color:#e6a817">@</span>=box &nbsp; '
        '<span style="color:#00bcd4">*</span>=player &nbsp; '
        '<span style="color:#4caf50">%</span>=player+goal &nbsp; '
        '<span style="color:#e74c3c">X</span>=goal &nbsp; '
        '<span style="color:#4caf50">$</span>=box+goal &nbsp; '
        '<span style="color:#555">-</span>=empty'
        '</div>',
    ]

    if pred:
        try:
            pred_snapshots = simulate_moves(grid_list, pred)
            parts.append(_html_snapshots(pred_snapshots, f"Predicted: <span style='color:#e6a817'>{pred}</span> ({len(pred)} moves)", max_cols))
        except Exception as e:
            parts.append(f"<p>Error simulating predicted moves: {e}</p>")

    if show_gt and gts:
        try:
            gt_snapshots = simulate_moves(grid_list, gts)
            parts.append(_html_snapshots(gt_snapshots, f"Ground Truth: <span style='color:#4caf50'>{gts}</span> ({len(gts)} moves)", max_cols))
        except Exception as e:
            parts.append(f"<p>Error simulating GT moves: {e}</p>")

    if feedback:
        import html as html_mod
        parts.append(f'<h3>Feedback</h3><div class="feedback">{html_mod.escape(feedback)}</div>')

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Visualize sokoban training rollouts")
    parser.add_argument("rollouts_dir", help="Path to rollouts directory")
    parser.add_argument("--step", type=int, default=None, help="Training step (jsonl file number). If not specified, shows all steps.")
    parser.add_argument("--sample", type=str, default="0", help="Comma-separated sample indices to visualize (default: 0)")
    parser.add_argument("--all", action="store_true", help="Visualize all samples")
    parser.add_argument("--no-gt", action="store_true", help="Don't show ground-truth moves")
    parser.add_argument("--max-cols", type=int, default=4, help="Max grids per row in side-by-side view (default: 4)")
    parser.add_argument("--score-range", type=str, default=None, help="Filter by score range, e.g. '0.0-0.5' or '1.0'")
    parser.add_argument("--save", action="store_true", help="Save as HTML file in the rollouts directory")
    parser.add_argument("--save-path", type=str, default=None, help="Custom path for the saved HTML file")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    if not rollouts_dir.exists():
        print(f"Error: {rollouts_dir} does not exist")
        sys.exit(1)

    # Find jsonl files
    if args.step is not None:
        files = [rollouts_dir / f"{args.step}.jsonl"]
        if not files[0].exists():
            print(f"Error: {files[0]} does not exist")
            sys.exit(1)
    else:
        files = sorted(rollouts_dir.glob("*.jsonl"), key=lambda f: int(f.stem))

    html_parts = [HTML_HEADER] if args.save or args.save_path else []

    for jsonl_file in files:
        print(f"\n{BOLD}{MAGENTA}{'▓' * 60}{RESET}")
        print(f"{BOLD}{MAGENTA}  File: {jsonl_file.name} (training step {jsonl_file.stem}){RESET}")
        print(f"{BOLD}{MAGENTA}{'▓' * 60}{RESET}")

        with open(jsonl_file) as f:
            samples = [json.loads(line) for line in f]

        # Filter by score range
        if args.score_range:
            if "-" in args.score_range:
                lo, hi = map(float, args.score_range.split("-"))
            else:
                lo = hi = float(args.score_range)
            samples_filtered = [(i, s) for i, s in enumerate(samples) if lo <= s.get("score", 0) <= hi]
            print(f"  Filtered {len(samples_filtered)}/{len(samples)} samples with score in [{lo}, {hi}]")
        else:
            samples_filtered = list(enumerate(samples))

        # Print summary stats
        scores = [s.get("score", 0) for _, s in samples_filtered]
        if scores:
            summary = (f"Samples: {len(samples_filtered)} | "
                       f"Avg score: {sum(scores)/len(scores):.3f} | "
                       f"Solved: {sum(1 for s in scores if s == 1.0)}/{len(scores)}")
            print(f"  {summary}")

        if html_parts:
            html_parts.append(f"<h1>Step {jsonl_file.stem} ({jsonl_file.name})</h1>")
            if scores:
                html_parts.append(f'<div class="summary">{summary}</div>')

        # Select samples to visualize
        if args.all:
            indices = list(range(len(samples_filtered)))
        else:
            indices = [int(x) for x in args.sample.split(",")]

        for idx in indices:
            if idx < len(samples_filtered):
                orig_idx, sample = samples_filtered[idx]
                visualize_sample(sample, orig_idx, show_gt=not args.no_gt)
                if html_parts:
                    html_parts.append(html_visualize_sample(
                        sample, orig_idx, show_gt=not args.no_gt,
                        max_cols=args.max_cols if args.max_cols else 6,
                    ))
            else:
                print(f"  {RED}Sample index {idx} out of range (max {len(samples_filtered) - 1}){RESET}")

    # Save HTML
    if html_parts:
        html_parts.append(HTML_FOOTER)
        if args.save_path:
            out_path = Path(args.save_path)
        else:
            # Build descriptive filename
            name_parts = ["vis"]
            if args.step is not None:
                name_parts.append(f"step{args.step}")
            if args.score_range:
                name_parts.append(f"score{args.score_range}")
            if args.all:
                name_parts.append("all")
            else:
                name_parts.append(f"sample{args.sample}")
            out_path = rollouts_dir / (("_".join(name_parts)) + ".html")
        out_path.write_text("\n".join(html_parts))
        print(f"\n{GREEN}Saved HTML to: {out_path}{RESET}")


if __name__ == "__main__":
    main()
