"""Visualize sudoku rollouts from training dumps.

Renders side-by-side grids:
  - Puzzle (given clues)
  - Predicted solution (errors highlighted in red)
  - Ground-truth solution
  - Per-cell diff summary

Usage:
    python scripts/visualize_sudoku_rollouts.py <rollouts_dir> [--step 1] [--sample 0] [--no-gt]
    python scripts/visualize_sudoku_rollouts.py <rollouts_dir> --step 1 --all --save
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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
BG_RED = "\033[41m"


def parse_grid(text: str) -> list[list[str]] | None:
    """Parse a 9x9 grid from text. Cells are digits or '_'."""
    rows = []
    for line in text.strip().split("\n"):
        cells = line.strip().split()
        # Only accept rows with 9 cells that are digits or '_'
        if len(cells) == 9 and all(c.isdigit() or c == "_" for c in cells):
            rows.append(cells)
    return rows if len(rows) == 9 else None


def extract_puzzle_from_input(input_text: str) -> list[list[str]] | None:
    """Extract the puzzle grid from the prompt."""
    # The puzzle appears after "Solve this Sudoku puzzle:" in the input
    lines = input_text.split("\n")
    grid_lines = []
    for line in lines:
        cells = line.strip().split()
        if len(cells) == 9 and all(c.isdigit() or c == "_" for c in cells):
            grid_lines.append(cells)
    return grid_lines if len(grid_lines) == 9 else None


def format_grid_ansi(grid: list[list[str]], puzzle: list[list[str]] | None = None,
                     gt: list[list[str]] | None = None, label: str = "") -> list[str]:
    """Format a 9x9 grid with ANSI colors.

    - Given clues (from puzzle) shown in bold white
    - Correct fills shown in green
    - Wrong fills shown in red (compared to gt)
    - Missing/blank shown as dot
    """
    lines = []
    if label:
        lines.append(f"  {BOLD}{label}{RESET}")

    lines.append(f"    {DIM}+-------+-------+-------+{RESET}")
    for r in range(9):
        row_str = f"    {DIM}|{RESET} "
        for c in range(9):
            val = grid[r][c] if grid else "_"
            is_given = puzzle and puzzle[r][c] != "_"

            if val == "_" or not val.isdigit():
                cell = f"{DIM}.{RESET}"
            elif is_given:
                cell = f"{BOLD}{val}{RESET}"
            elif gt and gt[r][c] != val:
                cell = f"{RED}{val}{RESET}"
            else:
                cell = f"{GREEN}{val}{RESET}"

            row_str += cell + " "
            if c in (2, 5):
                row_str += f"{DIM}|{RESET} "
        row_str += f"{DIM}|{RESET}"
        lines.append(row_str)
        if r in (2, 5):
            lines.append(f"    {DIM}+-------+-------+-------+{RESET}")
    lines.append(f"    {DIM}+-------+-------+-------+{RESET}")
    return lines


def compute_stats(puzzle: list[list[str]], pred: list[list[str]] | None,
                  gt: list[list[str]]) -> dict:
    """Compute accuracy stats."""
    blanks_total = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == "_")
    if not pred:
        return {"blanks_total": blanks_total, "correct": 0, "wrong": 0, "blank": blanks_total}
    correct = 0
    wrong = 0
    blank = 0
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != "_":
                continue  # skip givens
            if not pred[r][c].isdigit() or pred[r][c] == "_":
                blank += 1
            elif pred[r][c] == gt[r][c]:
                correct += 1
            else:
                wrong += 1
    return {"blanks_total": blanks_total, "correct": correct, "wrong": wrong, "blank": blank}


def print_side_by_side_grids(*grid_outputs: list[str], gap: str = "    "):
    """Print multiple grid outputs side by side."""
    max_lines = max(len(g) for g in grid_outputs)
    # Pad shorter grids
    padded = []
    for g in grid_outputs:
        padded.append(g + [""] * (max_lines - len(g)))

    for i in range(max_lines):
        parts = []
        for g in padded:
            parts.append(g[i] if i < len(g) else "")
        print(gap.join(parts))


def visualize_sample(sample: dict, sample_idx: int, show_gt: bool = True):
    """Visualize a single sudoku rollout sample."""
    input_text = sample["input"]
    pred_text = sample.get("pred", "")
    gts_text = sample.get("gts", "")
    score = sample.get("score", 0)
    feedback = sample.get("feedback", "")
    step = sample.get("step", "?")
    acc = sample.get("acc", "?")

    puzzle = extract_puzzle_from_input(input_text)
    pred_grid = parse_grid(pred_text) if pred_text else None
    gt_grid = parse_grid(gts_text) if gts_text else None

    if not puzzle:
        print(f"  {RED}Could not extract puzzle from input.{RESET}")
        return

    # Header
    score_color = GREEN if score >= 0.8 else (YELLOW if score >= 0.3 else RED)
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}  Sample {sample_idx} | Step {step} | "
          f"Score: {score_color}{score:.4f}{RESET}{BOLD} | Acc: {acc}{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")

    # Legend
    print(f"\n  {DIM}Legend:{RESET} "
          f"{BOLD}5{RESET}=given  "
          f"{GREEN}3{RESET}=correct fill  "
          f"{RED}7{RESET}=wrong fill  "
          f"{DIM}.{RESET}=blank")

    # Build grids
    puzzle_lines = format_grid_ansi(puzzle, label="Puzzle")
    if pred_grid:
        pred_lines = format_grid_ansi(pred_grid, puzzle=puzzle, gt=gt_grid, label="Prediction")
    else:
        pred_lines = [f"  {BOLD}Prediction{RESET}", f"  {RED}(could not parse){RESET}"]

    grids_to_show = [puzzle_lines, pred_lines]

    if show_gt and gt_grid:
        gt_lines = format_grid_ansi(gt_grid, puzzle=puzzle, label="Ground Truth")
        grids_to_show.append(gt_lines)

    print()
    print_side_by_side_grids(*grids_to_show)

    # Stats
    if gt_grid:
        stats = compute_stats(puzzle, pred_grid, gt_grid)
        print(f"\n  Blanks to fill: {stats['blanks_total']}  |  "
              f"{GREEN}Correct: {stats['correct']}{RESET}  |  "
              f"{RED}Wrong: {stats['wrong']}{RESET}  |  "
              f"{DIM}Blank: {stats['blank']}{RESET}")

    # Truncation / format warnings
    if sample.get("truncated"):
        print(f"  {YELLOW}WARNING: Response was truncated{RESET}")
    if sample.get("incorrect_format"):
        print(f"  {YELLOW}WARNING: Incorrect format{RESET}")

    # Feedback (truncated for readability)
    if feedback:
        print(f"\n  {DIM}Feedback:{RESET} {feedback[:300]}{'...' if len(feedback) > 300 else ''}")
    print()


# ── HTML output ──────────────────────────────────────────────────────────────

HTML_HEADER = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sudoku Rollout Visualization</title>
<style>
body { background: #1e1e1e; color: #ddd; font-family: 'Consolas', 'Monaco', monospace; padding: 20px; }
h1 { color: #bb86fc; border-bottom: 2px solid #bb86fc; padding-bottom: 8px; }
h2 { color: #03dac6; margin-top: 30px; }
h3 { color: #cf6679; margin: 8px 0 4px 0; font-size: 14px; }
.summary { background: #2d2d2d; padding: 10px 16px; border-radius: 6px; margin: 8px 0; display: inline-block; }
.legend { color: #aaa; margin: 8px 0; font-size: 13px; }
.grids-row { display: flex; flex-wrap: wrap; gap: 20px; margin: 12px 0; }
.grid-box { background: #2a2a2a; border-radius: 6px; padding: 12px 16px; }
.grid-box h3 { margin-top: 0; }
table.sudoku { border-collapse: collapse; }
table.sudoku td { width: 28px; height: 28px; text-align: center; font-size: 16px; border: 1px solid #444; }
table.sudoku td.given { color: #fff; font-weight: bold; }
table.sudoku td.correct { color: #4caf50; }
table.sudoku td.wrong { color: #e74c3c; background: #3a1a1a; }
table.sudoku td.blank { color: #555; }
table.sudoku td.border-right { border-right: 2px solid #888; }
table.sudoku td.border-bottom { border-bottom: 2px solid #888; }
.stats { margin: 8px 0; font-size: 13px; }
.stats .correct { color: #4caf50; }
.stats .wrong { color: #e74c3c; }
.stats .blank-stat { color: #888; }
.feedback { background: #2d2d2d; padding: 8px 12px; border-radius: 6px; margin: 8px 0; white-space: pre-wrap; color: #ffab91; font-size: 12px; max-height: 120px; overflow-y: auto; }
.sample-divider { border-top: 1px solid #444; margin: 24px 0; }
.score-good { color: #4caf50; }
.score-mid { color: #e6a817; }
.score-bad { color: #e74c3c; }
.warning { color: #e6a817; font-size: 13px; }
</style></head><body>
"""

HTML_FOOTER = "</body></html>\n"


def _html_sudoku_table(grid: list[list[str]], puzzle: list[list[str]] | None = None,
                       gt: list[list[str]] | None = None) -> str:
    rows = []
    for r in range(9):
        cells = []
        for c in range(9):
            val = grid[r][c] if grid else "_"
            is_given = puzzle and puzzle[r][c] != "_"

            classes = []
            if c in (2, 5):
                classes.append("border-right")
            if r in (2, 5):
                classes.append("border-bottom")

            if val == "_" or not val.isdigit():
                classes.append("blank")
                display = "."
            elif is_given:
                classes.append("given")
                display = val
            elif gt and gt[r][c] != val:
                classes.append("wrong")
                display = val
            else:
                classes.append("correct")
                display = val

            cls = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{cls}>{display}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<table class="sudoku">' + "".join(rows) + "</table>"


def _score_css_class(score: float) -> str:
    if score >= 0.8:
        return "score-good"
    elif score >= 0.3:
        return "score-mid"
    return "score-bad"


def html_visualize_sample(sample: dict, sample_idx: int, show_gt: bool = True) -> str:
    import html as html_mod

    input_text = sample["input"]
    pred_text = sample.get("pred", "")
    gts_text = sample.get("gts", "")
    score = sample.get("score", 0)
    feedback = sample.get("feedback", "")
    step = sample.get("step", "?")
    acc = sample.get("acc", "?")

    puzzle = extract_puzzle_from_input(input_text)
    pred_grid = parse_grid(pred_text) if pred_text else None
    gt_grid = parse_grid(gts_text) if gts_text else None

    if not puzzle:
        return f'<div class="sample-divider"></div><p>Sample {sample_idx}: could not extract puzzle.</p>'

    score_cls = _score_css_class(score)
    parts = [
        '<div class="sample-divider"></div>',
        f'<h2>Sample {sample_idx} &mdash; Step {step} &mdash; '
        f'Score: <span class="{score_cls}">{score:.4f}</span> &mdash; Acc: {acc}</h2>',
        '<div class="legend">'
        '<span style="color:#fff;font-weight:bold">5</span>=given &nbsp; '
        '<span style="color:#4caf50">3</span>=correct &nbsp; '
        '<span style="color:#e74c3c">7</span>=wrong &nbsp; '
        '<span style="color:#555">.</span>=blank'
        '</div>',
        '<div class="grids-row">',
        f'<div class="grid-box"><h3>Puzzle</h3>{_html_sudoku_table(puzzle)}</div>',
    ]

    if pred_grid:
        parts.append(f'<div class="grid-box"><h3>Prediction</h3>{_html_sudoku_table(pred_grid, puzzle=puzzle, gt=gt_grid)}</div>')
    else:
        parts.append('<div class="grid-box"><h3>Prediction</h3><p style="color:#e74c3c">(could not parse)</p></div>')

    if show_gt and gt_grid:
        parts.append(f'<div class="grid-box"><h3>Ground Truth</h3>{_html_sudoku_table(gt_grid, puzzle=puzzle)}</div>')

    parts.append("</div>")  # close grids-row

    if gt_grid:
        stats = compute_stats(puzzle, pred_grid, gt_grid)
        parts.append(
            f'<div class="stats">Blanks to fill: {stats["blanks_total"]} &nbsp;|&nbsp; '
            f'<span class="correct">Correct: {stats["correct"]}</span> &nbsp;|&nbsp; '
            f'<span class="wrong">Wrong: {stats["wrong"]}</span> &nbsp;|&nbsp; '
            f'<span class="blank-stat">Blank: {stats["blank"]}</span></div>'
        )

    warnings = []
    if sample.get("truncated"):
        warnings.append("Response was truncated")
    if sample.get("incorrect_format"):
        warnings.append("Incorrect format")
    if warnings:
        parts.append(f'<div class="warning">WARNING: {"; ".join(warnings)}</div>')

    if feedback:
        parts.append(f'<div class="feedback">{html_mod.escape(feedback[:500])}</div>')

    return "\n".join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize sudoku training rollouts")
    parser.add_argument("rollouts_dir", help="Path to rollouts directory")
    parser.add_argument("--step", type=int, default=None, help="Training step (jsonl file number)")
    parser.add_argument("--sample", type=str, default="0", help="Comma-separated sample indices (default: 0)")
    parser.add_argument("--all", action="store_true", help="Visualize all samples")
    parser.add_argument("--no-gt", action="store_true", help="Don't show ground-truth")
    parser.add_argument("--score-range", type=str, default=None, help="Filter by score range, e.g. '0.0-0.5'")
    parser.add_argument("--save", action="store_true", help="Save as HTML in the rollouts directory")
    parser.add_argument("--save-path", type=str, default=None, help="Custom HTML output path")
    args = parser.parse_args()

    rollouts_dir = Path(args.rollouts_dir)
    if not rollouts_dir.exists():
        print(f"Error: {rollouts_dir} does not exist")
        sys.exit(1)

    if args.step is not None:
        files = [rollouts_dir / f"{args.step}.jsonl"]
        if not files[0].exists():
            print(f"Error: {files[0]} does not exist")
            sys.exit(1)
    else:
        files = sorted(rollouts_dir.glob("*.jsonl"), key=lambda f: int(f.stem))

    html_parts = [HTML_HEADER] if args.save or args.save_path else []

    for jsonl_file in files:
        print(f"\n{BOLD}{MAGENTA}{'▓' * 70}{RESET}")
        print(f"{BOLD}{MAGENTA}  File: {jsonl_file.name} (training step {jsonl_file.stem}){RESET}")
        print(f"{BOLD}{MAGENTA}{'▓' * 70}{RESET}")

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

        scores = [s.get("score", 0) for _, s in samples_filtered]
        if scores:
            summary = (f"Samples: {len(samples_filtered)} | "
                       f"Avg score: {sum(scores)/len(scores):.3f} | "
                       f"Solved: {sum(1 for s in scores if s >= 0.99)}/{len(scores)}")
            print(f"  {summary}")

        if html_parts:
            html_parts.append(f"<h1>Step {jsonl_file.stem} ({jsonl_file.name})</h1>")
            if scores:
                html_parts.append(f'<div class="summary">{summary}</div>')

        if args.all:
            indices = list(range(len(samples_filtered)))
        else:
            indices = [int(x) for x in args.sample.split(",")]

        for idx in indices:
            if idx < len(samples_filtered):
                orig_idx, sample = samples_filtered[idx]
                visualize_sample(sample, orig_idx, show_gt=not args.no_gt)
                if html_parts:
                    html_parts.append(html_visualize_sample(sample, orig_idx, show_gt=not args.no_gt))
            else:
                print(f"  {RED}Sample index {idx} out of range (max {len(samples_filtered) - 1}){RESET}")

    if html_parts:
        html_parts.append(HTML_FOOTER)
        if args.save_path:
            out_path = Path(args.save_path)
        else:
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
