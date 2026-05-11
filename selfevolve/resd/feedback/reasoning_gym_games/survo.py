from typing import Optional, Any
from collections import Counter


def _parse_grid(answer: str) -> list[list[int]]:
    grid = []
    for line in answer.strip().split("\n"):
        row = []
        for c in line.strip().split():
            try:
                row.append(int(c))
            except ValueError:
                continue
        if row:
            grid.append(row)
    return grid


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score survo answer with partial credit and detailed constraint feedback."""
    if not isinstance(answer, str):
        return 0.0, "Empty or invalid answer."

    board_size = entry["metadata"]["board_size"]
    grid = _parse_grid(answer)
    true_grid = entry["metadata"]["solution"]

    if len(grid) != board_size or any(len(row) != board_size for row in grid):
        return 0.0, f"Grid dimensions mismatch. Expected {board_size}x{board_size}, got {len(grid)} rows with lengths {[len(r) for r in grid]}."

    # Cell-by-cell comparison
    total_cells = board_size * board_size
    wrong_cells = []
    for i in range(board_size):
        for j in range(board_size):
            if grid[i][j] != true_grid[i][j]:
                wrong_cells.append((i + 1, j + 1, grid[i][j], true_grid[i][j]))

    if not wrong_cells:
        return 1.0, ""

    correct_cells = total_cells - len(wrong_cells)
    score = correct_cells / total_cells

    # Derive target row/col sums from the solution
    target_row_sums = [sum(true_grid[i]) for i in range(board_size)]
    target_col_sums = [sum(true_grid[i][j] for i in range(board_size)) for j in range(board_size)]

    feedback_parts = []
    feedback_parts.append(f"{correct_cells}/{total_cells} cells correct.")

    # Wrong cells
    cell_msgs = [f"row {r} col {c}: got {got}, expected {exp}" for r, c, got, exp in wrong_cells]
    feedback_parts.append("Wrong cells: " + "; ".join(cell_msgs))

    # Row sum violations
    row_errors = []
    for i in range(board_size):
        actual_sum = sum(grid[i])
        if actual_sum != target_row_sums[i]:
            diff = actual_sum - target_row_sums[i]
            direction = "too high" if diff > 0 else "too low"
            row_errors.append(f"row {i + 1}: sum = {actual_sum}, expected {target_row_sums[i]} ({direction} by {abs(diff)})")
    if row_errors:
        feedback_parts.append("Row sum violations: " + "; ".join(row_errors))

    # Column sum violations
    col_errors = []
    for j in range(board_size):
        actual_sum = sum(grid[i][j] for i in range(board_size))
        if actual_sum != target_col_sums[j]:
            diff = actual_sum - target_col_sums[j]
            direction = "too high" if diff > 0 else "too low"
            col_errors.append(f"col {j + 1}: sum = {actual_sum}, expected {target_col_sums[j]} ({direction} by {abs(diff)})")
    if col_errors:
        feedback_parts.append("Column sum violations: " + "; ".join(col_errors))

    # Number usage: each 1..N² should appear exactly once
    expected_nums = set(range(1, total_cells + 1))
    all_nums = [grid[i][j] for i in range(board_size) for j in range(board_size)]
    counts = Counter(all_nums)

    duplicates = {v: c for v, c in counts.items() if c > 1}
    missing = sorted(expected_nums - set(all_nums))
    out_of_range = sorted(v for v in all_nums if v not in expected_nums)

    if duplicates:
        msgs = [f"{v} appears {c} times" for v, c in sorted(duplicates.items())]
        feedback_parts.append("Duplicate values: " + "; ".join(msgs))
    if missing:
        feedback_parts.append(f"Missing values: {missing}")
    if out_of_range:
        feedback_parts.append(f"Out-of-range values: {sorted(set(out_of_range))} (expected 1 to {total_cells})")

    # Show submitted grid
    grid_str = "\n".join(" ".join(str(v) for v in row) for row in grid)
    feedback_parts.append(f"Your submitted grid:\n{grid_str}")

    if entry.get("answer") is not None:
        feedback_parts.append(f"The correct solution is: {entry['answer']}")

    feedback = "\n".join(feedback_parts)
    return score, feedback
