from typing import Optional, Any


def _parse_grid(answer: str) -> list[list[int]]:
    grid = []
    for line in answer.strip().split("\n"):
        row = [int(c) for c in line.strip() if c in "01"]
        if row:  # skip empty lines (e.g. LaTeX \begin{array}, \end{array})
            grid.append(row)
    return grid


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score kakurasu answer: validate row/col weighted sums."""
    if not isinstance(answer, str):
        return 0.0, "Empty or invalid answer."

    metadata = entry["metadata"]
    row_sums, col_sums = metadata["row_sums"], metadata["col_sums"]
    n_rows, n_cols = metadata["n_rows"], metadata["n_cols"]

    try:
        grid = _parse_grid(answer)

        if len(grid) != n_rows or any(len(row) != n_cols for row in grid):
            return 0.0, f"Grid dimensions mismatch. Expected {n_rows}x{n_cols}, got {len(grid)} rows with lengths {[len(r) for r in grid]}."

        invalid_cells = [(i + 1, j + 1, cell) for i, row in enumerate(grid) for j, cell in enumerate(row) if cell not in [1, 0]]
        if invalid_cells:
            examples = invalid_cells[:3]
            return 0.0, f"Invalid cell values (must be 0 or 1): " + "; ".join(f"row {r} col {c} has {v}" for r, c, v in examples)

        ans_row_sums = [sum((j + 1) for j, cell in enumerate(row) if cell == 1) for row in grid]
        row_errors = [(i + 1, ans_row_sums[i], row_sums[i]) for i in range(n_rows) if ans_row_sums[i] != row_sums[i]]

        ans_col_sums = [sum((i + 1) for i in range(n_rows) if grid[i][j] == 1) for j in range(n_cols)]
        col_errors = [(j + 1, ans_col_sums[j], col_sums[j]) for j in range(n_cols) if ans_col_sums[j] != col_sums[j]]

        total_constraints = n_rows + n_cols
        correct_rows = n_rows - len(row_errors)
        correct_cols = n_cols - len(col_errors)
        correct_constraints = correct_rows + correct_cols

        if not row_errors and not col_errors:
            return 1.0, ""

        score = correct_constraints / total_constraints if total_constraints > 0 else 0.0

        feedback_parts = []
        feedback_parts.append(
            f"{correct_constraints}/{total_constraints} constraints satisfied "
            f"({correct_rows}/{n_rows} rows, {correct_cols}/{n_cols} columns)."
        )

        # Row constraint violations with detail
        if row_errors:
            feedback_parts.append("Row constraint violations:")
            for r, got, exp in row_errors:
                diff = got - exp
                direction = "too high" if diff > 0 else "too low"
                active_cols = [j + 1 for j, cell in enumerate(grid[r - 1]) if cell == 1]
                active_str = f" (active columns: {active_cols})" if active_cols else " (no columns active)"
                feedback_parts.append(
                    f"  Row {r}: weighted sum = {got}, expected {exp} ({direction} by {abs(diff)}){active_str}"
                )

        # Column constraint violations with detail
        if col_errors:
            feedback_parts.append("Column constraint violations:")
            for c, got, exp in col_errors:
                diff = got - exp
                direction = "too high" if diff > 0 else "too low"
                active_rows = [i + 1 for i in range(n_rows) if grid[i][c - 1] == 1]
                active_str = f" (active rows: {active_rows})" if active_rows else " (no rows active)"
                feedback_parts.append(
                    f"  Col {c}: weighted sum = {got}, expected {exp} ({direction} by {abs(diff)}){active_str}"
                )

        # Show the submitted grid
        grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in grid)
        feedback_parts.append(f"Your submitted grid:\n{grid_str}")

        if entry.get("answer") is not None:
            feedback_parts.append(f"The correct solution is: {entry['answer']}")

        feedback = "\n".join(feedback_parts)
        return score, feedback
    except Exception as e:
        return 0.0, f"Failed to parse answer: {e}"
