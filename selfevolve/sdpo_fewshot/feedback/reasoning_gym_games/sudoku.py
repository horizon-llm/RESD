from typing import Optional, Any
import math


def _parse_grid(answer: str, board_size: int) -> list[list[int]] | None:
    """Parse answer into a grid of ints, returning None if unparseable."""
    grid = []
    for ln in answer.split("\n"):
        numbers = [int(c) for c in ln if c in "123456789"]
        if len(numbers) == board_size:
            grid.append(numbers)
        if len(grid) == board_size:
            break
    return grid if len(grid) == board_size else None


def _check_constraints(grid: list[list[int]], board_size: int) -> dict:
    """Check sudoku row, column, and box constraints. Return detailed violations."""
    box_size = int(math.isqrt(board_size))
    row_violations = []
    col_violations = []
    box_violations = []

    # Row duplicates
    for r in range(board_size):
        seen = {}
        for c in range(board_size):
            v = grid[r][c]
            if v in seen:
                row_violations.append((r + 1, v, [seen[v] + 1, c + 1]))
            else:
                seen[v] = c

    # Column duplicates
    for c in range(board_size):
        seen = {}
        for r in range(board_size):
            v = grid[r][c]
            if v in seen:
                col_violations.append((c + 1, v, [seen[v] + 1, r + 1]))
            else:
                seen[v] = r

    # Box duplicates
    if box_size * box_size == board_size:
        for br in range(0, board_size, box_size):
            for bc in range(0, board_size, box_size):
                seen = {}
                for r in range(br, br + box_size):
                    for c in range(bc, bc + box_size):
                        v = grid[r][c]
                        if v in seen:
                            box_violations.append(
                                (br // box_size + 1, bc // box_size + 1, v,
                                 seen[v], (r + 1, c + 1))
                            )
                        else:
                            seen[v] = (r + 1, c + 1)

    # Missing values per row/col
    full_set = set(range(1, board_size + 1))
    missing_in_rows = {}
    for r in range(board_size):
        missing = full_set - set(grid[r])
        if missing:
            missing_in_rows[r + 1] = sorted(missing)

    missing_in_cols = {}
    for c in range(board_size):
        col_vals = {grid[r][c] for r in range(board_size)}
        missing = full_set - col_vals
        if missing:
            missing_in_cols[c + 1] = sorted(missing)

    return {
        "row_violations": row_violations,
        "col_violations": col_violations,
        "box_violations": box_violations,
        "missing_in_rows": missing_in_rows,
        "missing_in_cols": missing_in_cols,
    }


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score sudoku answer with partial credit per correct cell."""
    if not isinstance(answer, str) or len(answer) == 0:
        return 0.0, "Empty or invalid answer."

    oracle_answer = entry["answer"]
    metadata = entry["metadata"]
    solution: list[list[int]] = metadata["solution"]
    board_size: int = len(solution[0])

    # 1. match answer without trailing whitespaces
    answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
    oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))

    if answer_stripped == oracle_answer_stripped:
        reward = 1.0
        feedback = ""
    else:
        # 2. accept answers with correct numeric sequence (ignoring non-numeric characters)
        row = 0
        num_matching = 0
        wrong_cells = []
        for ln in answer.split("\n"):
            if row >= len(solution):
                break
            numbers = [int(c) for c in ln if c in "123456789"]
            if len(numbers) != board_size:
                continue
            for col, (a, b) in enumerate(zip(solution[row], numbers)):
                if a == b:
                    num_matching += 1
                else:
                    wrong_cells.append((row + 1, col + 1, b, a))
            row += 1

        total_cells = board_size * board_size
        reward = num_matching / total_cells
        reward *= 0.9  # penalty for not using standard format

        feedback_parts = [f"Non-standard format (0.9x penalty). {num_matching}/{total_cells} cells correct."]

        # Wrong cells
        if wrong_cells:
            cell_msgs = [f"row {r} col {c}: got {got}, expected {exp}" for r, c, got, exp in wrong_cells]
            feedback_parts.append("Wrong cells: " + "; ".join(cell_msgs))

        # Constraint violations
        grid = _parse_grid(answer, board_size)
        if grid:
            violations = _check_constraints(grid, board_size)

            if violations["row_violations"]:
                msgs = [f"row {r}: value {v} appears at cols {cols}" for r, v, cols in violations["row_violations"]]
                feedback_parts.append("Row duplicates: " + "; ".join(msgs))

            if violations["col_violations"]:
                msgs = [f"col {c}: value {v} appears at rows {rows}" for c, v, rows in violations["col_violations"]]
                feedback_parts.append("Column duplicates: " + "; ".join(msgs))

            if violations["box_violations"]:
                msgs = [f"box ({br},{bc}): value {v} at {pos1} and {pos2}"
                        for br, bc, v, pos1, pos2 in violations["box_violations"]]
                feedback_parts.append("Box duplicates: " + "; ".join(msgs))

            if violations["missing_in_rows"]:
                msgs = [f"row {r}: missing {vals}" for r, vals in violations["missing_in_rows"].items()]
                feedback_parts.append("Missing values in rows: " + "; ".join(msgs))

            if violations["missing_in_cols"]:
                msgs = [f"col {c}: missing {vals}" for c, vals in violations["missing_in_cols"].items()]
                feedback_parts.append("Missing values in cols: " + "; ".join(msgs))

            # Show submitted grid
            grid_str = "\n".join(" ".join(str(v) for v in row) for row in grid)
            feedback_parts.append(f"Your submitted grid:\n{grid_str}")

        feedback = "\n".join(feedback_parts)

    if len(answer) > len(oracle_answer):
        length_penalty = len(oracle_answer) / len(answer)
        reward *= length_penalty
        feedback = (feedback + f"\nResponse too long ({len(answer)} chars vs expected {len(oracle_answer)}), score reduced.").strip()
    return reward, feedback
