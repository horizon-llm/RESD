from typing import Optional, Any
import re


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


def _parse_inequality_constraints(puzzle_text: str, board_size: int) -> list[tuple]:
    """Parse inequality constraints from the puzzle prompt.

    Looks for patterns like 'cell(r1,c1) < cell(r2,c2)' or grid-based
    inequality symbols (< > v ^) between cells.
    Returns list of ((r1,c1), (r2,c2)) meaning grid[r1][c1] < grid[r2][c2].
    """
    constraints = []

    # Try to find inequality symbols in a grid representation
    # Futoshiki puzzles typically have rows like: 1 < _ > 3   _
    # with vertical constraints on alternating lines: v   ^
    lines = puzzle_text.strip().split("\n")
    grid_rows = []
    between_rows = []
    for line in lines:
        # Check if line has numbers or blanks (puzzle row)
        nums = re.findall(r'[\d_]', line)
        ineqs = re.findall(r'[<>]', line)
        if nums and len(nums) >= board_size:
            grid_rows.append(line)
        elif re.search(r'[v^]', line) and not nums:
            between_rows.append(line)

    # Horizontal constraints from grid rows
    for r_idx, line in enumerate(grid_rows):
        # Find < and > between cell positions
        tokens = re.split(r'\s+', line.strip())
        col = 0
        for i, tok in enumerate(tokens):
            if tok in ('<', '>'):
                if col > 0:
                    c1 = col - 1
                    c2 = col
                    if tok == '<':
                        constraints.append(((r_idx, c1), (r_idx, c2)))
                    else:
                        constraints.append(((r_idx, c2), (r_idx, c1)))
            elif re.match(r'[\d_]', tok):
                col += 1

    # Vertical constraints from between-rows
    for br_idx, line in enumerate(between_rows):
        if br_idx >= len(grid_rows) - 1:
            break
        tokens = re.split(r'\s+', line.strip())
        col = 0
        for tok in tokens:
            if tok == 'v':
                constraints.append(((br_idx, col), (br_idx + 1, col)))
                col += 1
            elif tok == '^':
                constraints.append(((br_idx + 1, col), (br_idx, col)))
                col += 1
            elif tok == ' ' or tok == '':
                col += 1

    return constraints


def _check_constraints(grid: list[list[int]], board_size: int, inequality_constraints: list[tuple]) -> dict:
    """Check futoshiki row, column, and inequality constraints."""
    full_set = set(range(1, board_size + 1))

    row_violations = []
    col_violations = []
    inequality_violations = []
    missing_in_rows = {}
    missing_in_cols = {}

    # Row duplicates
    for r in range(board_size):
        seen = {}
        for c in range(board_size):
            v = grid[r][c]
            if v in seen:
                row_violations.append((r + 1, v, [seen[v] + 1, c + 1]))
            else:
                seen[v] = c
        missing = full_set - set(grid[r])
        if missing:
            missing_in_rows[r + 1] = sorted(missing)

    # Column duplicates
    for c in range(board_size):
        seen = {}
        for r in range(board_size):
            v = grid[r][c]
            if v in seen:
                col_violations.append((c + 1, v, [seen[v] + 1, r + 1]))
            else:
                seen[v] = r
        col_vals = {grid[r][c] for r in range(board_size)}
        missing = full_set - col_vals
        if missing:
            missing_in_cols[c + 1] = sorted(missing)

    # Inequality constraints
    for (r1, c1), (r2, c2) in inequality_constraints:
        if r1 < board_size and c1 < board_size and r2 < board_size and c2 < board_size:
            v1 = grid[r1][c1]
            v2 = grid[r2][c2]
            if v1 >= v2:
                inequality_violations.append(((r1 + 1, c1 + 1, v1), (r2 + 1, c2 + 1, v2)))

    return {
        "row_violations": row_violations,
        "col_violations": col_violations,
        "inequality_violations": inequality_violations,
        "missing_in_rows": missing_in_rows,
        "missing_in_cols": missing_in_cols,
    }


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score futoshiki answer with partial credit per correct cell."""
    if not isinstance(answer, str):
        return 0.0, "Empty or invalid answer."

    oracle_answer = entry["answer"]
    metadata = entry["metadata"]
    solution: list[list[int]] = metadata["solution"]
    board_size: int = len(solution[0])

    answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
    oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))

    if answer_stripped == oracle_answer_stripped:
        reward = 1.0
        feedback = ""
    else:
        row = 0
        num_matching = 0
        wrong_cells = []
        for ln in answer.split("\n"):
            if row >= len(solution):
                break
            numbers = [int(c) for c in ln if c in "123456789"]
            if len(numbers) != len(solution[0]):
                continue
            for col, (a, b) in enumerate(zip(solution[row], numbers)):
                if a == b:
                    num_matching += 1
                else:
                    wrong_cells.append((row + 1, col + 1, b, a))
            row += 1

        total_cells = board_size * board_size
        reward = num_matching / total_cells
        reward *= 0.9

        feedback_parts = [f"Non-standard format (0.9x penalty). {num_matching}/{total_cells} cells correct."]

        # Wrong cells
        if wrong_cells:
            cell_msgs = [f"row {r} col {c}: got {got}, expected {exp}" for r, c, got, exp in wrong_cells]
            feedback_parts.append("Wrong cells: " + "; ".join(cell_msgs))

        # Constraint violations
        grid = _parse_grid(answer, board_size)
        if grid:
            # Try to parse inequality constraints from the puzzle prompt
            puzzle_text = entry.get("prompt", "") or entry.get("question", "") or ""
            ineq_constraints = _parse_inequality_constraints(puzzle_text, board_size)

            violations = _check_constraints(grid, board_size, ineq_constraints)

            if violations["row_violations"]:
                msgs = [f"row {r}: value {v} appears at cols {cols}" for r, v, cols in violations["row_violations"]]
                feedback_parts.append("Row duplicates: " + "; ".join(msgs))

            if violations["col_violations"]:
                msgs = [f"col {c}: value {v} appears at rows {rows}" for c, v, rows in violations["col_violations"]]
                feedback_parts.append("Column duplicates: " + "; ".join(msgs))

            if violations["inequality_violations"]:
                msgs = [f"cell (row {r1},col {c1})={v1} should be < cell (row {r2},col {c2})={v2}"
                        for (r1, c1, v1), (r2, c2, v2) in violations["inequality_violations"]]
                feedback_parts.append("Inequality violations: " + "; ".join(msgs))

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
