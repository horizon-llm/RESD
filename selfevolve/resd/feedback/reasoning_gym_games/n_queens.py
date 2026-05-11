from typing import Optional, Any


def _board_to_string(board: list[list[str]]) -> str:
    """Convert a board (list of lists) to string representation."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in board)


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score N-Queens answer against set of valid solutions."""
    if isinstance(answer, str):
        valid_solutions = entry["metadata"]["valid_answers"]
        if answer in valid_solutions:
            return 1.0, ""
        try:
            answer = _board_to_string(eval(answer))
            if answer in valid_solutions:
                return 0.5, "Answer matched a valid solution but required format conversion (0.5x penalty)."
        except Exception:
            pass
    valid = entry["metadata"]["valid_answers"]
    example = valid[0] if valid else ""
    return 0.0, f"Answer does not match any of the {len(valid)} valid N-Queens solutions. One valid solution is:\n{example}"
