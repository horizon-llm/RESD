import re
from typing import Optional, Any

import numpy as np
from sympy.parsing.sympy_parser import parse_expr

_num_re = re.compile(r"\b\d+\b")


def _extract_ints(expr_str: str) -> list[int]:
    """Grab the literal integers that appear in the source text."""
    return [int(m) for m in _num_re.findall(expr_str)]


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score countdown answer: validate expression evaluates to target using given numbers."""
    if answer is None or not answer.strip():
        return 0.01, "Empty or invalid answer."

    target = entry["metadata"]["target"]
    target_numbers = entry["metadata"]["numbers"]

    try:
        user_answer = float(parse_expr(answer))
        used_numbers = _extract_ints(answer)

        correct_expr = entry["answer"]

        if sorted(used_numbers) != sorted(target_numbers):
            return 0.05, f"Wrong numbers used. Expected {sorted(target_numbers)}, got {sorted(used_numbers)}. The correct expression is: {correct_expr}"

        if np.isclose(user_answer, target, atol=1e-6):
            return 1.0, ""

        return 0.05, f"Correct numbers used but expression evaluates to {user_answer}, not {target}. The correct expression is: {correct_expr}"
    except Exception as e:
        return 0.01, f"Failed to parse expression: {e}. The correct expression is: {entry['answer']}"
