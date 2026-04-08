import re
from typing import Optional, Any

from sympy.parsing.sympy_parser import parse_expr


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score puzzle24 answer: validate expression evaluates to 24 using exactly 4 numbers."""
    if answer is None:
        return 0.01, "Empty or invalid answer."

    numbers = entry["metadata"]["numbers"]
    min_value = min(numbers)
    max_value = max(numbers)

    try:
        answer = answer.strip()
        user_answer = int(parse_expr(answer))
        used_numbers = [int(num) for num in re.findall(r"\b\d+\b", answer)]

        correct_expr = entry["answer"]

        if len(used_numbers) != 4:
            return 0.01, f"Must use exactly 4 numbers, but found {len(used_numbers)}: {used_numbers}. The correct expression is: {correct_expr}"

        out_of_range = [n for n in used_numbers if n > max_value or n < min_value]
        if out_of_range:
            return 0.01, f"Numbers {out_of_range} are outside the valid range [{min_value}, {max_value}]. Available numbers: {numbers}. The correct expression is: {correct_expr}"

        if user_answer != 24:
            return 0.01, f"Expression evaluates to {user_answer}, not 24. The correct expression is: {correct_expr}"

        return 1.0, ""
    except Exception as e:
        return 0.01, f"Failed to parse expression: {e}. The correct expression is: {entry['answer']}"
