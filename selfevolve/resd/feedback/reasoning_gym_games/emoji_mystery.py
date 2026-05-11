from typing import Optional, Any


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score emoji mystery answer: char-level partial credit for same-length answers."""
    if answer is None:
        return 0.0, "Empty or invalid answer."

    expected = entry["answer"]
    try:
        if answer == expected:
            return 1.0, ""
        elif len(answer) == len(expected):
            wrong_positions = []
            for i, (a, b) in enumerate(zip(answer, expected)):
                if a != b:
                    wrong_positions.append(i + 1)
            correct = len(expected) - len(wrong_positions)
            score = correct / len(expected)
            feedback = f"{correct}/{len(expected)} characters correct."
            if wrong_positions:
                examples = wrong_positions[:5]
                feedback += f" Wrong at position(s): {examples}"
                if len(wrong_positions) > 5:
                    feedback += f" ...and {len(wrong_positions) - 5} more."
            feedback += f" The correct answer is: {expected}"
            return score, feedback
        else:
            return 0.01, f"Wrong length: got {len(answer)} characters, expected {len(expected)}. The correct answer is: {expected}"
    except Exception as e:
        return 0.01, f"Error scoring answer: {e}"
