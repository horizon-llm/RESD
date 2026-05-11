import re
from typing import Optional, Any


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score tsumego answer: coordinate matching with format detection."""
    oracle_answer = entry["answer"].strip()
    if answer is not None and len(answer) > 0:
        answer = answer.strip().upper()
        if answer == oracle_answer:
            return 1.0, ""
        elif oracle_answer in answer:
            score = len(oracle_answer) / len(answer)
            return score, f"Answer contains the correct coordinate '{oracle_answer}' but has extra content."
        elif re.match(r"^([A-Z])(\d+)$", answer):
            return 0.05, f"Valid coordinate format but wrong position. Got '{answer}', expected '{oracle_answer}'."
        else:
            return 0.01, f"Invalid coordinate format. Got '{answer}', expected format like '{oracle_answer}'."
    return 0.0, "Empty or invalid answer."
