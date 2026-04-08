"""CodeIO scoring: parse JSON answers and compare structurally with the oracle.

Provides rich feedback including missing/extra keys, value mismatches,
type differences, and partial credit based on correct fields.
"""
import json
import re
from typing import Any, Optional


def _try_parse_json(s: str) -> tuple[Any, Optional[str]]:
    """Attempt to parse JSON from a string, handling common formats.

    Tries raw parsing first, then strips markdown code fences, then
    extracts the first JSON object/array substring.

    Returns (parsed_value, None) on success or (None, error_message) on failure.
    """
    s = s.strip()

    # Try direct parse
    try:
        return json.loads(s), None
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown code fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", s, flags=re.MULTILINE)
    stripped = re.sub(r"\n?```\s*$", "", stripped, flags=re.MULTILINE).strip()
    if stripped != s:
        try:
            return json.loads(stripped), None
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to extract first JSON object or array
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        if open_ch in s and close_ch in s:
            start = s.index(open_ch)
            end = s.rindex(close_ch)
            if end > start:
                try:
                    return json.loads(s[start : end + 1]), None
                except (json.JSONDecodeError, ValueError):
                    pass

    return None, f"Could not parse JSON from answer. Got: {s[:200]}"


def _describe_type(obj: Any) -> str:
    """Return a human-friendly type description."""
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return f"str (length {len(obj)})"
    if isinstance(obj, dict):
        return f"dict with {len(obj)} key(s)"
    if isinstance(obj, list):
        return f"list with {len(obj)} item(s)"
    return type(obj).__name__


def _flatten_to_paths(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Recursively flatten a nested JSON structure to dot-path keys.

    Example: {"a": {"b": 1}, "c": [2, 3]} -> {"a.b": 1, "c[0]": 2, "c[1]": 3}
    """
    if depth > 20:
        return {prefix: obj}

    if isinstance(obj, dict):
        result = {}
        for key, value in sorted(obj.items()):
            path = f"{prefix}.{key}" if prefix else key
            result.update(_flatten_to_paths(value, path, depth + 1))
        if not obj:
            return {prefix: obj}
        return result
    elif isinstance(obj, list):
        result = {}
        for i, value in enumerate(obj):
            path = f"{prefix}[{i}]"
            result.update(_flatten_to_paths(value, path, depth + 1))
        if not obj:
            return {prefix: obj}
        return result
    else:
        return {prefix: obj}


def _compare_values(expected: Any, actual: Any, path: str) -> tuple[float, list[str]]:
    """Compare two leaf values and return (score, feedback_lines).

    Score: 1.0 exact match, 0.8 numeric closeness, 0.0 mismatch.
    """
    if expected == actual:
        return 1.0, []

    # Type mismatch
    if type(expected) != type(actual):
        # Allow int/float cross-comparison
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            pass  # fall through to numeric comparison
        else:
            return 0.0, [
                f"  '{path}': type mismatch — expected {_describe_type(expected)}, got {_describe_type(actual)}."
            ]

    # Numeric closeness
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        try:
            diff = abs(float(expected) - float(actual))
            divisor = max(abs(float(expected)), 1e-9)
            rel_error = diff / divisor
            if rel_error < 0.01 or diff < 1e-6:
                return 0.8, [
                    f"  '{path}': close — expected {expected}, got {actual} (off by {diff:.6g})."
                ]
        except (ValueError, OverflowError):
            pass
        return 0.0, [
            f"  '{path}': expected {expected}, got {actual}."
        ]

    # String comparison
    if isinstance(expected, str) and isinstance(actual, str):
        if expected.lower() == actual.lower():
            return 0.8, [
                f"  '{path}': case mismatch — expected '{expected}', got '{actual}'."
            ]
        return 0.0, [
            f"  '{path}': expected '{_truncate(expected, 80)}', got '{_truncate(actual, 80)}'."
        ]

    # Bool / None / other
    return 0.0, [
        f"  '{path}': expected {json.dumps(expected)}, got {json.dumps(actual)}."
    ]


def _truncate(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


# --- Scoring ---

def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score CodeIO answer: parse JSON and compare structurally with oracle."""
    if not isinstance(answer, str) or len(answer.strip()) == 0:
        return 0.0, "Empty or invalid answer."

    try:
        oracle_str = entry["answer"].strip()
        oracle_parsed, oracle_err = _try_parse_json(oracle_str)
        if oracle_err:
            # Fallback to exact match if oracle can't be parsed
            if answer.strip() == oracle_str:
                return 1.0, ""
            return 0.0, f"Incorrect answer.\nThe expected answer is: {oracle_str}"

        # Exact string match (fast path)
        if answer.strip() == oracle_str:
            return 1.0, ""

        answer_parsed, answer_err = _try_parse_json(answer)
        if answer_err:
            parts = [
                f"Failed to parse your answer as JSON: {answer_err}",
                f"Expected a value of type: {_describe_type(oracle_parsed)}.",
            ]
            parts.append(f"The expected answer is: {oracle_str}")
            return 0.0, "\n".join(parts)

        # Parsed equality check
        if answer_parsed == oracle_parsed:
            return 1.0, ""

        # Top-level type mismatch
        if type(oracle_parsed) != type(answer_parsed):
            # Allow int/float
            if not (isinstance(oracle_parsed, (int, float)) and isinstance(answer_parsed, (int, float))):
                parts = [
                    f"Top-level type mismatch: expected {_describe_type(oracle_parsed)}, got {_describe_type(answer_parsed)}.",
                ]
                parts.append(f"The expected answer is: {oracle_str}")
                return 0.0, "\n".join(parts)

        # Flatten and compare
        expected_paths = _flatten_to_paths(oracle_parsed)
        actual_paths = _flatten_to_paths(answer_parsed)

        expected_keys = set(expected_paths.keys())
        actual_keys = set(actual_paths.keys())

        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys
        common_keys = expected_keys & actual_keys

        # Score each common key
        total_score = 0.0
        value_feedback = []
        for key in sorted(common_keys):
            s, fb = _compare_values(expected_paths[key], actual_paths[key], key)
            total_score += s
            value_feedback.extend(fb)

        n_expected = len(expected_keys)
        score = total_score / n_expected if n_expected > 0 else 0.0
        score = min(score, 0.99)  # cap below 1.0 since not exact match

        correct_count = sum(
            1 for k in common_keys if expected_paths[k] == actual_paths[k]
        )

        # Build feedback
        parts = []
        parts.append(
            f"{correct_count}/{n_expected} fields correct."
        )

        if missing_keys:
            keys_str = ", ".join(sorted(missing_keys)[:10])
            suffix = f" (and {len(missing_keys) - 10} more)" if len(missing_keys) > 10 else ""
            parts.append(f"Missing keys: {keys_str}{suffix}")

        if extra_keys:
            keys_str = ", ".join(sorted(extra_keys)[:10])
            suffix = f" (and {len(extra_keys) - 10} more)" if len(extra_keys) > 10 else ""
            parts.append(f"Extra keys (not expected): {keys_str}{suffix}")

        if value_feedback:
            parts.append("Value mismatches:")
            parts.extend(value_feedback[:15])
            if len(value_feedback) > 15:
                parts.append(f"  ... and {len(value_feedback) - 15} more mismatches.")

        parts.append(f"The expected answer is: {oracle_str}")
        return score, "\n".join(parts)

    except Exception as e:
        return 0.0, f"Failed to score answer: {e}"
