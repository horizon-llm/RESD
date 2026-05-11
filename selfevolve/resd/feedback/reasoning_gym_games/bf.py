"""BF (Brainfuck) scoring: compare predicted program output with expected output.

Provides rich feedback including prefix match, character-level diffs,
longest common subsequence, numeric closeness, and BFit source hints.
"""
from typing import Any, Optional


def _common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix."""
    length = min(len(a), len(b))
    for i in range(length):
        if a[i] != b[i]:
            return i
    return length


def _longest_common_subsequence_length(a: str, b: str, cap: int = 500) -> int:
    """Compute LCS length using standard DP, capped at `cap` characters."""
    a = a[:cap]
    b = b[:cap]
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _char_diff_summary(expected: str, actual: str, max_diffs: int = 3) -> list[str]:
    """Show the first `max_diffs` character differences with context."""
    diffs = []
    max_len = max(len(expected), len(actual))
    for i in range(min(max_len, 200)):
        exp_ch = expected[i] if i < len(expected) else "<end>"
        act_ch = actual[i] if i < len(actual) else "<end>"
        if exp_ch != act_ch:
            exp_repr = repr(exp_ch) if exp_ch != "<end>" else "<end of string>"
            act_repr = repr(act_ch) if act_ch != "<end>" else "<end of string>"
            diffs.append(f"  Position {i}: expected {exp_repr}, got {act_repr}")
            if len(diffs) >= max_diffs:
                break
    return diffs


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    try:
        return float(s)
    except (ValueError, OverflowError):
        return None


# --- Scoring ---

def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score BF answer: compare predicted output with expected program output."""
    if not isinstance(answer, str) or len(answer.strip()) == 0:
        return 0.0, "Empty or invalid answer."

    try:
        expected = entry["answer"]
        answer_stripped = answer.strip()
        expected_stripped = expected.strip()

        # Exact match
        if answer_stripped == expected_stripped:
            return 1.0, ""

        exp_len = len(expected_stripped)
        ans_len = len(answer_stripped)

        if exp_len == 0:
            return 0.0, f"Expected empty output, but got {ans_len} character(s): {repr(answer_stripped[:50])}"

        # Compute metrics
        prefix_len = _common_prefix_length(answer_stripped, expected_stripped)
        lcs_len = _longest_common_subsequence_length(answer_stripped, expected_stripped)
        prefix_ratio = prefix_len / exp_len
        lcs_ratio = lcs_len / exp_len

        # Base score from string similarity
        score = max(prefix_ratio * 0.7, lcs_ratio * 0.5)

        # Numeric closeness bonus
        numeric_feedback = None
        exp_num = _try_parse_number(expected_stripped)
        ans_num = _try_parse_number(answer_stripped)
        if exp_num is not None and ans_num is not None:
            diff = abs(exp_num - ans_num)
            divisor = max(abs(exp_num), 1e-9)
            rel_error = diff / divisor
            if rel_error < 0.01:
                numeric_score = 0.9
            elif rel_error < 0.1:
                numeric_score = 0.5
            elif rel_error < 0.5:
                numeric_score = 0.2
            else:
                numeric_score = 0.0
            score = max(score, numeric_score)
            numeric_feedback = f"Numeric comparison: expected {exp_num:g}, got {ans_num:g} (off by {diff:g}, {rel_error:.1%} relative error)."

        score = min(score, 0.99)  # cap below 1.0

        # Build feedback
        parts = []
        parts.append(
            f"Output mismatch. Your output has {ans_len} character(s), expected {exp_len} character(s)."
        )

        # Prefix match
        if prefix_len > 0:
            parts.append(
                f"First {prefix_len} character(s) correct (prefix match: {prefix_len}/{exp_len} = {prefix_ratio:.0%})."
            )

        # First difference
        if prefix_len < max(exp_len, ans_len):
            exp_ch = repr(expected_stripped[prefix_len]) if prefix_len < exp_len else "<end of string>"
            act_ch = repr(answer_stripped[prefix_len]) if prefix_len < ans_len else "<end of string>"
            parts.append(
                f"First difference at position {prefix_len}: expected {exp_ch}, got {act_ch}."
            )

        # Character diffs
        diffs = _char_diff_summary(expected_stripped, answer_stripped)
        if diffs:
            parts.append("Character differences:")
            parts.extend(diffs)

        # LCS
        parts.append(
            f"Longest common subsequence: {lcs_len}/{exp_len} character(s) ({lcs_ratio:.0%})."
        )

        # Numeric feedback
        if numeric_feedback:
            parts.append(numeric_feedback)

        # BFit source hint
        metadata = entry.get("metadata", {})
        bfit_code = metadata.get("bfit_code")
        if bfit_code:
            parts.append(f"Hint — the high-level program (BFit) that generated this BF code:\n{bfit_code.strip()}")

        parts.append(f"The expected output is: {repr(expected_stripped)}")

        return score, "\n".join(parts)

    except Exception as e:
        return 0.0, f"Failed to score answer: {e}"
