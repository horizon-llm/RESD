"""
Bouncingsim (Polygon Dynamics / ballsim) reward for iterative self-evolve training.

Design (aligned with manufactoria.py + code.py):
  - Python solution is taken from the last ```python ... ``` fenced block (strict).
  - Ground truth is a list of assert strings (JSON or compressed via decode_tests).
  - Scoring: dense pass rate by default; sparse (all-or-nothing) on split == \"test\".
  - Feedback: LeetCode-like (Runtime Error / Wrong Answer / Incorrect Format) via
    format_test_feedback, using per-test records compatible with code.py's schema.

Verifier logic is loaded from ballsim_pkg/verifier/ballsim_utils.py via importlib so we
avoid importing the old package __init__ (Box2D / pygame). The SDK lives under ballsim_pkg/
to avoid a name clash with this module (bouncingsim.py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# Standalone import path (e.g. importlib.util.spec_from_file_location)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_BALLSIM_UTILS = Path(__file__).resolve().parent / "ballsim_pkg" / "verifier" / "ballsim_utils.py"


def _load_ballsim_utils():
    spec = importlib.util.spec_from_file_location("sdpo_bouncingsim_ballsim_utils", _BALLSIM_UTILS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ballsim_utils from {_BALLSIM_UTILS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ballsim = _load_ballsim_utils()
decode_tests = _ballsim.decode_tests
get_successful_tests_fast = _ballsim.get_successful_tests_fast
get_successful_tests_with_details = _ballsim.get_successful_tests_with_details
should_execute = _ballsim.should_execute

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
FORMAT_PENALTY = False
DEFAULT_MAX_EXECUTION_TIME = 1.0


def extract_python_code(model_output: str) -> Optional[str]:
    """Extract the last ```python``` (or plain ```) block; None if no fenced block."""
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_tests(ground_truth: Any) -> List[str]:
    decoded = decode_tests(ground_truth)
    if not decoded:
        return []
    return [str(x) for x in decoded]


def run_tests(
    program: str,
    tests: List[str],
    max_execution_time: float,
):
    """
    Run program against assert tests. Returns records matching code.py / manufactoria:
      {test_idx, input, expected, actual, passed, debug, time}
    Here `input` holds the assert string; `expected` is empty (assertions are self-contained).

    Uses the decomposed test runner to distinguish real timeouts from exceptions
    and to capture actual output on wrong-answer cases.
    """
    if not tests:
        return []

    if not should_execute(program=program, tests=tests):
        return [
            {
                "test_idx": i,
                "input": t,
                "expected": "",
                "actual": f"{ERROR_PREFIX}Execution blocked by safety filter (restricted patterns in code).",
                "passed": False,
                "debug": "",
                "time": 0.0,
            }
            for i, t in enumerate(tests)
        ]

    try:
        compile(program, '<string>', 'exec')
    except SyntaxError as e:
        return [
            {
                "test_idx": i,
                "input": t,
                "expected": "",
                "actual": f"{ERROR_PREFIX}Syntax Error: {e}",
                "passed": False,
                "debug": "Compilation failed.",
                "time": 0.0,
            }
            for i, t in enumerate(tests)
        ]

    try:
        results, runtimes, errors, actuals = get_successful_tests_with_details(
            program=program,
            tests=tests,
            max_execution_time=max_execution_time,
        )
    except Exception as e:
        return [
            {
                "test_idx": 0,
                "input": tests[0] if tests else "",
                "expected": "",
                "actual": f"{ERROR_PREFIX}Evaluator error: {e}",
                "passed": False,
                "debug": traceback.format_exc(),
                "time": 0.0,
            }
        ]

    records = []
    for i, test in enumerate(tests):
        r = results[i] if i < len(results) else 0
        rt = runtimes[i] if i < len(runtimes) else -1.0
        err = errors[i] if i < len(errors) else ""
        act_output = actuals[i] if i < len(actuals) else ""
        passed = r == 1

        if passed:
            actual = "passed"
            debug = f"runtime_s={rt:.4f}" if rt is not None and rt >= 0 else ""
        elif rt == -1.0 and not err:
            # Process was killed — real timeout (runtime sentinel was never overwritten)
            actual = f"{ERROR_PREFIX}Time limit exceeded (killed after {max_execution_time}s). " \
                     f"Your code likely has an infinite loop in collision handling."
            debug = ""
        elif err.startswith("WrongAnswer:"):
            # Code ran but produced wrong positions
            actual = err  # e.g. "WrongAnswer: avg_distance=127.5, threshold=50.0\nBall 1: ..."
            debug = f"actual_output={act_output}" if act_output else ""
        else:
            # Exception raised (runtime == -2.0 or positive with error)
            actual = f"{ERROR_PREFIX}{err}" if err else f"{ERROR_PREFIX}Unknown error."
            debug = f"actual_output={act_output}" if act_output else ""

        records.append(
            {
                "test_idx": i,
                "input": test,
                "expected": "",
                "actual": actual,
                "passed": passed,
                "debug": debug,
                "time": float(rt) if rt is not None else 0.0,
            }
        )

    return records


def format_test_feedback(
    records,
    was_truncated=False,
    max_tests_to_show=2,
    sort_test_cases_by_length=True,
    max_length=2000,
    max_input_chars=400,
    max_expected_chars=250,
    max_actual_chars=400,
    max_debug_lines=10,
    max_debug_line_chars=300,
):
    """
    LeetCode-like feedback for assert-based tests. Only failing cases (unless only errors).
    Uses \"Assert\" instead of stdin \"Input\" for clarity.
    """
    if not records:
        return "No test execution information available."

    def _truncate_str(value, max_chars):
        if not isinstance(value, str):
            value = str(value)
        if max_chars is not None and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value

    failing = [r for r in records if not r["passed"]]

    # Prioritise first runtime/safety error (same idea as manufactoria)
    selected = None
    for rec in failing:
        actual = rec.get("actual", "")
        if isinstance(actual, str) and actual.startswith(ERROR_PREFIX):
            selected = rec
            break

    if selected is not None:
        failing = [selected]
    else:
        if sort_test_cases_by_length:
            failing = sorted(failing, key=lambda x: len(str(x["input"])) + len(str(x["actual"])))
        if max_tests_to_show is not None:
            failing = failing[: int(max_tests_to_show)]

    if not failing:
        return ""

    parts = []

    def _render_debug_block(dbg_text):
        dbg = (dbg_text or "").strip()
        if not dbg:
            return
        parts.append("")
        parts.append("Debug Output")
        dbg_lines = dbg.split("\n")
        limit = int(max_debug_lines) if max_debug_lines is not None else None
        for line in dbg_lines[:limit]:
            parts.append(_truncate_str(line, max_debug_line_chars))
        if max_debug_lines is not None and len(dbg_lines) > int(max_debug_lines):
            parts.append(f"... ({len(dbg_lines) - int(max_debug_lines)} more lines)")

    for r in failing:
        test_idx = r["test_idx"] + 1
        actual = r["actual"]
        expected = r["expected"]
        assert_text = r["input"]
        debug_text = r.get("debug", "")

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_incorrect_format = actual == INCORRECT_FORMAT
        is_wrong_answer = isinstance(actual, str) and actual.startswith("WrongAnswer:")

        if is_error:
            if "safety filter" in actual:
                parts.append("Runtime Error")
                parts.append(actual[len(ERROR_PREFIX) :])
            elif "Time limit" in actual:
                parts.append("Time Limit Exceeded")
                parts.append(actual[len(ERROR_PREFIX) :])
            else:
                parts.append("Runtime Error")
                parts.append(actual[len(ERROR_PREFIX) :])
            parts.append("")
            t_match = re.search(r'predict_position\(([^)]+)\)', assert_text)
            if t_match:
                parts.append("Input")
                parts.append(f"t = {t_match.group(1)}")
                parts.append("")
            parts.append("Assert")
            parts.append(_truncate_str(assert_text, max_input_chars))
            _render_debug_block(debug_text)
        elif is_incorrect_format:
            if was_truncated:
                parts.append("Truncated Attempt: Your previous response was too long and truncated.")
            else:
                parts.append("Incorrect Format: Put your code inside a ```python ... ``` block.")
        elif is_wrong_answer:
            # Enriched wrong-answer feedback with per-ball distance breakdown
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            # Parse the WrongAnswer detail lines
            wa_lines = actual.split("\n")
            # First line: "WrongAnswer: avg_distance=X, threshold=Y"
            parts.append(wa_lines[0].replace("WrongAnswer: ", ""))
            parts.append("")
            # Extract and show the time input explicitly
            t_match = re.search(r'predict_position\(([^)]+)\)', assert_text)
            if t_match:
                parts.append("Input")
                parts.append(f"t = {t_match.group(1)}")
                parts.append("")
            # Per-ball breakdown (remaining lines)
            if len(wa_lines) > 1:
                parts.append("Per-Ball Distances")
                for line in wa_lines[1:]:
                    if line.strip():
                        parts.append(_truncate_str(line, max_actual_chars))
                parts.append("")
            _render_debug_block(debug_text)
        else:
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            parts.append("Assert")
            parts.append(_truncate_str(assert_text, max_input_chars))
            parts.append("")
            parts.append("Result")
            parts.append(_truncate_str(actual, max_actual_chars))
            if expected:
                parts.append("")
                parts.append("Expected")
                parts.append(_truncate_str(expected, max_expected_chars))
            _render_debug_block(debug_text)

        parts.append("")

    result = "\n".join(parts).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info=None,
    sparse_rewards=False,
    max_test_cases=None,
    max_execution_time: float | None = None,
    **kwargs,
):
    """
    Self-evolve style score dict (keys aligned with manufactoria.py / VERL).
    """
    extra_info = extra_info or {}
    split = extra_info.get("split", "train")
    was_truncated = bool(extra_info.get("truncated", False))

    if split == "test":
        sparse_rewards = True

    max_t = max_execution_time
    if max_t is None:
        max_t = float(extra_info.get("bouncingsim_max_execution_time", DEFAULT_MAX_EXECUTION_TIME))

    tests = _parse_tests(ground_truth)
    if not tests:
        try:
            preview = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth)
        except Exception:
            preview = str(ground_truth)[:200]
        print("Error when reading bouncingsim tests: " + str(preview)[:1000])
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "combined_score": 0.0,
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": "Failed to parse ground truth (expected assert list via decode_tests).",
        }

    if max_test_cases and split != "test":
        tests = tests[: int(max_test_cases)]

    code = extract_python_code(solution_str)
    if code is None or not code.strip():
        return {
            "score": -0.5 if FORMAT_PENALTY and split == "train" and not was_truncated else 0.0,
            "acc": 0.0,
            "pred": "",
            "combined_score": 0.0,
            "incorrect_format": 1,
            "error_in_test_cases": 0,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": format_test_feedback(
                [
                    {
                        "test_idx": 0,
                        "input": "",
                        "expected": None,
                        "actual": INCORRECT_FORMAT,
                        "passed": False,
                        "debug": "",
                        "time": 0.0,
                    }
                ],
                was_truncated=was_truncated,
            ),
        }

    records = run_tests(code, tests, max_t)

    if not records:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": code,
            "combined_score": 0.0,
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 0,
            "feedback": "No test results produced.",
        }

    correct_answers = [1.0 if r["passed"] else 0.0 for r in records]
    predictions = str([r["actual"] for r in records])[-5000:]
    accuracy = float(np.mean(correct_answers)) if correct_answers else 0.0

    if sparse_rewards:
        reward = 1.0 if accuracy == 1.0 else 0.0
    else:
        reward = accuracy

    incorrect_format = False
    error_in_test_cases = any(
        (not r["passed"]) and isinstance(r.get("actual"), str) and ERROR_PREFIX in r["actual"]
        for r in records
    )
    timed_out = any(
        (not r["passed"]) 
        and isinstance(r.get("actual"), str) 
        and ("Time limit" in r["actual"] or r["actual"] == TIMEOUT)
        for r in records
    )

    if FORMAT_PENALTY and split == "train" and incorrect_format and not was_truncated:
        reward -= 0.5

    print(
        f"[bouncingsim compute_score] score={reward}, acc={accuracy}, "
        f"passed={sum(correct_answers)}/{len(correct_answers)}, "
        f"error_in_test_cases={error_in_test_cases}, timed_out={timed_out}"
    )

    return {
        "score": reward,
        "acc": accuracy,
        "pred": predictions,
        "combined_score": reward,
        "incorrect_format": 1 if incorrect_format else 0,
        "error_in_test_cases": 1 if error_in_test_cases else 0,
        "timed_out": 1 if timed_out else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if (incorrect_format and was_truncated) else 0,
        "feedback": format_test_feedback(records, was_truncated=was_truncated),
    }
