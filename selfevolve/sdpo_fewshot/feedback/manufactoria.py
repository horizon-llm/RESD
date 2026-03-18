"""
Manufactoria DSL reward function for iterative self-evolve training.

Adapted from:
  - ManufactoriaVerifier in rl-grok-recipe/open-instruct/open_instruct/ground_truth_utils.py
  - manufactoria_api.py  in rl-grok-recipe/manufactoria/verifier/manufactoria_api.py
  - manufactoria_parser.py in rl-grok-recipe/manufactoria/verifier/manufactoria_parser.py

Design:
  - The original verifier (ManufactoriaVerifier) calls an external API to execute
    Manufactoria DSL code against test cases. Here we embed the parser and executor
    locally (copied into ./manufactoria/verifier/) so we can capture detailed
    tracebacks — execution path, rejection reasons, and parse errors with line
    numbers — and format them as feedback, analogous to code.py's _short_trace.
  - DSL code is extracted from the last ```manufactoria ... ``` block in the model output.
  - Scoring supports both sparse (all-or-nothing) and dense (pass rate) modes,
    controlled by sparse_rewards and the split field in extra_info.
  - Feedback is rendered in a LeetCode-like style (Runtime Error / Wrong Answer)
    consistent with code.py's format_test_feedback.
"""

import json
import os
import random
import re
import sys
from typing import Optional

import numpy as np

# Ensure the sibling `manufactoria/` package is importable when this file
# is loaded standalone (e.g. via importlib.util.spec_from_file_location).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manufactoria.verifier.manufactoria_parser import (
    ParseError,
    create_robot_factory,
)

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
FORMAT_PENALTY = False


def extract_manufactoria_code(model_output: str) -> Optional[str]:
    """Extract the last code block between ``` markers from the model output."""
    pattern = r"```(?:manufactoria)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _format_execution_trace(path, max_nodes=15):
    """Format the execution path (list of node IDs) into a traceback-like string."""
    if not path:
        return ""
    lines = []
    if len(path) > max_nodes:
        lines.append(f"  ... ({len(path) - max_nodes} earlier steps omitted)")
        shown = path[-max_nodes:]
        offset = len(path) - max_nodes
    else:
        shown = path
        offset = 0
    for j, node_id in enumerate(shown):
        step = offset + j + 1
        lines.append(f"  Step {step}: {node_id}")
    return "\n".join(lines)


def run_tests(dsl_code, test_cases):
    """
    Run DSL code against test cases locally using the embedded parser/executor.

    Returns a list of record dicts matching code.py's format:
      {test_idx, input, expected, actual, passed, debug, time}
    """
    # Parse DSL
    try:
        factory = create_robot_factory(dsl_code)
    except ParseError as e:
        return [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": f"{ERROR_PREFIX}DSL Parse Error: {e}",
            "passed": False,
            "debug": "",
            "time": 0.0,
        }]
    except Exception as e:
        return [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": f"{ERROR_PREFIX}Unexpected parse error: {e}",
            "passed": False,
            "debug": "",
            "time": 0.0,
        }]

    records = []
    for i, test_case in enumerate(test_cases):
        input_tape = test_case.get("input", "")
        expected_output = test_case.get("expected_output", "")
        expected_accepted = test_case.get("expected_accepted", True)
        check_output = test_case.get("check_output", True)

        try:
            result = factory.process_robot(input_tape)

            # Determine pass/fail (same logic as manufactoria_api.py)
            if check_output:
                has_regex = any(c in expected_output for c in ['.', '+', '*', '?', '|', '(', ')'])
                if has_regex:
                    try:
                        output_matches = bool(re.fullmatch(expected_output, result.final_tape))
                    except re.error:
                        output_matches = result.final_tape == expected_output
                else:
                    output_matches = result.final_tape == expected_output
                passed = (output_matches and result.finished) == expected_accepted
            else:
                passed = (result.finished == expected_accepted)

            actual = result.final_tape

            # Build debug trace for failing tests
            debug = ""
            if not passed:
                debug_parts = []
                if result.path:
                    debug_parts.append("Execution Trace:")
                    debug_parts.append(_format_execution_trace(result.path))
                if result.rejection_reason:
                    debug_parts.append(f"Rejection: {result.rejection_reason}")
                debug_parts.append(f"Finished: {result.finished}")
                debug_parts.append(f"Actual output: '{result.final_tape}'")
                if check_output:
                    debug_parts.append(f"Expected output: '{expected_output}'")
                debug_parts.append(f"Expected accepted: {expected_accepted}")
                debug = "\n".join(debug_parts)

            # For error cases, prefix the actual with ERROR_PREFIX
            if not passed and result.rejection_reason:
                actual = f"{ERROR_PREFIX}{result.rejection_reason}"

        except Exception as e:
            passed = False
            actual = f"{ERROR_PREFIX}{e}"
            debug = ""

        records.append({
            "test_idx": i,
            "input": input_tape,
            "expected": expected_output,
            "actual": actual,
            "passed": passed,
            "debug": debug,
            "time": 0.0,
        })

    return records


def format_test_feedback(
    records,
    was_truncated=False,
    max_tests_to_show=2,
    sort_test_cases_by_length=True,
    max_length=2000,
    max_input_chars=250,
    max_expected_chars=250,
    max_actual_chars=250,
    max_debug_lines=10,
    max_debug_line_chars=300,
):
    """
    Render test feedback in a LeetCode-like style, matching code.py format.
    Only shows failing cases.
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

    # Prioritise error/timeout cases
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
            failing = failing[:int(max_tests_to_show)]

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
        stdin = r["input"]
        debug_text = r.get("debug", "")

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_incorrect_format = actual == INCORRECT_FORMAT

        if is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            parts.append("Last Executed Input")
            parts.append(_truncate_str(stdin, max_input_chars))
            _render_debug_block(debug_text)
        elif is_incorrect_format:
            if was_truncated:
                parts.append("Truncated Attempt: Your previous response was too long and truncated.")
            else:
                parts.append("Incorrect Format: Put your code inside a ```manufactoria ... ``` block.")
        else:
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            parts.append("Input")
            parts.append(_truncate_str(stdin, max_input_chars))
            parts.append("")
            parts.append("Output")
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
    ground_truth: str,
    extra_info=None,
    sparse_rewards=False,
    max_test_cases=None,
    **kwargs,
):
    split = extra_info["split"] if extra_info else "train"
    was_truncated = extra_info.get("truncated", False) if extra_info else False

    if split == "test":
        sparse_rewards = True

    # Parse test cases
    try:
        test_cases = json.loads(ground_truth)
    except Exception:
        print("Error when reading tests: " + ground_truth[:1000])
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": "Failed to parse ground truth test cases.",
        }

    if not isinstance(test_cases, list):
        test_cases = [test_cases]

    if max_test_cases and split != "test":
        test_cases = test_cases[:max_test_cases]

    # print(f"[compute_score] split={split}, num_test_cases={len(test_cases)}, sparse_rewards={sparse_rewards}")

    # Extract DSL code
    dsl_code = extract_manufactoria_code(solution_str)
    # print(f"[compute_score] extracted DSL code: {dsl_code[:200] if dsl_code else None}")
    if dsl_code is None:
        return {
            "score": -0.5 if FORMAT_PENALTY and split == "train" and not was_truncated else 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 1,
            "error_in_test_cases": 0,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": format_test_feedback(
                [{"test_idx": 0, "input": None, "expected": None,
                  "actual": INCORRECT_FORMAT, "passed": False, "debug": "", "time": 0.0}],
                was_truncated=was_truncated,
            ),
        }

    # Show a random sample of test cases
    # sample_size = min(3, len(test_cases))
    # sample_cases = random.sample(test_cases, sample_size)
    # print(f"[compute_score] sample test cases ({sample_size}/{len(test_cases)}):")
    # for tc in sample_cases:
    #     print(f"  input={tc.get('input', '')!r}  expected_output={tc.get('expected_output', '')!r}  expected_accepted={tc.get('expected_accepted', True)}")

    # Run tests locally
    records = run_tests(dsl_code, test_cases)

    if not records:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": dsl_code,
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 0,
            "feedback": "No test results produced.",
        }

    # Show a random sample of results
    sample_records = random.sample(records, min(3, len(records)))
    print(f"[compute_score] sample results ({len(sample_records)}/{len(records)}):")
    for r in sample_records:
        print(f"  test_idx={r['test_idx']}  input={r['input']!r}  expected={r['expected']!r}  actual={r['actual']!r}  passed={r['passed']}")

    # Compute metrics (matching code.py)
    correct_answers = [1.0 if r["passed"] else 0.0 for r in records]
    predictions = str([r["actual"] for r in records])[-5000:]
    accuracy = np.mean(correct_answers)

    if sparse_rewards:
        reward = 1.0 if accuracy == 1.0 else 0.0
    else:
        reward = accuracy

    incorrect_format = False
    error_in_test_cases = any(
        (not r["passed"]) and isinstance(r["actual"], str) and ERROR_PREFIX in r["actual"]
        for r in records
    )
    timed_out = np.mean([
        1.0 if (not r["passed"]) and (r["actual"] == TIMEOUT) else 0.0
        for r in records
    ])

    if FORMAT_PENALTY and split == "train" and incorrect_format and not was_truncated:
        reward -= 0.5

    print(f"[compute_score] score={reward}, acc={accuracy}, passed={sum(correct_answers)}/{len(correct_answers)}, error_in_test_cases={error_in_test_cases}, timed_out={timed_out}")

    return {
        "score": reward,
        "acc": accuracy,
        "pred": predictions,
        "incorrect_format": 1 if incorrect_format else 0,
        "error_in_test_cases": 1 if error_in_test_cases else 0,
        "timed_out": 1 if timed_out else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": format_test_feedback(records, was_truncated=was_truncated),
    }
