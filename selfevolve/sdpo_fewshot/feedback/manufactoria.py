"""
Manufactoria DSL reward function for iterative self-evolve training.

Adapted from ManufactoriaVerifier in:
  rl-grok-recipe/open-instruct/open_instruct/ground_truth_utils.py

Design:
  - The original verifier is a class (ManufactoriaVerifier) that calls an external
    Manufactoria DSL execution API to run test cases. This module flattens that into
    a standalone compute_score() matching the interface used by code.py.
  - DSL code is extracted from the last ```manufactoria ... ``` block in the model output.
  - Verification is delegated to an external API (default: http://localhost:8080/verify)
    which accepts {dsl, test_cases, max_execution_time} and returns per-test pass/fail
    results with rejection reasons.
  - Scoring supports both sparse (all-or-nothing) and dense (pass rate) modes,
    controlled by sparse_rewards and the split field in extra_info.
  - Feedback is rendered in a LeetCode-like style (Runtime Error / Wrong Answer)
    consistent with code.py's format_test_feedback.

API response format (expected):
  {
    "valid": bool,               # whether the DSL parsed successfully
    "message": str,              # error message if valid=False
    "all_passed": bool,          # whether all test cases passed
    "results": [                 # per-test-case results
      {
        "passed": bool,
        "input": str,            # input tape
        "output": str,           # actual output tape
        "expected_output": str,  # expected output tape
        "rejection_reason": str  # reason for failure (if any)
      },
      ...
    ]
  }
"""

import asyncio
import json
import re
from typing import Optional

import numpy as np
import requests

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
FORMAT_PENALTY = False

DEFAULT_API_URL = "http://localhost:8080/verify"
DEFAULT_MAX_EXECUTION_TIME = 5.0


def extract_manufactoria_code(model_output: str) -> Optional[str]:
    """Extract the last code block between ``` markers from the model output."""
    pattern = r"```(?:manufactoria)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


async def async_verify(dsl_code, test_cases, api_url, max_execution_time):
    """Call the Manufactoria verification API asynchronously."""
    payload = {
        "dsl": dsl_code,
        "test_cases": test_cases,
        "max_execution_time": max_execution_time,
    }

    def make_request():
        response = requests.post(
            api_url, json=payload, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    return await asyncio.to_thread(make_request)


def _parse_api_results(api_response):
    """
    Parse API response into a list of record dicts matching the code.py format:
      {test_idx, input, expected, actual, passed, debug, time}
    """
    records = []

    raw_results = api_response.get("results", [])
    if not isinstance(raw_results, list):
        return records

    for i, r in enumerate(raw_results):
        if not isinstance(r, dict):
            continue

        passed = r.get("passed", False)
        input_tape = r.get("input", "")
        expected = r.get("expected_output", r.get("expected", ""))
        actual_output = r.get("output", r.get("actual", ""))
        rejection_reason = r.get("rejection_reason", "")

        if passed:
            actual = actual_output
        elif rejection_reason:
            actual = f"{ERROR_PREFIX}{rejection_reason}"
        else:
            actual = actual_output

        records.append({
            "test_idx": i,
            "input": input_tape,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "debug": "",
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

    for r in failing:
        test_idx = r["test_idx"] + 1
        actual = r["actual"]
        expected = r["expected"]
        stdin = r["input"]

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_incorrect_format = actual == INCORRECT_FORMAT

        if is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            parts.append("Last Executed Input")
            parts.append(_truncate_str(stdin, max_input_chars))
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
    api_url=DEFAULT_API_URL,
    max_execution_time=DEFAULT_MAX_EXECUTION_TIME,
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

    # Extract DSL code
    dsl_code = extract_manufactoria_code(solution_str)
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

    # Call external API
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            api_response = asyncio.ensure_future(
                async_verify(dsl_code, test_cases, api_url, max_execution_time)
            )
            # If we're already in an async context, we need to await
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                api_response = pool.submit(
                    lambda: asyncio.run(async_verify(dsl_code, test_cases, api_url, max_execution_time))
                ).result()
        else:
            api_response = asyncio.run(
                async_verify(dsl_code, test_cases, api_url, max_execution_time)
            )
    except Exception as e:
        error_msg = f"API request failed: {e}"
        print(error_msg)
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": dsl_code,
            "incorrect_format": 0,
            "error_in_test_cases": 1,
            "timed_out": 0,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 0,
            "feedback": error_msg,
        }

    # Check for DSL validation errors from the API
    if "valid" in api_response and not api_response["valid"]:
        dsl_error = api_response.get("message", "DSL validation failed")
        records = [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": f"{ERROR_PREFIX}{dsl_error}",
            "passed": False,
            "debug": "",
            "time": 0.0,
        }]
    else:
        records = _parse_api_results(api_response)

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
            "feedback": "No test results returned from API.",
        }

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
