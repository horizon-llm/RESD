"""
Reward function for competition coding problems.
Ported from rl-grok-recipe/open-instruct/open_instruct/code/code_utils.py

Executes assertion-based tests in sandboxed subprocesses with:
- Per-call timing wrapper around solve()
- SINGLE_IN_GENERATORS injected for runtime/timeout tests
- Batched parallel execution with timeout
"""

import faulthandler
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

# Load generators via importlib to work regardless of how this module is loaded
# (both as a package member and via load_extern_object / spec_from_file_location)
_GENERATORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competitioncode_generators.py")
_spec = importlib.util.spec_from_file_location("competitioncode_generators", _GENERATORS_PATH)
_generators_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generators_mod)
sys.modules["competitioncode_generators"] = _generators_mod

SINGLE_IN_GENERATORS_CDQ_DC = _generators_mod.SINGLE_IN_GENERATORS_CDQ_DC
SINGLE_IN_GENERATORS_MEET_IN_THE_MIDDLE = _generators_mod.SINGLE_IN_GENERATORS_MEET_IN_THE_MIDDLE
SINGLE_IN_GENERATORS_MO_ALGORITHM = _generators_mod.SINGLE_IN_GENERATORS_MO_ALGORITHM
SINGLE_IN_GENERATORS_SEGMENT_TREE_DC = _generators_mod.SINGLE_IN_GENERATORS_SEGMENT_TREE_DC
SINGLE_IN_GENERATORS_SQRT_DC = _generators_mod.SINGLE_IN_GENERATORS_SQRT_DC

# Use fork context to avoid pickling issues when loaded via spec_from_file_location
_mp_ctx = multiprocessing.get_context("fork")

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "
DEFAULT_TIMEOUT = 10.0
MAX_CONCURRENT_TESTS = 4
MAX_ADDITIONAL_MEMORY_BYTES = 1024 * 1024 * 1024  # 1GB

FILENAME = "Solution.py"

# Timing wrapper template injected after the solution code
TIMING_WRAPPER_TEMPLATE = """
import time as __omega_time
__omega_solve_time_calls = []
__omega_SOLVE_TIME_LIMIT = {time_limit}
def __omega_wrap_solve(__omega_orig):
    def __omega_wrapped(*__omega_args, **__omega_kwargs):
        __omega_start = __omega_time.perf_counter()
        try:
            return __omega_orig(*__omega_args, **__omega_kwargs)
        finally:
            __omega_elapsed = __omega_time.perf_counter() - __omega_start
            __omega_solve_time_calls.append(__omega_elapsed)
            if __omega_elapsed > __omega_SOLVE_TIME_LIMIT:
                raise AssertionError("solve call exceeded time limit: " + str(round(__omega_elapsed, 3)) + "s > " + str(__omega_SOLVE_TIME_LIMIT) + "s")
    return __omega_wrapped
try:
    solve
    solve = __omega_wrap_solve(solve)
except NameError:
    pass
"""


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Disable destructive functions to prevent interference with tests."""
    if maximum_memory_bytes is not None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        except Exception:
            pass

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _short_trace(e, limit=3):
    """Return a compact traceback focused on the user's solution."""
    frames = traceback.extract_tb(e.__traceback__)
    solution_frames = [
        f for f in frames
        if isinstance(getattr(f, "filename", None), str)
        and f.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{type(e).__name__}: {e}"]
    for f in tail:
        if f.line:
            lines.append(f"  {f.line}")
        lines.append(f"Line {f.lineno} in {f.name} ({FILENAME})")
    return "\n".join(lines)


def extract_code(response: str) -> Optional[str]:
    """Extract the last Python code block from model output."""
    blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    return max((code for _, code in blocks), key=len)


def run_tests_for_one_example(completion: str, test: str, send_conn, test_idx: int, time_limit: float):
    """Run a single test assertion in an isolated subprocess."""
    reliability_guard(MAX_ADDITIONAL_MEMORY_BYTES)

    record = {
        "test_idx": test_idx,
        "input": test[:500],
        "expected": "",
        "actual": "",
        "passed": False,
        "debug": "",
        "time": float("inf"),
    }

    try:
        execution_context: Dict[str, Any] = {"__builtins__": __builtins__}
        execution_context["SINGLE_IN_GENERATORS_MO_ALGORITHM"] = SINGLE_IN_GENERATORS_MO_ALGORITHM
        execution_context["SINGLE_IN_GENERATORS_CDQ_DC"] = SINGLE_IN_GENERATORS_CDQ_DC
        execution_context["SINGLE_IN_GENERATORS_MEET_IN_THE_MIDDLE"] = SINGLE_IN_GENERATORS_MEET_IN_THE_MIDDLE
        execution_context["SINGLE_IN_GENERATORS_SEGMENT_TREE_DC"] = SINGLE_IN_GENERATORS_SEGMENT_TREE_DC
        execution_context["SINGLE_IN_GENERATORS_SQRT_DC"] = SINGLE_IN_GENERATORS_SQRT_DC

        # Compile and exec solution (defines solve())
        instrumented_code = completion + TIMING_WRAPPER_TEMPLATE.format(time_limit=time_limit)
        code_obj = compile(instrumented_code, FILENAME, "exec")
        exec(code_obj, execution_context)

        # Run the assertion test
        start_time = time.time()
        exec(test, execution_context)
        elapsed = time.time() - start_time

        # Use per-solve timing if available
        omega_calls = execution_context.get("__omega_solve_time_calls", None)
        if isinstance(omega_calls, list) and len(omega_calls) > 0:
            elapsed = max(omega_calls)

        record["passed"] = True
        record["actual"] = "Pass"
        record["time"] = elapsed

    except AssertionError as e:
        err_str = str(e)
        if "solve call exceeded time limit" in err_str or "solve exceeded time limit" in err_str:
            record["actual"] = TIMEOUT
        elif err_str == "" or "assert" in test.lower():
            record["actual"] = "Wrong Answer"
        else:
            record["actual"] = f"{ERROR_PREFIX}{_short_trace(e)}"
        record["time"] = time.time() - (start_time if 'start_time' in dir() else time.time())

    except Exception as e:
        record["actual"] = f"{ERROR_PREFIX}{_short_trace(e)}"

    try:
        send_conn.send(record)
    except Exception:
        pass
    finally:
        send_conn.close()


def run_tests(tests: List[str], solution: str, max_test_cases: Optional[int],
              max_execution_time: float, max_workers: Optional[int] = None):
    """Run all tests against the solution in parallel batched subprocesses."""
    completion = extract_code(solution)
    if completion is None:
        return [{
            "test_idx": 0,
            "input": None,
            "expected": None,
            "actual": INCORRECT_FORMAT,
            "passed": False,
            "debug": "",
            "time": float("inf"),
        }]

    num_test_cases = min(max_test_cases, len(tests)) if max_test_cases else len(tests)
    if max_workers is None:
        max_workers = MAX_CONCURRENT_TESTS

    records = []

    for batch_start in range(0, num_test_cases, max_workers):
        batch_indices = range(batch_start, min(batch_start + max_workers, num_test_cases))
        process_data = []

        for test_idx in batch_indices:
            parent_conn, child_conn = _mp_ctx.Pipe(duplex=False)
            p = _mp_ctx.Process(
                target=run_tests_for_one_example,
                args=(completion, tests[test_idx], child_conn, test_idx, max_execution_time)
            )
            p.start()
            child_conn.close()
            process_data.append({"process": p, "parent_conn": parent_conn})

        batch_start_time = time.time()
        for test_idx, data in zip(batch_indices, process_data):
            p = data["process"]
            parent_conn = data["parent_conn"]

            timeout_this_test = max(0, max_execution_time + 2 - (time.time() - batch_start_time))
            if parent_conn.poll(timeout_this_test):
                try:
                    result = parent_conn.recv()
                except Exception:
                    result = {
                        "test_idx": test_idx,
                        "input": tests[test_idx][:500],
                        "expected": "",
                        "actual": f"{ERROR_PREFIX}Process communication error",
                        "passed": False,
                        "debug": "",
                        "time": float("inf"),
                    }
            else:
                result = {
                    "test_idx": test_idx,
                    "input": tests[test_idx][:500],
                    "expected": "",
                    "actual": TIMEOUT,
                    "passed": False,
                    "debug": "",
                    "time": float("inf"),
                }

            records.append(result)

            parent_conn.close()
            p.join(timeout=0)
            if p.is_alive():
                p.kill()
                p.join()

    assert len(records) == num_test_cases
    return records


def format_test_feedback(
    records: List[Dict],
    was_truncated: bool = False,
    max_tests_to_show: int = 2,
    max_length: int = 2000,
    max_input_chars: int = 250,
    max_input_lines: int = 8,
):
    """Render test feedback in LeetCode-like style."""
    if not records:
        return "No test execution information available."

    failing = [r for r in records if not r["passed"]]
    if not failing:
        return ""

    def _truncate(value, max_chars):
        if not isinstance(value, str):
            value = str(value)
        return value[:max_chars] + "..." if len(value) > max_chars else value

    # Prioritize: errors > timeouts > wrong answers
    selected = None
    for rec in failing:
        actual = rec.get("actual", "")
        if isinstance(actual, str) and actual.startswith(ERROR_PREFIX):
            selected = rec
            break
    if selected is None:
        for rec in failing:
            if rec.get("actual") == TIMEOUT:
                selected = rec
                break
    if selected is None:
        for rec in failing:
            if rec.get("actual") == INCORRECT_FORMAT:
                selected = rec
                break

    if selected is not None:
        failing = [selected]
    else:
        failing = sorted(failing, key=lambda x: len(str(x.get("input", ""))))
        failing = failing[:max_tests_to_show]

    parts = []

    for r in failing:
        actual = r["actual"]
        stdin = r.get("input", "")

        is_error = isinstance(actual, str) and actual.startswith(ERROR_PREFIX)
        is_timeout = actual == TIMEOUT
        is_incorrect_format = actual == INCORRECT_FORMAT

        if is_error:
            parts.append("Runtime Error")
            parts.append(actual[len(ERROR_PREFIX):])
            parts.append("")
            if stdin:
                parts.append("Last Executed Input")
                text = str(stdin)
                lines = text.splitlines()
                for line in lines[:max_input_lines]:
                    parts.append(_truncate(line, max_input_chars))
                if len(lines) > max_input_lines:
                    parts.append(f"... ({len(lines) - max_input_lines} more lines)")
        elif is_timeout:
            parts.append("Time Limit Exceeded")
            parts.append("")
            if stdin:
                parts.append("Last Executed Input")
                parts.append(_truncate(str(stdin), max_input_chars))
        elif is_incorrect_format:
            if was_truncated:
                parts.append("Truncated Attempt: Your previous response was too long and truncated because it reached the maximum response length. Try again with a shorter response.")
            else:
                parts.append("Incorrect Format: Put your code inside a ```python ... ``` block.")
        else:
            test_idx = r["test_idx"] + 1
            parts.append(f"Test Case {test_idx}: Wrong Answer")
            parts.append("")
            if stdin:
                parts.append("Input")
                parts.append(_truncate(str(stdin), max_input_chars))
            parts.append("")
            parts.append("Output")
            parts.append(_truncate(str(actual), max_input_chars))

        parts.append("")

    result = "\n".join(parts).rstrip()
    if len(result) > max_length:
        result = result[:max_length]
    return result


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    sparse_rewards: bool = False,
    max_test_cases: Optional[int] = None,
    max_execution_time: float = 10.0,
    **kwargs,
) -> dict:
    """
    Compute reward for a competition coding solution.

    Args:
        solution_str: Model output containing Python code in a ```python block
        ground_truth: JSON string of list of assertion strings
        extra_info: Dict with "split", "index", etc.
        sparse_rewards: If True, reward is 1.0 only if all tests pass
        max_test_cases: Limit number of tests during training
        max_execution_time: Per-solve-call time limit in seconds
    """
    split = extra_info.get("split", "train") if extra_info else "train"
    was_truncated = extra_info.get("truncated", False) if extra_info else False

    if split == "test":
        sparse_rewards = True

    try:
        tests = json.loads(ground_truth)
        if not isinstance(tests, list):
            raise ValueError("ground_truth must be a JSON list of assertion strings")
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

    records = run_tests(
        tests=tests,
        solution=solution_str,
        max_test_cases=max_test_cases if split != "test" else None,
        max_execution_time=max_execution_time,
    )

    correct_answers = [1.0 if r["passed"] else 0.0 for r in records]
    predictions = str([r["actual"] for r in records])[-5000:]
    accuracy = np.mean(correct_answers)

    if sparse_rewards:
        reward = 1.0 if accuracy == 1.0 else 0.0
    else:
        reward = float(accuracy)

    incorrect_format = (len(records) == 1) and (not records[0]["passed"]) and (records[0]["actual"] == INCORRECT_FORMAT)
    error_in_test_cases = any(
        (not r["passed"]) and isinstance(r["actual"], str) and ERROR_PREFIX in r["actual"]
        for r in records
    )
    timed_out = any(
        (not r["passed"]) and r["actual"] == TIMEOUT
        for r in records
    )

    return {
        "score": reward,
        "acc": float(accuracy),
        "pred": predictions,
        "incorrect_format": 1 if incorrect_format else 0,
        "error_in_test_cases": 1 if error_in_test_cases else 0,
        "timed_out": 1 if timed_out else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": format_test_feedback(records, was_truncated=was_truncated),
    }
