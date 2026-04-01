"""
SkyDiscover math benchmarks: unified reward via each task's `evaluate(program_path)`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

_BASE = Path(__file__).resolve().parent / "skydiscover_math"


def _extract_code(response: str | None) -> str | None:
    if not response:
        return None
    blocks = re.findall(r"```(\w*)\n(.*?)```", response, re.DOTALL)
    if not blocks:
        return None
    return max((code for _, code in blocks), key=len)


# Evaluators run candidate code via importlib / subprocess using the same interpreter as training.
# Models often paste `import matplotlib` at module scope even when unused; if that interpreter
# has no matplotlib (or Ray uses a different env), evaluation fails before the real entrypoint runs.
# Strip only top-of-file style lines so lazy imports inside functions remain (e.g. optional viz).
_VIZ_TOP_IMPORT = re.compile(
    r"^\s*(?:import\s+matplotlib\b.*|from\s+matplotlib\b.*|import\s+seaborn\b.*|from\s+seaborn\b.*)\s*(?:#.*)?$",
)


def _strip_optional_plot_imports(code: str) -> str:
    out_lines: list[str] = []
    for line in code.splitlines():
        if _VIZ_TOP_IMPORT.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


_EVALUATOR_REL_PATHS: dict[str, str] = {
    "heilbronn_triangle": "heilbronn_triangle/evaluator/evaluator.py",
    "circle_packing": "circle_packing/evaluator.py",
    "first_autocorr_ineq": "first_autocorr_ineq/evaluator/evaluator.py",
    "second_autocorr_ineq": "second_autocorr_ineq/evaluator/evaluator.py",
    "third_autocorr_ineq": "third_autocorr_ineq/evaluator/evaluator.py",
    "uncertainty_ineq": "uncertainty_ineq/evaluator/evaluator.py",
    "sums_diffs_finite_sets": "sums_diffs_finite_sets/evaluator/evaluator.py",
    "minimizing_max_min_dist_2": "minimizing_max_min_dist_2/evaluator/evaluator.py",
    "minimizing_max_min_dist_3": "minimizing_max_min_dist_3/evaluator/evaluator.py",
    "circle_packing_rect": "circle_packing_rect/evaluator/evaluator.py",
    "hexagon_packing_11": "hexagon_packing_11/evaluator/evaluator.py",
    "hexagon_packing_12": "hexagon_packing_12/evaluator/evaluator.py",
    "matmul": "matmul/evaluator/evaluator.py",
    "erdos_min_overlap": "erdos_min_overlap/evaluator/evaluator.py",
    "heilbronn_convex_13": "heilbronn_convex_13/evaluator/evaluator.py",
    "heilbronn_convex_14": "heilbronn_convex_14/evaluator/evaluator.py",
    "signal_processing": "signal_processing/evaluator/evaluator.py",
}

_EVAL_FN_CACHE: dict[str, Callable[..., dict[str, Any]]] = {}


def _load_evaluate(benchmark: str) -> Callable[..., dict[str, Any]]:
    if benchmark in _EVAL_FN_CACHE:
        return _EVAL_FN_CACHE[benchmark]
    rel = _EVALUATOR_REL_PATHS.get(benchmark)
    if not rel:
        raise KeyError(f"Unknown benchmark: {benchmark}")
    path = _BASE / rel
    if not path.is_file():
        raise FileNotFoundError(f"Evaluator not found: {path}")
    name = f"skydiscover_math_{benchmark}_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "evaluate", None)
    if not callable(fn):
        raise AttributeError(f"No evaluate() in {path}")
    _EVAL_FN_CACHE[benchmark] = fn
    return fn


def _parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    raise TypeError(f"ground_truth must be str or dict, got {type(ground_truth)}")




def _is_timeout_result(result: dict[str, Any]) -> bool:
    parts = [str(result.get("error") or "")]
    art = result.get("artifacts")
    if isinstance(art, dict):
        parts.append(str(art.get("error", "")))
    low = " ".join(parts).lower()
    return "timeout" in low or "timed out" in low


def _build_feedback(
    benchmark: str,
    result: dict[str, Any],
    timed_out: bool,
    incorrect_format: bool,
    outer_error: str,
) -> str:
    if incorrect_format:
        return "Missing or empty ```python ... ``` code block."
    if outer_error:
        return f"Evaluation error: {outer_error}"
    if timed_out:
        return "Execution timed out"

    err = result.get("error") or ""
    score = float(result.get("combined_score", 0.0))

    if benchmark == "circle_packing":
        if result.get("validity", 0) >= 1.0 and score > 0:
            sr = result.get("sum_radii", 0.0)
            return f"Success (sum_radii={float(sr):.6f})"
        reason = err or "constraints violated (overlap, out of square, or bad shapes)"
        return f"Invalid packing: {reason}"

    if benchmark == "circle_packing_rect":
        if score > 0 and not err:
            rs = result.get("radii_sum", 0.0)
            return f"Success (radii_sum={float(rs):.6f})"
        return f"Invalid packing: {err or 'validation failed'}"

    if benchmark in ("hexagon_packing_11", "hexagon_packing_12"):
        if score > 0 and not err:
            inv = result.get("inv_outer_hex_side_length", 0.0)
            return f"Success (inv_outer_hex_side_length={float(inv):.6f})"
        return f"Invalid packing: {err or 'validation failed'}"

    if benchmark == "erdos_min_overlap":
        if score > 0 and not err:
            c5 = result.get("c5_bound", 0.0)
            return f"Success (c5_bound={float(c5):.6f})"
        return err or "Verification failed"

    if benchmark == "matmul":
        if score > 0 and not err:
            r = result.get("rank", 0.0)
            loss = result.get("loss", 0.0)
            return f"Success (rank={r}, loss={loss})"
        return err or "Tensor decomposition check failed"

    if benchmark == "signal_processing":
        if score > 0 and not err:
            cc = result.get("composite_score", result.get("overall_score", 0.0))
            return f"Success (composite_score={float(cc):.6f})"
        return err or "Signal processing metrics failed"

    if benchmark == "heilbronn_triangle" or benchmark.startswith("heilbronn_convex"):
        if score > 0 and not err:
            mn = result.get("min_area_normalized", 0.0)
            return f"Success (min_area_normalized={float(mn):.6f})"
        return err or "Heilbronn constraints failed"

    if "autocorr" in benchmark or benchmark == "uncertainty_ineq" or benchmark == "sums_diffs_finite_sets":
        if score > 0 and not err:
            return f"Success (combined_score={score:.6f})"
        return err or "Inequality / verification failed"

    if benchmark.startswith("minimizing_max_min_dist"):
        if score > 0 and not err:
            return f"Success (combined_score={score:.6f})"
        return err or "Distance objective failed"

    if score > 0 and not err:
        return f"Success (combined_score={score:.6f})"
    return err or f"Low or zero score (combined_score={score:.6f})"


def compute_score(
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
   
    del extra_info, kwargs

    gt = _parse_ground_truth(ground_truth)
    benchmark = gt['bench_mark']
    pred = _extract_code(solution_str)
    incorrect_format = pred is None or (isinstance(pred, str) and not pred.strip())

    if incorrect_format:
        return {
            "score": 0.0,
            "pred": pred or "",
            "incorrect_format": 1,
            "error": "",
            "timed_out": False,
            "feedback": _build_feedback(
                benchmark or "",
                {},
                False,
                True,
                "",
            ),
        }

    pred = _strip_optional_plot_imports(pred)

    program_path: str | None = None

    evaluate_fn = _load_evaluate(benchmark)

    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(pred)
        tmp.close()
        program_path = tmp.name

        result = evaluate_fn(program_path)
        if not isinstance(result, dict):
            result = {}

        score = float(result.get("combined_score", 0.0))
        err_msg = str(result.get("error", "") or "")
        timed_out = _is_timeout_result(result)

        return {
            "score": score,
            "pred": pred,
            "incorrect_format": 0,
            "error": err_msg,
            "timed_out": timed_out,
            "feedback": _build_feedback(benchmark, result, timed_out, False, ""),
        }
    except Exception as e:
        es = str(e)
        timed_out = bool(re.search(r"timeout|timed out", es, re.I))
        return {
            "score": 0.0,
            "pred": pred,
            "incorrect_format": 0,
            "error": es,
            "timed_out": timed_out,
            "feedback": _build_feedback(benchmark, {}, timed_out, False, es),
        }
    finally:
        if program_path and os.path.isfile(program_path):
            try:
                os.unlink(program_path)
            except OSError:
                pass


__all__ = ["compute_score", "_EVALUATOR_REL_PATHS"]
