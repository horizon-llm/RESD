import importlib
import json
import re
import sys
import types
from pathlib import Path
from typing import Any


_IFEVAL_DIR = Path(__file__).resolve().parents[1] / "third_party" / "instruction_following_eval"


def _remove_thinking_trace(text: str) -> str:
    # Case 1: complete <think>...</think> block in response
    out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Case 2: <think> was in the prompt, response starts with thinking content
    out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
    return out_text


def _ensure_ifeval_pkg() -> None:
    """Expose vendored IFEval directory as `instruction_following_eval` package."""
    if "instruction_following_eval" in sys.modules:
        return
    pkg = types.ModuleType("instruction_following_eval")
    pkg.__path__ = [str(_IFEVAL_DIR)]  # type: ignore[attr-defined]
    sys.modules["instruction_following_eval"] = pkg


def _load_registry():
    _ensure_ifeval_pkg()
    return importlib.import_module("instruction_following_eval.instructions_registry")


def _parse_ground_truth(ground_truth: Any) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        return ground_truth
    raise ValueError(f"Unsupported ground_truth type: {type(ground_truth).__name__}")


def _response_variants(response: str) -> list[str]:
    lines = response.split("\n")
    response_remove_first = "\n".join(lines[1:]).strip()
    response_remove_last = "\n".join(lines[:-1]).strip()
    response_remove_both = "\n".join(lines[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    return [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]


def _check_following(
    response: str,
    prompt: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
    loose: bool = False,
) -> list[bool]:
    registry = _load_registry()
    candidates = _response_variants(response) if loose else [response]
    results: list[bool] = []

    for idx, instruction_id in enumerate(instruction_id_list):
        try:
            instruction_cls = registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            filtered_kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
            instruction.build_description(**filtered_kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            followed = False
            for cand in candidates:
                if cand.strip() and instruction.check_following(cand):
                    followed = True
                    break
            results.append(followed)
        except Exception:
            # Keep reward computation robust in training loops.
            results.append(False)
    return results


def compute_score(solution_str: str, ground_truth: Any, extra_info: dict | None = None, **kwargs) -> dict:
    """
    Compute IFEval reward based on official strict/loose matching logic.

    ground_truth is expected to contain:
      - prompt: str
      - instruction_id_list: list[str]
      - kwargs: list[dict]
    """
    extra_info = extra_info or {}
    response = "" if solution_str is None else _remove_thinking_trace(str(solution_str))
    was_truncated = bool(extra_info.get("truncated", False))

    try:
        gt = _parse_ground_truth(ground_truth)
        prompt = gt.get("prompt", extra_info.get("prompt", ""))
        instruction_id_list = gt["instruction_id_list"]
        kwargs_list = gt["kwargs"]
        if not isinstance(instruction_id_list, list) or not isinstance(kwargs_list, list):
            raise ValueError("instruction_id_list/kwargs must be lists")
        if len(instruction_id_list) != len(kwargs_list):
            raise ValueError("instruction_id_list and kwargs length mismatch")
    except Exception as e:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pred": "",
            "incorrect_format": 1,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if was_truncated else 0,
            "feedback": f"Invalid IFEval ground truth: {e}",
        }

    strict_list = _check_following(
        response=response,
        prompt=prompt,
        instruction_id_list=instruction_id_list,
        kwargs_list=kwargs_list,
        loose=False,
    )
    loose_list = _check_following(
        response=response,
        prompt=prompt,
        instruction_id_list=instruction_id_list,
        kwargs_list=kwargs_list,
        loose=True,
    )

    total = max(len(strict_list), 1)
    strict_inst_acc = sum(strict_list) / total
    loose_inst_acc = sum(loose_list) / total
    strict_prompt_acc = float(all(strict_list))
    loose_prompt_acc = float(all(loose_list))

    # Dense reward for RL training while keeping official prompt-level metric in acc.
    score = strict_inst_acc
    acc = strict_prompt_acc
    incorrect_format = int(not response.strip())

    failed_ids = [iid for iid, ok in zip(instruction_id_list, strict_list, strict=True) if not ok]
    if not response.strip():
        feedback = "Empty response."
    elif failed_ids:
        feedback = "Failed instructions: " + ", ".join(failed_ids[:8])
    else:
        feedback = ""

    pred = json.dumps(
        {
            "strict_follow_instruction_list": strict_list,
            "loose_follow_instruction_list": loose_list,
        },
        ensure_ascii=False,
    )

    return {
        "score": score,
        "acc": acc,
        "pred": pred,
        "incorrect_format": incorrect_format,
        "strict_prompt_acc": strict_prompt_acc,
        "loose_prompt_acc": loose_prompt_acc,
        "strict_inst_acc": strict_inst_acc,
        "loose_inst_acc": loose_inst_acc,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }