from typing import Optional
import json
import re


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""


def remove_boxed(s: str) -> str:
    r"""Remove the LaTeX \boxed{} command from a string."""
    left = r"\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    else:
        return ""


def _remove_thinking_trace(text: str) -> str:
    # Case 1: complete <think>...</think> block in response
    out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Case 2: <think> was in the prompt, response starts with thinking content
    out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
    return out_text


def extract_answer(solution_str: str) -> tuple[str, bool]:
    """Try to extract answer from \\boxed{}, fall back to raw response.

    Returns:
        (extracted_answer, used_boxed_format)
    """
    solution_str = _remove_thinking_trace(solution_str)
    # Strip <answer>...</answer> wrapper if present
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.DOTALL)
    if answer_match:
        solution_str = answer_match.group(1).strip()
    boxed = last_boxed_only_string(solution_str)
    if boxed:
        pred = remove_boxed(boxed)
        if pred:
            return pred, True
    return solution_str.strip(), False


def wrap_score(score_fn, solution_str, ground_truth, extra_info=None, sparse_rewards=False):
    """Shared wrapper: extract answer, reconstruct entry, call score_fn, return standard dict."""
    was_truncated = extra_info.get("truncated", False) if extra_info else False
    split = extra_info.get("split", "train") if extra_info else "train"

    # Test split always uses dense (partial-credit) rewards for faithful evaluation.
    if split == "test":
        sparse_rewards = False

    answer, used_boxed = extract_answer(solution_str)

    # Reconstruct the reasoning-gym entry dict.
    # The data generation script stores task-specific metadata (e.g. num_disks,
    # solution, board_config) in "metadata_full" as a JSON string, while
    # "metadata" only contains generic fields (source_dataset, difficulty, ...).
    # Prefer metadata_full so that task-specific scoring functions can access
    # the fields they need (partial credit, detailed feedback, etc.).
    metadata = extra_info.get("metadata", {}) if extra_info else {}
    metadata_full = extra_info.get("metadata_full", None) if extra_info else None
    if metadata_full:
        if isinstance(metadata_full, str):
            try:
                metadata_full = json.loads(metadata_full)
            except (json.JSONDecodeError, TypeError):
                metadata_full = None
        if isinstance(metadata_full, dict):
            metadata = metadata_full
    question = extra_info.get("question", "") if extra_info else ""
    entry = {
        "question": question,
        "answer": ground_truth,
        "metadata": metadata,
    }

    try:
        result = score_fn(answer, entry)
        if isinstance(result, tuple):
            rg_score, task_feedback = result
        else:
            rg_score, task_feedback = result, ""
    except Exception:
        rg_score = 1.0 if answer == ground_truth else 0.0
        task_feedback = ""

    acc = 1.0 if rg_score >= 1.0 else 0.0

    if sparse_rewards:
        rg_score = acc

    feedback_parts = []
    if was_truncated:
        feedback_parts.append("Your response was truncated because it exceeded the maximum length.")
    if task_feedback:
        feedback_parts.append(task_feedback)
    feedback = " ".join(feedback_parts)

    return {
        "score": rg_score,
        "acc": acc,
        "pred": answer,
        "incorrect_format": 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 0,
        "feedback": feedback,
    }


def default_score_answer(answer, entry):
    """Default scoring: exact match or substring partial credit."""
    if not isinstance(answer, str) or len(answer) == 0:
        return 0.0, "Empty or invalid answer."
    oracle_answer = entry["answer"]
    if answer == oracle_answer:
        return 1.0, ""
    elif oracle_answer in answer:
        score = len(oracle_answer) / len(answer)
        return score, f"Answer contains the correct solution but has extra content ({len(answer)} chars vs expected {len(oracle_answer)})."
    return 0.0, f"Incorrect answer. Expected: {oracle_answer}"


# Task registry mapping source_dataset -> score_answer function
_TASK_REGISTRY = {}


def _load_sibling_module(name):
    """Load a sibling .py file by absolute path, avoiding relative imports.

    This is needed because the module is loaded dynamically via
    importlib.util.spec_from_file_location with a synthetic name,
    which breaks `from . import ...` style relative imports.
    """
    import importlib.util
    import os

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(parent_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"reasoning_gym_games.{name}", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_registry():
    if _TASK_REGISTRY:
        return

    task_names = [
        "sudoku",
        "mini_sudoku",
        "futoshiki",
        "survo",
        "kakurasu",
        "tower_of_hanoi",
        "n_queens",
        "tsumego",
        "emoji_mystery",
        "countdown",
        "puzzle24",
        "rush_hour",
        "sokoban",
        "boxnet",
        "knight_swap",
        "codeio",
        "bf",
    ]
    for name in task_names:
        mod = _load_sibling_module(name)
        _TASK_REGISTRY[name] = mod.score_answer

    # Default scoring tasks (no separate file needed)
    _TASK_REGISTRY["maze"] = default_score_answer
    _TASK_REGISTRY["mahjong_puzzle"] = default_score_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None, sparse_rewards=False, **kwargs):
    """Main dispatcher: route to per-task scoring based on source_dataset."""
    _ensure_registry()

    metadata = extra_info.get("metadata", {}) if extra_info else {}
    source_dataset = metadata.get("source_dataset", "")

    score_fn = _TASK_REGISTRY.get(source_dataset, default_score_answer)
    result = wrap_score(score_fn, solution_str, ground_truth, extra_info, sparse_rewards=sparse_rewards)
    print(f"[REWARD] data_source={data_source}, task={source_dataset}, score={result['score']}, acc={result['acc']}", flush=True)
    return result
