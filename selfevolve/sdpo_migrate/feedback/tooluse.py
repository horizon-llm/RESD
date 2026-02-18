import re
import json
import random
import logging
from collections import Counter

logger = logging.getLogger(__name__)

DEBUG_PRINT_PROB = 0.05  # probability of printing parsed predictions for debugging


def strip_thinking_block(text: str) -> str:
    """Strip <think>...</think> blocks from the text to avoid extracting
    rehearsed actions from the thinking model's internal reasoning."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    text = strip_thinking_block(text)
    actions = re.findall(r'Action:\s*(\w+)', text)
    return actions


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'."""
    text = strip_thinking_block(text)
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)

    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Action Input JSON: {e} | raw block: {block!r}")

    return combined_dict


def merge_action_inputs(action_inputs_list: list[dict]) -> dict:
    """Merge a list of action input dicts into a single dict."""
    combined = {}
    for d in action_inputs_list:
        if d:
            combined.update(d)
    return combined


def is_correct_format(text: str) -> bool:
    """Check if the text contains the expected Action/Action Input format."""
    text = strip_thinking_block(text)
    # Previous implementation required Action Input: immediately after \n, which
    # fails when lines have prefixes (e.g. Ray TaskRunner pid) or extra blank lines:
    #   pattern = re.compile(r"Action:.*?\nAction Input:.*?", re.DOTALL)
    #   return pattern.search(text) is not None
    has_action = bool(re.search(r'Action:\s*\w+', text))
    has_action_input = bool(re.search(r'Action Input:\s*\{', text))
    return has_action and has_action_input


def compute_score(solution_str: str = None, ground_truth: str = None, data_source: str = None, extra_info: dict = None, **kwargs) -> dict:
    """
    Compute score for tooluse task.

    Args:
        solution_str: The model's response text (named to match NaiveRewardManager calling convention)
        ground_truth: JSON string containing list of dicts with 'Action' and 'Action_Input' keys
                      e.g., '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'
        data_source: Data source identifier (unused, accepted for compatibility)
        extra_info: Extra info dict (unused, accepted for compatibility)

    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    solution = solution_str
    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
    except json.JSONDecodeError:
        # If ground_truth is already a list (passed directly), handle that case
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 1,
                "feedback": "Failed to parse ground truth JSON",
            }
    
    # Extract ground truth actions and action inputs
    gt_actions = [item['Action'] for item in gt_list]
    gt_action_inputs_list = []
    for item in gt_list:
        try:
            parsed_input = json.loads(item['Action_Input']) if isinstance(item['Action_Input'], str) else item['Action_Input']
            gt_action_inputs_list.append(parsed_input)
        except (json.JSONDecodeError, KeyError):
            gt_action_inputs_list.append({})
    gt_action_inputs = merge_action_inputs(gt_action_inputs_list)
    
    # Extract predicted actions and action inputs from solution
    pred_actions = extract_actions(solution)
    pred_action_inputs = extract_action_inputs(solution)
    
    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    action_inputs_correct = pred_action_inputs == gt_action_inputs
    
    # Both must be correct for full score
    is_correct = actions_correct and action_inputs_correct
    reward = 1.0 if is_correct else 0.0
    
    # Check format: actions exist, action inputs exist, and JSON was parseable
    correct_format = bool(pred_actions) and bool(pred_action_inputs)
    
    # Build prediction string for logging
    prediction = f"Actions: {pred_actions}, Inputs: {pred_action_inputs}"
    
    # Build feedback
    feedback_parts = []
    if not actions_correct:
        feedback_parts.append(f"Actions mismatch: predicted {pred_actions}, expected {gt_actions}")
    if not action_inputs_correct:
        feedback_parts.append(f"Action inputs mismatch: predicted {pred_action_inputs}, expected {gt_action_inputs}")

    if len(feedback_parts) == 0:
        feedback = "" # no feedback means correct
    else:
        feedback = "; ".join(feedback_parts)
    
    if random.random() < DEBUG_PRINT_PROB:
        logger.info(
            f"[tooluse debug] score={reward} format_ok={correct_format}\n"
            f"  pred_actions={pred_actions} gt_actions={gt_actions}\n"
            f"  pred_inputs={pred_action_inputs} gt_inputs={gt_action_inputs}\n"
            f"  solution (first 500 chars): {strip_thinking_block(solution)[:500]!r}"
        )

    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
    }
