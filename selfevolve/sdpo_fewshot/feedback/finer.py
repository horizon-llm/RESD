import random
import re


def _remove_thinking_trace(text: str) -> str:
    # Case 1: complete <think>...</think> block in response
    out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Case 2: <think> was in the prompt, response starts with thinking content
    out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
    return out_text

def extract_solution(solution_str):
    """Extract the answer from <answer>...</answer> tags."""
    solution_str = _remove_thinking_trace(solution_str)
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) < 1:
        return None

    # Return the last match (in case model outputs multiple)
    return matches[-1].group(1).strip()

def finer_tag_score(predicted, ground_truth):
    """Compute partial credit score for FiNER tag matching.

    Splits predicted and ground truth by comma, compares positionally
    (case-insensitive), and returns fraction of correct tags.

    Args:
        predicted: comma-separated predicted tags
        ground_truth: comma-separated ground truth tags

    Returns:
        float: fraction of correctly matched tags (0.0 to 1.0)
    """
    pred = predicted.split(",")
    pred = [val.lower().strip() for val in pred]
    label = ground_truth.split(",")
    label = [val.lower().strip() for val in label]
    count = 0

    # Handle length mismatch: truncate or pad predictions
    if len(pred) > len(label):
        pred = pred[:len(label)]
    elif len(pred) < len(label):
        pred += [""] * (len(label) - len(pred))

    for prediction, ground_truth in zip(pred, label):
        try:
            ground_truth = eval(ground_truth)
            prediction = eval(prediction.replace(",", "").replace("$", ""))
        except:
            pass
        if prediction == ground_truth:
            count += 1
    score = count / len(pred) if pred else 0
    return score

def compute_score(
    solution_str,
    ground_truth,
    format_score=0.0,
    score=1.0,
    extra_info=None,
    **kwargs,
):
    """Reward function for FiNER XBRL tagging task.

    No partial credit: model gets full score only if all tags are correct.

    Args:
        solution_str: the model's full output text
        ground_truth: dict with key "target" containing ground truth tags
        format_score: score when format is correct but answer is wrong
        score: maximum score for a fully correct answer
    """
    answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")
    
    was_truncated = extra_info.get("truncated", False) if extra_info else False
    incorrect_format = answer is None

    if answer is None:
        return {
            "score": 0,
            "acc": 0,
            "pred": "",
            "incorrect_format": 1,
            "truncated": 1 if was_truncated else 0,
            "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
            "feedback": "Your answer had the wrong format. The solution must be given in the format: <answer>your_answer</answer>."
        }
    else:
        tag_score = finer_tag_score(answer, ground_truth)
        if tag_score == 1.0:
            return {
                "score": score,
                "acc": 1,
                "pred": answer,
                "incorrect_format": 0,
                "truncated": 1 if was_truncated else 0,
                "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
                "feedback": ""
            }
        else:
            return {
                "score": format_score,
                "acc": 0,
                "pred": answer,
                "incorrect_format": 0,
                "truncated": 1 if was_truncated else 0,
                "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
                "feedback": "Your answer is incorrect. The correct answer is {ground_truth}."
            }

def compute_score_count(
    solution_str,
    ground_truth,
    format_score=0.0,
    score=1.0,
    extra_info=None,
    format_feedback=True,
    correctness_feedback=False,
    **kwargs,
):
    """Reward function for FiNER XBRL tagging task with feedback.

    Supports partial credit: if the model gets 3 out of 4 tags right,
    score is 0.75 * score.

    Args:
        solution_str: the model's full output text
        ground_truth: comma-separated ground truth tags
        format_score: score when format is correct but answer is wrong
        score: maximum score for a fully correct answer
        extra_info: dict with optional "truncated" flag
        format_feedback: whether to generate format feedback
        correctness_feedback: whether to generate correctness feedback
    """
    answer = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    was_truncated = extra_info.get("truncated", False) if extra_info else False
    incorrect_format = answer is None

    if answer is None:
        final_score = 0
        tag_score = 0.0
    else:
        tag_score = finer_tag_score(answer, ground_truth)
        if tag_score == 1.0:
            final_score = score
        else:
            final_score = format_score + tag_score * score

    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = "Your answer had the wrong format. The solution must be given in the format: <answer>your_answer</answer>."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif tag_score > 0.0 and tag_score < 1.0 and not incorrect_format and correctness_feedback:
        feedback = f"Your answer is partially correct. The correct answer is {ground_truth}."
    elif tag_score == 0.0 and not incorrect_format and correctness_feedback:
        feedback = f"Your answer is completely incorrect. The correct answer is {ground_truth}."
    
    pred = answer if answer is not None else ""

    return {
        "score": final_score,
        "acc": 1 if final_score == score else 0,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }