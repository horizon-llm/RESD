import random
import re


def extract_solution(solution_str):
    """Extract the answer from <answer>...</answer> tags."""
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

def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0, **kwargs):
    """Reward function for FiNER XBRL tagging task.

    Supports partial credit: if the model gets 3 out of 4 tags right,
    score is 0.75 * score.

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

    if answer is None:
        return 0
    else:
        tag_score = finer_tag_score(answer, ground_truth)
        if tag_score == 1.0:
            return score
        else:
            return format_score

def compute_score_count(solution_str, ground_truth, format_score=0.0, score=1.0, **kwargs):
    """Reward function for FiNER XBRL tagging task.

    Supports partial credit: if the model gets 3 out of 4 tags right,
    score is 0.75 * score.

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

    if answer is None:
        return 0
    else:
        tag_score = finer_tag_score(answer, ground_truth)
        if tag_score == 1.0:
            return score
        else:
            return format_score + tag_score * score
