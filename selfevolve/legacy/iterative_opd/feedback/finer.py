import random
import re
import json


def _remove_thinking_trace(text: str) -> str:
    # Case 1: complete <think>...</think> block in response
    out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Case 2: <think> was in the prompt, response starts with thinking content
    out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
    return out_text

def _extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_solution(response):
    """Extract final answer from model response"""

    # remove thinking trace
    response = _remove_thinking_trace(response)

    try:
        # First try JSON parsing
        parsed = json.loads(response.strip())
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        answer = None

        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
        
        if answer is None:
            # Try to get final answer from JSON style response with regex matching 
            # Try double quotes first
            matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
            if matches:
                answer = matches[-1]
        
        if answer is None:
            # Try single quotes
            matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
            if matches:
                answer = matches[-1]
        
        if answer is None:
            # Handle JSON format without quotes (for simple expressions)
            matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
            if matches:
                answer = matches[-1].strip()
                # Clean up trailing characters
                answer = re.sub(r'[,}]*$', '', answer)
        
        if answer is None:
            # Fallback for "The final answer is: X" pattern with boxed
            final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
            match = re.search(final_answer_pattern, response)
            if match:
                # Extract boxed content starting from this match
                remaining_text = response[match.start():]
                boxed_content = _extract_boxed_content(remaining_text)
                answer = boxed_content
        
        if answer is None:
            # More general pattern for "final answer is X"
            matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
            if matches:
                answer = matches[-1].strip()
                # Clean up common formatting
                answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
                answer = answer.replace('$', '').strip()
        
        if answer is not None:
            answer_pattern = r"<answer>(.*?)</answer>"
            match = re.finditer(answer_pattern, answer, re.DOTALL)
            matches = list(match)

            if len(matches) < 1:
                return answer.strip()

            # Return the last match (in case model outputs multiple)
            return matches[-1].group(1).strip()
        else:
            # extract from the entire response as a last resort
            answer_pattern = r"<answer>(.*?)</answer>"
            match = re.finditer(answer_pattern, response, re.DOTALL)
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
    elif tag_score == 1.0 and correctness_feedback:
        feedback = "Your answer is completely correct. Great job!"
    
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