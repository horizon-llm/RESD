import re


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def is_correct_format(text: str) -> bool:
    """
    Check if the text is in the correct XML format.

    The text should contain at the end of the text:
    <answer>
    (A|B|C|D)
    </answer>
    """
    pattern = r"<answer>\s*(A|B|C|D)\s*</answer>$"
    return re.search(pattern, text) is not None

def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    multiple_choice_answer = extract_xml_answer(solution_str)

    reward = float(multiple_choice_answer == ground_truth)
    # incorrect_format = is_correct_format(solution_str)
    incorrect_format = multiple_choice_answer not in {"A", "B", "C", "D"}

    if incorrect_format:
        feedback = f"Answer '{multiple_choice_answer}' is not in the correct format. Expected one of A, B, C, D."
    elif reward == 0.0:
        feedback = f"Answer '{multiple_choice_answer}' is incorrect. The correct answer is '{ground_truth}'."
    else:
        feedback = ""

    return {
      "score": reward,
      "acc": reward,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if incorrect_format else 0,
      "feedback": feedback,
    }
