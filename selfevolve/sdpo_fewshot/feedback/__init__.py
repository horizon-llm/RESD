from ....utils.reward_score.feedback import math
from ....utils.reward_score.feedback import code
from ....utils.reward_score.feedback import gpqa
from ....utils.reward_score.feedback import mcq
from ....utils.reward_score.feedback import tooluse
from ....utils.reward_score.feedback import IFeval
from ....utils.reward_score.feedback import qa
from ....utils.reward_score.feedback import privacy

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval","MedQA"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    elif data_source in ["IFeval"]:
        results = IFeval.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["WikiDYK"]:
        results = qa.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["ContextualIntegritySynthetic"]:
        results = privacy.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
