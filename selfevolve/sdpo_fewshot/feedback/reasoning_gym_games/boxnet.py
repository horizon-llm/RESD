import copy
import json
import re
from typing import Optional, Any

import numpy as np


def _action_from_response(pg_dict_input, original_response_dict_list):
    """Execute actions on the grid state and return the resulting state."""
    pg_dict_current = copy.deepcopy(pg_dict_input)

    for original_response_dict in original_response_dict_list:
        transformed_dict = {}
        for key, value in original_response_dict.items():
            coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))
            match = re.match(r"move\((.*?),\s(.*?)\)", value)
            if match:
                item, location = match.groups()
                if "square" in location:
                    location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
                transformed_dict[coordinates] = [item, location]

        for key, value in transformed_dict.items():
            current_pos = f"{key[0]}_{key[1]}"

            if current_pos not in pg_dict_current:
                continue

            # Box-target matching
            if (
                value[0] in pg_dict_current[current_pos]
                and isinstance(value[1], str)
                and value[1] in pg_dict_current[current_pos]
                and value[0].startswith("box_")
                and value[1].startswith("target_")
                and value[0][4:] == value[1][7:]
            ):
                pg_dict_current[current_pos].remove(value[0])
                pg_dict_current[current_pos].remove(value[1])

            # Movement to adjacent square
            elif value[0] in pg_dict_current[current_pos] and isinstance(value[1], tuple):
                if (np.abs(key[0] - value[1][0]) == 0 and np.abs(key[1] - value[1][1]) == 1) or (
                    np.abs(key[0] - value[1][0]) == 1 and np.abs(key[1] - value[1][1]) == 0
                ):
                    target_pos = f"{value[1][0]}_{value[1][1]}"
                    pg_dict_current[current_pos].remove(value[0])
                    pg_dict_current[target_pos].append(value[0])

    return pg_dict_current


def score_answer(answer: Optional[str], entry: dict[str, Any]) -> tuple[float, str]:
    """Score boxnet answer: parse JSON actions, simulate, measure lifted ratio."""
    if answer is None:
        return 0.0, "Empty or invalid answer."

    try:
        answer_dict = json.loads(answer)
    except Exception as e:
        return 0.0, f"Failed to parse JSON answer: {e}"

    pg_dict_returned = _action_from_response(entry["metadata"]["initial_state"], answer_dict)

    initial_boxes = 0
    for items in entry["metadata"]["initial_state"].values():
        initial_boxes += sum(1 for item in items if item.startswith("box_"))

    remaining_boxes = 0
    remaining_names = []
    for pos, items in pg_dict_returned.items():
        for item in items:
            if item.startswith("box_"):
                remaining_boxes += 1
                remaining_names.append(f"{item} at {pos}")

    if initial_boxes == 0:
        return 0.0, "No boxes found in the initial state."

    lifted_ratio = (initial_boxes - remaining_boxes) / initial_boxes
    reward = max(0.05, lifted_ratio)

    if remaining_boxes == 0:
        return reward, ""

    feedback = f"{remaining_boxes}/{initial_boxes} boxes still unmatched."
    if remaining_names:
        examples = remaining_names[:5]
        feedback += " Remaining: " + "; ".join(examples)
        if len(remaining_names) > 5:
            feedback += f" ...and {len(remaining_names) - 5} more."
    if entry.get("answer") is not None:
        feedback += f" The correct solution is: {entry['answer']}"
    return reward, feedback
