import json
import re
from typing import List, Tuple

from verl import DataProto

from .prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT

def _extract_bullet_ids(response: str, use_json_mode: bool) -> List[str]:
    """
    Extract bullet IDs from generator response.
    
    Args:
        response: The generator's response
        use_json_mode: Whether JSON mode was used
        
    Returns:
        List of bullet IDs
    """
    bullet_ids = []
    
    if use_json_mode:
        try:
            response_json = json.loads(response)
            bullet_ids = response_json.get("bullet_ids", [])
        except (json.JSONDecodeError, KeyError):
            # If parsing fails, try regex extraction
            bullet_ids = _extract_bullet_ids_regex(response)
    else:
        bullet_ids = _extract_bullet_ids_regex(response)
    
    return bullet_ids

def _extract_bullet_ids_regex(text: str) -> List[str]:
    """
    Extract bullet IDs using regex pattern matching.
    
    Args:
        text: Text to extract bullet IDs from
        
    Returns:
        List of bullet IDs
    """
    # Pattern matches: [xxx-00001], [abc-00042], etc.
    pattern = r'\[([a-z]{3,}-\d{5})\]'
    matches = re.findall(pattern, text)
    return matches

class ACEContextUpdater:
    def __init__(self, config):
        self.config = config

        self.playbook = self._initialize_empty_playbook()
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    def update(self, batch: DataProto, worker, tokenizer, feedback_list: List[str]):
        # get response texts
        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        bullet_ids = 
        prompt_texts = [msgs[-1] for msgs in batch.non_tensor_batch["raw_prompt"]]
        
        # generate reflector prompts
        reflector_prompts = []
        for prompt, response, feedback in zip(prompt_texts, response_texts, feedback_list):
            reflector_prompt = REFLECTOR_PROMPT.format(
                prompt,
                response,
                feedback
            )
            reflector_prompts.append(reflector_prompt)