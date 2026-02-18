import json
import re
from typing import List, Tuple

from verl import DataProto

from .prompts.ace_generator import GENERATOR_PROMPT

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

    def update(self, batch: DataProto):
        pass