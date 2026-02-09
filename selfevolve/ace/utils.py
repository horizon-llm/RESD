import re
import os
import json
from typing import Dict, List

def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None

def extract_bullet_tags(
    response: str,
    use_json_mode: bool
) -> List[Dict[str, str]]:
    """
    Extract bullet tags from reflector response.
    
    Args:
        response: The reflector's response
        use_json_mode: Whether JSON mode was used
        
    Returns:
        List of dicts with 'id' and 'tag' keys
    """
    bullet_tags = []
    
    if use_json_mode:
        try:
            response_json = json.loads(response)
            bullet_tags = response_json.get("bullet_tags", [])
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Failed to parse bullet tags from JSON response")
    else:
        # Try to extract from non-JSON response
        # This is a fallback and may not always work
        try:
            # Look for JSON-like structure in the response
            start_idx = response.find('"bullet_tags"')
            if start_idx != -1:
                # Find the array
                bracket_idx = response.find('[', start_idx)
                if bracket_idx != -1:
                    # Find matching closing bracket
                    depth = 0
                    end_idx = bracket_idx
                    for i in range(bracket_idx, len(response)):
                        if response[i] == '[':
                            depth += 1
                        elif response[i] == ']':
                            depth -= 1
                            if depth == 0:
                                end_idx = i + 1
                                break
                    
                    bullet_tags_str = response[bracket_idx:end_idx]
                    bullet_tags = json.loads(bullet_tags_str)
        except Exception as e:
            print(f"Warning: Failed to extract bullet tags: {e}")
    
    return bullet_tags

def extract_bullet_ids(response: str, use_json_mode: bool) -> List[str]:
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

def extract_boxed_content(text):
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

def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer
        
        return "No final answer found"

def compute_score(solution_str, ground_truth):
    answer = extract_answer(solution_str)
    if answer is None:
        correct = False
    else:
        if answer == ground_truth:
            correct = True
        else:
            correct = False
    
    reward = 1.0 if correct else -1.0
    acc = correct
    
    return {
        "score": reward,
        "acc": acc,
        "pred": answer,
    }