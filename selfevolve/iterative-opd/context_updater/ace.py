import json
import re
from typing import List, Tuple, Dict, Any
from copy import deepcopy

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from .prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT
from .playbook_utils import extract_playbook_bullets, update_bullet_counts, get_playbook_stats, extract_json_from_text, apply_curator_operations

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

def _extract_bullet_tags(
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

def _extract_and_validate_operations(
    response: str
) -> Dict[str, Any]:
    """
    Extract and validate operations from curator response.
    
    Args:
        response: The curator's response
        
    Returns:
        Dictionary with 'reasoning' and 'operations' keys
        
    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    # Extract operations info
    operations_info = extract_json_from_text(response, "operations")
    
    # Validate JSON structure is correct
    if not operations_info:
        raise ValueError("Failed to extract valid JSON from curator response")
    
    # Validate required fields
    if "reasoning" not in operations_info:
        raise ValueError("JSON missing required 'reasoning' field")
    
    if "operations" not in operations_info:
        raise ValueError("JSON missing required 'operations' field")
    
    # Validate field types
    if not isinstance(operations_info["reasoning"], str):
        raise ValueError("'reasoning' field must be a string")
    
    if not isinstance(operations_info["operations"], list):
        raise ValueError("'operations' field must be a list")
    
    # Validate operations structure
    for i, op in enumerate(operations_info["operations"]):
        if not isinstance(op, dict):
            raise ValueError(f"Operation {i} must be a dictionary")
        
        if "type" not in op:
            raise ValueError(f"Operation {i} missing required 'type' field")
        
        op_type = op["type"]
        
        # Currently only ADD operations are fully supported
        # Note: You can add support for UPDATE, MERGE, DELETE operations here
        if op_type not in ["ADD", "UPDATE", "MERGE", "DELETE", "CREATE_META"]:
            print(f"Warning: Operation type '{op_type}' may not be fully supported")
        
        # Validate ADD operation structure
        if op_type == "ADD":
            required_fields = {"type", "section", "content"}
            missing_fields = required_fields - set(op.keys())
            if missing_fields:
                raise ValueError(f"ADD operation {i} missing fields: {list(missing_fields)}")
    
    return operations_info

class ACEContextUpdater:
    def __init__(self, config):
        self.config = config

        self.playbook = self._initialize_empty_playbook()
        self.next_global_id = 1
    
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
        responses = batch.batch["responses"]
        response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1] for msgs in batch.non_tensor_batch["raw_prompt"]]

        # extract playbook parts used by the generator
        bullet_ids = [_extract_bullet_ids(response, use_json_mode=True) for response in response_texts]
        playbook_parts_used = extract_playbook_bullets(deepcopy(self.playbook), bullet_ids)
        
        # prepare reflector prompts
        reflector_prompts = []
        for prompt, response, feedback, playbook_parts in zip(prompt_texts, response_texts, feedback_list, playbook_parts_used):
            reflector_prompt = REFLECTOR_PROMPT.format(
                prompt=prompt,
                response=response,
                feedback=feedback,
                playbook=playbook_parts
            )
            reflector_prompts.append(reflector_prompt)

        # Tokenize
        encoding = tokenizer(reflector_prompts, padding=True, truncation=True, return_tensors="pt", truncation_side="right").to(device)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        position_ids = compute_position_id_with_mask(attention_mask).to(device)

        # Build DataProto for the worker
        reflector_batch = DataProto.from_dict(tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        })

        # Pad to be divisible by worker.world_size
        reflector_batch_padded, pad_size = pad_dataproto_to_divisor(reflector_batch, worker.world_size)

        # Generate reflections via the rollout worker
        reflector_output_padded = worker.generate_sequences(reflector_batch_padded)
        reflector_output = unpad_dataproto(reflector_output_padded, pad_size)

        # Decode reflections and extract bullet tags
        reflection_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in reflector_output.batch["responses"]
        ]

        bullet_tags = [_extract_bullet_tags(response, use_json_mode=True) for response in reflection_texts]

        # Sequentially update the bullet counts
        for tags in bullet_tags:
            if tags:
                self.playbook = update_bullet_counts(deepcopy(self.playbook), tags)
        
        # prepare curator prompts
        stats = get_playbook_stats(self.playbook)
        stats_str = json.dumps(stats, indent=2)
        curator_prompts = []
        for prompt, reflection in zip(prompt_texts, reflection_texts):
            curator_prompt = CURATOR_PROMPT.format(
                prompt=prompt,
                recent_reflection=reflection,
                playbook_stats=stats_str,
                current_playbook=self.playbook
            )
            curator_prompts.append(curator_prompt)
        
        # Tokenize curator prompts
        curator_encoding = tokenizer(curator_prompts, padding=True, truncation=True, return_tensors="pt", truncation_side="right").to(device)
        curator_input_ids = curator_encoding["input_ids"].to(device)
        curator_attention_mask = curator_encoding["attention_mask"].to(device)
        curator_position_ids = compute_position_id_with_mask(curator_attention_mask).to(device)

        # Build DataProto for the worker
        curator_batch = DataProto.from_dict(tensors={
            "input_ids": curator_input_ids,
            "attention_mask": curator_attention_mask,
            "position_ids": curator_position_ids
        })

        # Pad to be divisible by worker.world_size
        curator_batch_padded, curator_pad_size = pad_dataproto_to_divisor(curator_batch, worker.world_size)

        # Generate curated playbooks via the rollout worker
        curator_output_padded = worker.generate_sequences(curator_batch_padded)
        curator_output = unpad_dataproto(curator_output_padded, curator_pad_size)

        # Decode
        curation_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in curator_output.batch["responses"]
        ]

        # Post-process curator outputs to update the playbook
        all_operations = []
        for curation in curation_texts:
            if curation.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
                print(f"⏭️  Skipping curator operation due to empty response")
                continue

            # Extract and validate operations
            try:
                operations_info = _extract_and_validate_operations(curation)

                operations = operations_info["operations"]
                all_operations.extend(operations)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                print(f"❌ Curator JSON parsing failed: {e}")
                print(f"📄 Raw curator response preview: {curation[:300]}...")

                print("⏭️  Skipping curator operation due to invalid JSON format")
                continue
            except Exception as e:
                print(f"❌ Curator operation failed: {e}")
                print(f"📄 Raw curator response preview: {curation[:300]}...")

                print("⏭️  Skipping curator operation and continuing training")
                continue
        if all_operations:
            self.playbook, self.next_global_id = apply_curator_operations(
                self.playbook, all_operations, self.next_global_id
            )
        
        return reflection_texts
