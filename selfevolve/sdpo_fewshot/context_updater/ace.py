import json
import re
import random
from typing import List, Tuple, Dict, Any
from copy import deepcopy

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from .prompts import REFLECTOR_PROMPT, CURATOR_PROMPT
from .playbook_utils import extract_playbook_bullets, update_bullet_counts, get_playbook_stats, extract_json_from_text, apply_curator_operations, parse_playbook_line


def _get_context_updater_cfg_value(self_distillation_cfg, nested_key: str, legacy_key: str, default):
    if self_distillation_cfg is None:
        return default
    nested_cfg = self_distillation_cfg.get("context_updater", None)
    if nested_cfg is not None:
        return nested_cfg.get(nested_key, default)
    if hasattr(self_distillation_cfg, "get"):
        return self_distillation_cfg.get(legacy_key, default)
    return getattr(self_distillation_cfg, legacy_key, default)


def _remove_thinking_trace(text: str) -> str:
    # Case 1: complete <think>...</think> block in response
    out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Case 2: <think> was in the prompt, response starts with thinking content
    out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
    return out_text

def _check_prompt_lengths(prompts: List[str], tokenizer, max_prompt_length: int, label: str = "prompt") -> None:
    """Raise an error if any prompt exceeds max_prompt_length after chat-template tokenization."""
    for i, p in enumerate(prompts):
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(token_ids) > max_prompt_length:
            raise ValueError(
                f"[ACE] {label} {i} has {len(token_ids)} tokens which exceeds "
                f"prompt_length={max_prompt_length}. "
                f"Increase rollout.prompt_length or shorten the prompt template."
            )


def _extract_bullet_ids(response: str, use_json_mode: bool) -> List[str]:
    """
    Extract bullet IDs from generator response.
    
    Args:
        response: The generator's response
        use_json_mode: Whether JSON mode was used
        
    Returns:
        List of bullet IDs
    """
    response = _remove_thinking_trace(response)
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

def _extract_bullet_tags_regex(text: str) -> List[Dict[str, str]]:
    """
    Extract bullet tags using regex pattern matching.

    Args:
        text: Text to extract bullet tags from

    Returns:
        List of dicts with 'id' and 'tag' keys
    """
    bullet_tags = []
    id_pat = r'[a-z]{3,}-\d{5}'
    # Try {"id": "xxx-00001", "tag": "..."}
    matches = re.findall(
        r'"id"\s*:\s*"(' + id_pat + r')"\s*,\s*"tag"\s*:\s*"(\w+)"', text
    )
    for bullet_id, tag in matches:
        bullet_tags.append({"id": bullet_id, "tag": tag})
    if not bullet_tags:
        # Try reversed key order: {"tag": "...", "id": "xxx-00001"}
        matches = re.findall(
            r'"tag"\s*:\s*"(\w+)"\s*,\s*"id"\s*:\s*"(' + id_pat + r')"', text
        )
        for tag, bullet_id in matches:
            bullet_tags.append({"id": bullet_id, "tag": tag})
    return bullet_tags


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
    response = _remove_thinking_trace(response)
    bullet_tags = []

    if use_json_mode:
        try:
            response_json = extract_json_from_text(response)
            bullet_tags = response_json.get("bullet_tags", [])
        except (json.JSONDecodeError, KeyError, AttributeError):
            print(f"Warning: Failed to parse bullet tags from JSON response, falling back to regex")
            bullet_tags = _extract_bullet_tags_regex(response)
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
    response = _remove_thinking_trace(response)
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

        self.playbook = self.get_empty_playbook()
        self.next_global_id = 1
        self.context_update_count = 0

    def state_dict(self) -> dict:
        """Return serializable state for checkpoint saving."""
        return {
            "playbook": self.playbook,
            "next_global_id": self.next_global_id,
            "context_update_count": self.context_update_count,
        }

    def load_state_dict(self, state_dict: dict):
        """Restore state from a checkpoint."""
        self.playbook = state_dict["playbook"]
        self.next_global_id = state_dict["next_global_id"]
        self.context_update_count = state_dict["context_update_count"]
    
    @staticmethod
    def get_empty_playbook() -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    def _concise_playbook(
        self,
        playbook: str,
        max_bullets: int = None,
        concise_method: str = "reset",
    ) -> str:
        """
        Concise the playbook to at most max_bullets entries.

        Args:
            playbook: The current playbook content
            max_bullets: Target maximum number of bullets to keep.
                If None (e.g. triggered only by frequency), helpful bullets
                are not randomly truncated — only unused and harmful ones are removed.
            concise_method: Strategy to use for reduction:
                - "reset": discard everything and return a blank playbook (original behaviour)
                - "prioritized": remove unused → harmful → randomly truncate helpful
        Returns:
            Concised playbook content
        """
        if concise_method == "reset":
            return self.get_empty_playbook()

        # ------------------------------------------------------------------
        # "prioritized" method
        # ------------------------------------------------------------------

        lines = playbook.strip().split('\n')

        # Categorise every bullet
        unused_ids: set = set()
        harmful_ids: set = set()
        helpful_ids: list = []  # ordered list so random.sample is stable

        for line in lines:
            parsed = parse_playbook_line(line)
            if not parsed:
                continue
            bid = parsed['id']
            h, harm = parsed['helpful'], parsed['harmful']
            if h + harm == 0:
                unused_ids.add(bid)
            elif harm >= h and harm > 0:
                harmful_ids.add(bid)
            else:
                helpful_ids.append(bid)

        print(f"[ACE] Concising playbook (method={concise_method}) — "
              f"unused={len(unused_ids)}, harmful={len(harmful_ids)}, "
              f"helpful={len(helpful_ids)}, limit={max_bullets}")

        ids_to_remove: set = unused_ids | harmful_ids

        # Randomly truncate helpful bullets if still over the limit
        dropped_helpful: set = set()
        if max_bullets is not None:
            n_helpful_to_drop = max(0, len(helpful_ids) - max_bullets)
            if n_helpful_to_drop > 0:
                dropped_helpful = set(random.sample(helpful_ids, n_helpful_to_drop))
                ids_to_remove |= dropped_helpful
                print(f"[ACE] Randomly dropping {n_helpful_to_drop} helpful bullets "
                      f"to fit within limit of {max_bullets}")

        # Rebuild playbook, skipping removed bullets and collapsing extra blank lines
        new_lines = []
        prev_blank = False
        for line in lines:
            parsed = parse_playbook_line(line)
            if parsed and parsed['id'] in ids_to_remove:
                continue
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            new_lines.append(line)
            prev_blank = is_blank

        print(f"[ACE] Playbook concised: removed {len(ids_to_remove)} bullets "
              f"({len(unused_ids)} unused, {len(harmful_ids)} harmful, "
              f"{len(dropped_helpful)} helpful truncated)")

        return '\n'.join(new_lines)

    def update(self, batch: DataProto, async_rollout_manager, tokenizer, feedback_list: List[str], teacher_feedback_list: List[str] = None, acc_list: List[float] = None):
        batch_size = batch.batch["input_ids"].shape[0]
        print(f"[ACE] Starting context update for {batch_size} samples "
              f"(playbook bullets: {self.next_global_id - 1})")

        # Concise the playbook when triggered by frequency or bullet-count limit
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        concise_frequency = _get_context_updater_cfg_value(
            self_distillation_cfg,
            nested_key="concise_frequency",
            legacy_key="concise_frequency",
            default=None,
        )
        max_bullets = _get_context_updater_cfg_value(
            self_distillation_cfg,
            nested_key="max_bullets",
            legacy_key="max_bullets",
            default=None,
        )
        concise_method = _get_context_updater_cfg_value(
            self_distillation_cfg,
            nested_key="concise_method",
            legacy_key="concise_method",
            default="reset",
        )
        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length

        current_stats = get_playbook_stats(self.playbook)
        frequency_trigger = concise_frequency and self.context_update_count % concise_frequency == 0
        limit_trigger = max_bullets and current_stats['total_bullets'] > max_bullets

        if not concise_frequency and not max_bullets:
            print(f"[ACE] No concise_frequency or max_bullets set, skipping playbook concising")
        elif frequency_trigger or limit_trigger:
            reasons = []
            if frequency_trigger:
                reasons.append(f"frequency={concise_frequency}")
            if limit_trigger:
                reasons.append(f"bullets={current_stats['total_bullets']} > limit={max_bullets}")
            print(f"[ACE] Concising playbook (reason: {', '.join(reasons)}, method={concise_method})...")
            # Pass max_bullets only when the limit triggered — frequency-only runs skip helpful truncation
            effective_max = max_bullets if limit_trigger else None
            self.playbook = self._concise_playbook(self.playbook, effective_max, concise_method)

        # get response texts
        responses = batch.batch["responses"]
        response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        key = "raw_prompt_original" if "raw_prompt_original" in batch.non_tensor_batch else "raw_prompt"
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch[key]]

        # Determine which samples are incorrect and need reflection
        if acc_list is not None:
            incorrect_indices = [i for i, acc in enumerate(acc_list) if acc < 1.0]
        else:
            incorrect_indices = list(range(batch_size))
        num_correct = batch_size - len(incorrect_indices)
        print(f"[ACE] Skipping reflection on {num_correct}/{batch_size} correct samples")

        if not incorrect_indices:
            # All samples are correct, skip reflection and curation entirely
            self.context_update_count += 1
            final_stats = get_playbook_stats(self.playbook)
            print(f"[ACE] All samples correct, skipping context update")
            return {
                "response_texts": response_texts,
                "reflection_texts": [""] * batch_size,
                "final_stats": final_stats,
            }

        # Filter to only incorrect samples for reflection
        inc_prompt_texts = [prompt_texts[i] for i in incorrect_indices]
        inc_response_texts = [response_texts[i] for i in incorrect_indices]
        inc_feedback_list = [feedback_list[i] for i in incorrect_indices]
        inc_teacher_feedback_list = [(teacher_feedback_list[i] if teacher_feedback_list else "") for i in incorrect_indices]

        # prepare reflector prompts (only for incorrect samples)
        reflector_prompts = []
        for prompt, response, feedback, teacher_feedback in zip(inc_prompt_texts, inc_response_texts, inc_feedback_list, inc_teacher_feedback_list):
            reflector_prompt = REFLECTOR_PROMPT.format(
                prompt=prompt,
                response=_remove_thinking_trace(response),
                feedback=feedback or "",
                teacher_feedback=teacher_feedback or "",
                playbook=self.playbook,
            )
            reflector_prompts.append(reflector_prompt)

        # Build DataProto for the async server interface.
        # The agent loop (SingleTurnAgentLoop) expects raw_prompt: a list of message dicts.
        # It applies the chat template and tokenizes internally, so we skip manual tokenization.
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        reflector_messages = np.array(
            [[{"role": "user", "content": p}] for p in reflector_prompts], dtype=object
        )
        reflector_batch = DataProto.from_dict(
            non_tensors={"raw_prompt": reflector_messages},
        )
        reflector_batch.meta_info = {"validate": False}

        # Validate prompt lengths before generating
        _check_prompt_lengths(reflector_prompts, tokenizer, max_prompt_length, label="reflector prompt")

        # Pad to be divisible by num_workers, generate, then unpad
        reflector_batch_padded, pad_size = pad_dataproto_to_divisor(reflector_batch, num_workers)
        num_incorrect = len(incorrect_indices)
        print(f"[ACE] Generating reflections for {num_incorrect} incorrect samples...")
        reflector_output_padded = async_rollout_manager.generate_sequences(reflector_batch_padded)
        reflector_output = unpad_dataproto(reflector_output_padded, pad_size)

        # Decode reflections and extract bullet tags (only for incorrect samples)
        inc_reflection_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in reflector_output.batch["responses"]
        ]

        bullet_tags = [_extract_bullet_tags(response, use_json_mode=True) for response in inc_reflection_texts]

        for reflection, tags in zip(inc_reflection_texts, bullet_tags):
            if random.random() < 1/8:
                print(f"Reflection preview: {reflection}...")
                print(f"Extracted bullet tags: {tags}")

        # Sequentially update the bullet counts
        tags_updated = sum(1 for t in bullet_tags if t)
        print(f"[ACE] Updating bullet counts from {tags_updated}/{num_incorrect} reflections...")
        for tags in bullet_tags:
            if tags:
                self.playbook = update_bullet_counts(self.playbook, tags)

        # prepare curator prompts (only for incorrect samples)
        stats = get_playbook_stats(self.playbook)
        stats_str = json.dumps(stats, indent=2)
        curator_prompts = []
        for prompt, reflection in zip(inc_prompt_texts, inc_reflection_texts):
            curator_prompt = CURATOR_PROMPT.format(
                prompt=prompt,
                recent_reflection=_remove_thinking_trace(reflection),
                playbook_stats=stats_str,
                current_playbook=self.playbook
            )
            curator_prompts.append(curator_prompt)

        # Build DataProto for the curator via async server interface
        curator_messages = np.array(
            [[{"role": "user", "content": p}] for p in curator_prompts], dtype=object
        )
        curator_batch = DataProto.from_dict(
            non_tensors={"raw_prompt": curator_messages},
        )
        curator_batch.meta_info = {"validate": False}

        # Validate prompt lengths before generating
        _check_prompt_lengths(curator_prompts, tokenizer, max_prompt_length, label="curator prompt")

        # Pad to be divisible by num_workers, generate, then unpad
        curator_batch_padded, curator_pad_size = pad_dataproto_to_divisor(curator_batch, num_workers)
        print(f"[ACE] Generating curator operations for {num_incorrect} incorrect samples...")
        curator_output_padded = async_rollout_manager.generate_sequences(curator_batch_padded)
        curator_output = unpad_dataproto(curator_output_padded, curator_pad_size)

        # Decode
        curation_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in curator_output.batch["responses"]
        ]

        for curation in curation_texts:
            if random.random() < 1/8:
                print(f"Curator response preview: {curation}...")

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
        self.context_update_count += 1
        final_stats = get_playbook_stats(self.playbook)
        print(f"[ACE] Playbook updated: {len(all_operations)} operations applied, "
              f"total_bullets={final_stats['total_bullets']}, "
              f"high_performing={final_stats['high_performing']}, "
              f"problematic={final_stats['problematic']}, "
              f"unused={final_stats['unused']}")

        # Reconstruct full-batch reflection_texts (empty for correct samples)
        reflection_texts = [""] * batch_size
        for idx, inc_idx in enumerate(incorrect_indices):
            reflection_texts[inc_idx] = inc_reflection_texts[idx]

        return {
            "response_texts": response_texts,
            "reflection_texts": reflection_texts,
            "final_stats": final_stats
        }
