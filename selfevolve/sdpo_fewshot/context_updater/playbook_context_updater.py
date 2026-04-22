"""
PlaybookContextUpdater — unified context updater supporting both global and per-example playbook modes.

Modes (set via config ``playbook_mode``):
  - ``"global"``:      single shared playbook across all examples (mirrors ACEContextUpdater)
  - ``"per_example"``: each training example gets its own dedicated playbook, keyed by
                       the stable ``extra_info["index"]`` identifier
"""

import re
import json
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

from .playbook_utils import (
    apply_curator_operations,
    update_bullet_counts,
    get_playbook_stats,
    parse_playbook_line,
    extract_json_from_text
)

_GLOBAL_KEY = "__global__"


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


class PlaybookContextUpdater:
    """Context updater that supports both global and per-example playbook strategies."""

    def __init__(self, config):
        self.config = config

        # Resolve playbook mode from config
        sd_cfg = config.actor_rollout_ref.actor.get("self_distillation", None)
        self.playbook_mode: str = _get_context_updater_cfg_value(
            sd_cfg, nested_key="playbook_mode", legacy_key="playbook_mode", default="global",
        )
        assert self.playbook_mode in ("global", "per_example"), (
            f"Unknown playbook_mode: {self.playbook_mode!r}. Must be 'global' or 'per_example'."
        )

        ctx_cfg = sd_cfg.get("context_updater", None) if sd_cfg is not None else None
        self.reflector_prompt_template = self._resolve_prompt_template(
            ctx_cfg, "reflector_prompt_file", "reflector_prompt_template",
            __import__("selfevolve.sdpo_fewshot.context_updater.prompts", fromlist=["REFLECTOR_PROMPT"]).REFLECTOR_PROMPT,
        )
        self.curator_prompt_template = self._resolve_prompt_template(
            ctx_cfg, "curator_prompt_file", "curator_prompt_template",
            __import__("selfevolve.sdpo_fewshot.context_updater.prompts", fromlist=["CURATOR_PROMPT"]).CURATOR_PROMPT,
        )

        # Success tagging: run a lightweight reflector on correct samples to reinforce bullet counts
        self.tag_correct_samples: bool = _get_context_updater_cfg_value(
            sd_cfg, nested_key="tag_correct_samples", legacy_key="tag_correct_samples", default=False,
        )
        if self.tag_correct_samples:
            self.success_reflector_prompt_template = self._resolve_prompt_template(
                ctx_cfg, "success_reflector_prompt_file", "success_reflector_prompt_template",
                __import__("selfevolve.sdpo_fewshot.context_updater.prompts", fromlist=["SUCCESS_REFLECTOR_PROMPT"]).SUCCESS_REFLECTOR_PROMPT,
            )
            print(f"[PlaybookCU] Success tagging enabled — correct samples will reinforce bullet counts")

        # Playbook storage: Dict[playbook_key -> state_dict]
        self._playbooks: Dict[str, dict] = {}
        if self.playbook_mode == "global":
            self._playbooks[_GLOBAL_KEY] = self._make_fresh_state()

        # Deduplicate rollouts: when rollout.n > 1, only process one sample per
        # unique example_id in reflector/curator/success-tagging to avoid
        # redundant bullets from multiple rollouts of the same prompt.
        self.deduplicate_rollouts: bool = _get_context_updater_cfg_value(
            sd_cfg, nested_key="deduplicate_rollouts", legacy_key="deduplicate_rollouts", default=False,
        )
        if self.deduplicate_rollouts:
            print(f"[PlaybookCU] Rollout deduplication enabled — one sample per example_id for reflector/curator")

        # Solution buffer: persist successful response texts across steps.
        # Controlled by context_updater.use_solution_buffer (default: False).
        self.use_solution_buffer: bool = _get_context_updater_cfg_value(
            sd_cfg, nested_key="use_solution_buffer", legacy_key="use_solution_buffer", default=False,
        )
        self._solution_buffer: Dict[str, str] = {}
        if self.use_solution_buffer:
            print(f"[PlaybookCU] Solution buffer enabled — successful trials will be cached across steps")

        # Student playbook snapshot: a frozen copy synced at concise boundaries.
        self.use_playbook_in_student_rollout: bool = _get_context_updater_cfg_value(
            sd_cfg, nested_key="use_playbook_in_student_rollout",
            legacy_key="use_playbook_in_student_rollout", default=False,
        )
        self._student_playbooks: Dict[str, str] = {}
        concise_freq = _get_context_updater_cfg_value(
            sd_cfg, nested_key="concise_frequency", legacy_key="concise_frequency", default=4,
        )
        self._student_sync_frequency: int = _get_context_updater_cfg_value(
            sd_cfg, nested_key="student_playbook_sync_frequency",
            legacy_key="student_playbook_sync_frequency", default=None,
        ) or concise_freq or 4
        if self.use_playbook_in_student_rollout:
            print(f"[PlaybookCU] Student rollout playbook enabled — sync every {self._student_sync_frequency} updates")

        # Post-curation concise: run concise immediately after curator adds bullets
        # to keep total bullet count within max_bullets.
        self.concise_after_curation: bool = _get_context_updater_cfg_value(
            sd_cfg, nested_key="concise_after_curation", legacy_key="concise_after_curation", default=False,
        )
        if self.concise_after_curation:
            print(f"[PlaybookCU] Post-curation concise enabled — will trim after curator adds bullets")

        print(f"[PlaybookCU] Initialized with playbook_mode={self.playbook_mode!r}")
    
    @staticmethod
    def _resolve_prompt_template(ctx_cfg, file_key: str, template_key: str, default: str) -> str:
        """Resolve a prompt template with priority: file > inline template > built-in default."""
        if ctx_cfg is not None:
            file_path = getattr(ctx_cfg, file_key, None)
            if file_path:
                with open(file_path, "r") as f:
                    content = f.read()
                print(f"[ACE] Loaded {template_key} from file: {file_path}")
                return content
            inline = getattr(ctx_cfg, template_key, None)
            if inline:
                return inline
        return default
    
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

    # ------------------------------------------------------------------
    # Playbook key dispatch
    # ------------------------------------------------------------------

    def get_playbook_key(self, example_id: str) -> str:
        """Map an example_id to its playbook key.

        Override this method to implement custom grouping (e.g. per-cluster).
        """
        if self.playbook_mode == "global":
            return _GLOBAL_KEY
        return example_id

    # ------------------------------------------------------------------
    # Playbook access
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fresh_state() -> dict:
        return {
            "playbook": PlaybookContextUpdater.get_empty_playbook(),
            "next_global_id": 1,
            "update_count": 0,
            "bullet_last_used": {},  # bullet_id -> last step where it was tagged
        }

    def _get_or_create_state(self, key: str) -> dict:
        if key not in self._playbooks:
            self._playbooks[key] = self._make_fresh_state()
        return self._playbooks[key]

    def get_playbook(self, example_id: str) -> str:
        """Return the playbook string for a given example_id."""
        key = self.get_playbook_key(example_id)
        return self._get_or_create_state(key)["playbook"]

    # ------------------------------------------------------------------
    # Student playbook snapshot
    # ------------------------------------------------------------------

    def get_student_playbook(self, example_id: str) -> str:
        """Return the student playbook snapshot for a given example_id.

        This is a frozen copy that only updates at sync boundaries (see sync_student_playbooks).
        Returns the empty playbook if no snapshot has been synced yet.
        """
        key = self.get_playbook_key(example_id)
        return self._student_playbooks.get(key, self.get_empty_playbook())

    def should_sync_student(self) -> bool:
        """Check whether the student playbook snapshot should be synced this step.

        Uses the update_count of the first playbook key as the reference counter.
        Returns True when update_count > 0 and divisible by sync_frequency.
        """
        if not self._playbooks:
            return False
        # Use the first available key's update_count as the reference
        first_state = next(iter(self._playbooks.values()))
        uc = first_state.get("update_count", 0)
        return uc > 0 and uc % self._student_sync_frequency == 0

    def sync_student_playbooks(self) -> None:
        """Copy the current playbook to the student snapshot for all active keys."""
        for key, state in self._playbooks.items():
            self._student_playbooks[key] = state["playbook"]
        n_synced = len(self._student_playbooks)
        print(f"[PlaybookCU] Synced {n_synced} student playbook snapshot(s)")

    # ------------------------------------------------------------------
    # Solution buffer
    # ------------------------------------------------------------------

    def store_solution(self, example_id: str, solution_text: str) -> None:
        """Cache a successful response for *example_id* so future steps can use it."""
        if self.use_solution_buffer:
            self._solution_buffer[example_id] = solution_text

    def get_buffered_solution(self, example_id: str) -> Optional[str]:
        """Return a previously cached successful response, or None."""
        if self.use_solution_buffer:
            return self._solution_buffer.get(example_id, None)
        return None

    # ------------------------------------------------------------------
    # Backward-compatible .playbook property (used in global mode)
    # ------------------------------------------------------------------

    @property
    def playbook(self) -> str:
        """Convenience property for global mode — returns the single playbook."""
        if self.playbook_mode == "global":
            return self._playbooks[_GLOBAL_KEY]["playbook"]
        raise AttributeError(
            "PlaybookContextUpdater.playbook is only available in 'global' mode. "
            "Use get_playbook(example_id) for per-example mode."
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d = {
            "playbook_mode": self.playbook_mode,
            "playbooks": self._playbooks,
        }
        if self.use_solution_buffer:
            d["solution_buffer"] = self._solution_buffer
        if self.use_playbook_in_student_rollout:
            d["student_playbooks"] = self._student_playbooks
        return d

    def load_state_dict(self, state_dict: dict):
        saved_mode = state_dict.get("playbook_mode", "global")
        if saved_mode != self.playbook_mode:
            print(
                f"[PlaybookCU] WARNING: checkpoint playbook_mode={saved_mode!r} "
                f"differs from current={self.playbook_mode!r}. Loading anyway."
            )
        self._playbooks = state_dict["playbooks"]
        # Backfill bullet_last_used for old checkpoints that lack it
        for state in self._playbooks.values():
            if "bullet_last_used" not in state:
                state["bullet_last_used"] = {}
        if self.use_solution_buffer:
            self._solution_buffer = state_dict.get("solution_buffer", {})
            print(f"[PlaybookCU] Restored {len(self._playbooks)} playbook(s) and "
                  f"{len(self._solution_buffer)} buffered solution(s) from checkpoint.")
        else:
            print(f"[PlaybookCU] Restored {len(self._playbooks)} playbook(s) from checkpoint.")
        if self.use_playbook_in_student_rollout:
            self._student_playbooks = state_dict.get("student_playbooks", {})
            print(f"[PlaybookCU] Restored {len(self._student_playbooks)} student playbook snapshot(s) from checkpoint.")

    # ------------------------------------------------------------------
    # Concising (per playbook key)
    # ------------------------------------------------------------------

    def _maybe_concise(self, key: str, concise_frequency, max_bullets, concise_method: str):
        """Concise a single playbook if triggered by frequency or bullet-count limit."""
        state = self._playbooks[key]
        pb = state["playbook"]
        uc = state["update_count"]

        current_stats = get_playbook_stats(pb)
        frequency_trigger = concise_frequency and uc > 0 and uc % concise_frequency == 0
        limit_trigger = max_bullets and current_stats["total_bullets"] > max_bullets

        if not concise_frequency and not max_bullets:
            return  # nothing configured
        if not (frequency_trigger or limit_trigger):
            return

        reasons = []
        if frequency_trigger:
            reasons.append(f"frequency={concise_frequency}")
        if limit_trigger:
            reasons.append(f"bullets={current_stats['total_bullets']} > limit={max_bullets}")
        print(f"[PlaybookCU] Concising playbook key={key!r} (reason: {', '.join(reasons)}, method={concise_method})...")

        effective_max = max_bullets if limit_trigger else None
        state["playbook"] = self._concise_playbook(
            pb, effective_max, concise_method,
            bullet_last_used=state.get("bullet_last_used", {}),
            current_step=uc,
        )
        # Clean up bullet_last_used for removed bullets
        remaining_ids = {
            parsed["id"]
            for line in state["playbook"].strip().split("\n")
            if (parsed := parse_playbook_line(line))
        }
        state["bullet_last_used"] = {
            bid: step for bid, step in state["bullet_last_used"].items() if bid in remaining_ids
        }

    @staticmethod
    def _concise_playbook(
        playbook: str,
        max_bullets: Optional[int],
        concise_method: str,
        bullet_last_used: Optional[Dict[str, int]] = None,
        current_step: Optional[int] = None,
    ) -> str:
        """Concise a playbook string.

        concise_method:
          - "reset": wipe and return an empty playbook.
          - "prioritized": remove unused and harmful bullets;
            if still over max_bullets, randomly drop helpful ones.
          - "staleness": remove all harmful and stale unused bullets first
            (stale unused = unused for more than 1 step). If max_bullets is
            set and still over the cap, remove the longest-unused
            non-harmful bullets until satisfied. If max_bullets is not set,
            just remove harmful + stale unused.
        """
        if concise_method == "reset":
            return PlaybookContextUpdater.get_empty_playbook()

        lines = playbook.strip().split("\n")
        unused_ids: set = set()
        harmful_ids: set = set()
        helpful_ids: list = []

        for line in lines:
            parsed = parse_playbook_line(line)
            if not parsed:
                continue
            bid = parsed["id"]
            h, harm = parsed["helpful"], parsed["harmful"]
            if h + harm == 0:
                unused_ids.add(bid)
            elif harm >= h and harm > 0:
                harmful_ids.add(bid)
            else:
                helpful_ids.append(bid)

        if concise_method == "staleness":
            last_used = bullet_last_used or {}
            step = current_step or 0
            # Stale unused: unused for more than 1 step
            stale_unused = {bid for bid in unused_ids if step - last_used.get(bid, 0) > 1}
            ids_to_remove: set = harmful_ids | stale_unused
            print(f"[PlaybookCU] Staleness concise: removing {len(harmful_ids)} harmful + "
                  f"{len(stale_unused)} stale unused bullets")

            if max_bullets is not None:
                surviving_unused = [bid for bid in unused_ids if bid not in ids_to_remove]
                total_remaining = len(helpful_ids) + len(surviving_unused)
                n_over = total_remaining - max_bullets
                if n_over > 0:
                    all_remaining = helpful_ids + surviving_unused
                    random.shuffle(all_remaining)
                    sorted_by_staleness = sorted(
                        all_remaining,
                        key=lambda bid: last_used.get(bid, 0),
                    )
                    ids_to_remove |= set(sorted_by_staleness[:n_over])
                    print(f"[PlaybookCU] Staleness: additionally removing {min(n_over, len(sorted_by_staleness))} "
                          f"longest-unused bullets to meet max_bullets={max_bullets}")
        else:
            # "prioritized" method
            ids_to_remove = unused_ids | harmful_ids

            if max_bullets is not None:
                n_drop = max(0, len(helpful_ids) - max_bullets)
                if n_drop > 0:
                    ids_to_remove |= set(random.sample(helpful_ids, n_drop))

        new_lines = []
        prev_blank = False
        for line in lines:
            parsed = parse_playbook_line(line)
            if parsed and parsed["id"] in ids_to_remove:
                continue
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            new_lines.append(line)
            prev_blank = is_blank

        return "\n".join(new_lines)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        batch: DataProto,
        async_rollout_manager,
        tokenizer,
        feedback_list: List[str],
        teacher_feedback_list: List[str] = None,
        acc_list: List[float] = None,
        example_ids: List[str] = None,
    ):
        """Run reflection + curation to update playbook(s).

        In global mode the behavior is identical to ACEContextUpdater.update().
        In per-example mode each example's playbook is updated independently.

        Args:
            example_ids: Stable identifiers for each sample in the batch
                (from ``extra_info["index"]``). Required for per-example mode.
        """
        batch_size = batch.batch["input_ids"].shape[0]

        if self.playbook_mode == "per_example":
            assert example_ids is not None, (
                "example_ids must be provided for per_example playbook_mode"
            )
            assert len(example_ids) == batch_size

        # In global mode, all examples share one key
        if example_ids is None:
            example_ids = [_GLOBAL_KEY] * batch_size
        playbook_keys = [self.get_playbook_key(eid) for eid in example_ids]

        # Ensure playbook states exist for all keys in this batch
        unique_keys = set(playbook_keys)
        for k in unique_keys:
            self._get_or_create_state(k)

        # Read config values
        sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        concise_frequency = _get_context_updater_cfg_value(sd_cfg, "concise_frequency", "concise_frequency", None)
        max_bullets = _get_context_updater_cfg_value(sd_cfg, "max_bullets", "max_bullets", None)
        concise_method = _get_context_updater_cfg_value(sd_cfg, "concise_method", "concise_method", "reset")
        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length

        # Maybe concise each active playbook
        for k in unique_keys:
            self._maybe_concise(k, concise_frequency, max_bullets, concise_method)

        stats = self._aggregate_stats()
        print(f"[PlaybookCU] Starting context update for {batch_size} samples, "
              f"mode={self.playbook_mode}, active_playbooks={stats['num_playbooks']}, "
              f"avg_bullets={stats['avg_bullets']:.1f}")

        # Decode responses and prompts
        responses = batch.batch["responses"]
        response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        key = "raw_prompt_original" if "raw_prompt_original" in batch.non_tensor_batch else "raw_prompt"
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch[key]]

        # Determine incorrect and correct samples
        if acc_list is not None:
            incorrect_indices = [i for i, acc in enumerate(acc_list) if acc < 1.0]
            correct_indices = [i for i, acc in enumerate(acc_list) if acc >= 1.0]
        else:
            incorrect_indices = list(range(batch_size))
            correct_indices = []
        num_correct = len(correct_indices)
        print(f"[PlaybookCU] {num_correct}/{batch_size} correct samples"
              f"{' (success tagging enabled)' if self.tag_correct_samples and num_correct > 0 else ''}")

        if not incorrect_indices:
            # Run success tagging before returning if enabled
            if self.tag_correct_samples and correct_indices:
                self._run_success_tagging(
                    correct_indices, prompt_texts, response_texts, playbook_keys,
                    async_rollout_manager, tokenizer, max_prompt_length,
                )
            for k in unique_keys:
                self._playbooks[k]["update_count"] += 1
            final_stats = self._aggregate_stats(unique_keys)
            print(f"[PlaybookCU] All samples correct, skipping error reflection")
            return {
                "response_texts": response_texts,
                "reflection_texts": [""] * batch_size,
                "final_stats": final_stats,
            }

        # Filter to incorrect samples
        inc_prompt_texts = [prompt_texts[i] for i in incorrect_indices]
        inc_response_texts = [response_texts[i] for i in incorrect_indices]
        inc_feedback_list = [feedback_list[i] for i in incorrect_indices]
        inc_teacher_feedback_list = [
            (teacher_feedback_list[i] if teacher_feedback_list else "") for i in incorrect_indices
        ]
        inc_playbook_keys = [playbook_keys[i] for i in incorrect_indices]

        # ----- Reflector pass -----
        reflector_prompts = []
        for j, (prompt, response, feedback, teacher_feedback) in enumerate(
            zip(inc_prompt_texts, inc_response_texts, inc_feedback_list, inc_teacher_feedback_list)
        ):
            pb = self._playbooks[inc_playbook_keys[j]]["playbook"]
            reflector_prompt = self.reflector_prompt_template.format(
                prompt=prompt,
                response=_remove_thinking_trace(response),
                feedback=feedback or "",
                teacher_feedback=teacher_feedback or "",
                playbook=pb,
            )
            reflector_prompts.append(reflector_prompt)

        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        reflector_messages = np.array(
            [[{"role": "user", "content": p}] for p in reflector_prompts], dtype=object
        )
        reflector_batch = DataProto.from_dict(non_tensors={"raw_prompt": reflector_messages})
        reflector_batch.meta_info = {"validate": False}

        _check_prompt_lengths(reflector_prompts, tokenizer, max_prompt_length, label="reflector prompt")

        reflector_batch_padded, pad_size = pad_dataproto_to_divisor(reflector_batch, num_workers)
        num_incorrect = len(incorrect_indices)
        print(f"[PlaybookCU] Generating reflections for {num_incorrect} incorrect samples...")
        reflector_output_padded = async_rollout_manager.generate_sequences(reflector_batch_padded)
        reflector_output = unpad_dataproto(reflector_output_padded, pad_size)

        inc_reflection_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in reflector_output.batch["responses"]
        ]

        # Extract bullet tags and update counts per playbook key
        bullet_tags = [_extract_bullet_tags(r, use_json_mode=True) for r in inc_reflection_texts]
        for reflection, tags in zip(inc_reflection_texts, bullet_tags):
            if random.random() < 1 / 8:
                print(f"Reflection preview: {reflection}...")
                print(f"Extracted bullet tags: {tags}")

        # Group bullet tag updates by playbook key
        tags_by_key: Dict[str, list] = defaultdict(list)
        for j, tags in enumerate(bullet_tags):
            if tags:
                tags_by_key[inc_playbook_keys[j]].append(tags)

        for pk, tag_lists in tags_by_key.items():
            for tags in tag_lists:
                self._update_bullet_counts_and_staleness(pk, tags)

        # ----- Success tagging pass (correct samples) -----
        # Run before curator so that updated helpful counts are visible in {current_playbook}
        if self.tag_correct_samples and correct_indices:
            self._run_success_tagging(
                correct_indices, prompt_texts, response_texts, playbook_keys,
                async_rollout_manager, tokenizer, max_prompt_length,
            )

        # ----- Curator pass -----
        curator_prompts = []
        for j, (prompt, reflection) in enumerate(zip(inc_prompt_texts, inc_reflection_texts)):
            pk = inc_playbook_keys[j]
            pb = self._playbooks[pk]["playbook"]
            stats = get_playbook_stats(pb)
            stats_str = json.dumps(stats, indent=2)
            curator_prompt = self.curator_prompt_template.format(
                prompt=prompt,
                recent_reflection=_remove_thinking_trace(reflection),
                playbook_stats=stats_str,
                current_playbook=pb,
            )
            curator_prompts.append(curator_prompt)

        curator_messages = np.array(
            [[{"role": "user", "content": p}] for p in curator_prompts], dtype=object
        )
        curator_batch = DataProto.from_dict(non_tensors={"raw_prompt": curator_messages})
        curator_batch.meta_info = {"validate": False}

        _check_prompt_lengths(curator_prompts, tokenizer, max_prompt_length, label="curator prompt")

        curator_batch_padded, curator_pad_size = pad_dataproto_to_divisor(curator_batch, num_workers)
        print(f"[PlaybookCU] Generating curator operations for {num_incorrect} incorrect samples...")
        curator_output_padded = async_rollout_manager.generate_sequences(curator_batch_padded)
        curator_output = unpad_dataproto(curator_output_padded, curator_pad_size)

        curation_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in curator_output.batch["responses"]
        ]

        for curation in curation_texts:
            if random.random() < 1 / 8:
                print(f"Curator response preview: {curation}...")

        # Group operations by playbook key and apply
        ops_by_key: Dict[str, list] = defaultdict(list)
        for j, curation in enumerate(curation_texts):
            if curation.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
                print(f"⏭️  Skipping curator operation due to empty response")
                continue
            try:
                operations_info = _extract_and_validate_operations(curation)
                operations = operations_info["operations"]
                ops_by_key[inc_playbook_keys[j]].extend(operations)
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

        total_ops = 0
        for pk, ops in ops_by_key.items():
            if ops:
                state = self._playbooks[pk]
                state["playbook"], state["next_global_id"] = apply_curator_operations(
                    state["playbook"], ops, state["next_global_id"]
                )
                # Record creation step for any newly added bullets
                for line in state["playbook"].strip().split("\n"):
                    parsed = parse_playbook_line(line)
                    if parsed and parsed["id"] not in state["bullet_last_used"]:
                        state["bullet_last_used"][parsed["id"]] = state["update_count"]
                total_ops += len(ops)

        # Post-curation concise: trim playbooks that exceeded max_bullets after curator additions
        if self.concise_after_curation and max_bullets:
            for pk in ops_by_key:
                state = self._playbooks[pk]
                post_stats = get_playbook_stats(state["playbook"])
                if post_stats["total_bullets"] > max_bullets:
                    print(f"[PlaybookCU] Post-curation concise key={pk!r}: "
                          f"{post_stats['total_bullets']} bullets > limit={max_bullets}")
                    state["playbook"] = self._concise_playbook(
                        state["playbook"], max_bullets, concise_method,
                        bullet_last_used=state.get("bullet_last_used", {}),
                        current_step=state["update_count"],
                    )
                    remaining_ids = {
                        parsed["id"]
                        for line in state["playbook"].strip().split("\n")
                        if (parsed := parse_playbook_line(line))
                    }
                    state["bullet_last_used"] = {
                        bid: step for bid, step in state["bullet_last_used"].items() if bid in remaining_ids
                    }

        # Increment update counts for all active keys
        for k in unique_keys:
            self._playbooks[k]["update_count"] += 1

        final_stats = self._aggregate_stats(unique_keys)
        print(f"[PlaybookCU] Update complete: {total_ops} operations applied across "
              f"{len(ops_by_key)} playbook(s), active_playbooks={len(self._playbooks)}, "
              f"avg_bullets={final_stats['avg_bullets']:.1f}")

        if len(self._playbooks) > 10000:
            print(f"[PlaybookCU] WARNING: {len(self._playbooks)} playbooks stored — "
                  f"consider enabling per-example concising or reducing dataset size.")

        # Reconstruct full-batch reflection_texts
        reflection_texts = [""] * batch_size
        for idx, inc_idx in enumerate(incorrect_indices):
            reflection_texts[inc_idx] = inc_reflection_texts[idx]

        return {
            "response_texts": response_texts,
            "reflection_texts": reflection_texts,
            "final_stats": final_stats,
        }

    # ------------------------------------------------------------------
    # Success tagging
    # ------------------------------------------------------------------

    def _run_success_tagging(
        self,
        correct_indices: List[int],
        prompt_texts: List[str],
        response_texts: List[str],
        playbook_keys: List[str],
        async_rollout_manager,
        tokenizer,
        max_prompt_length: int,
    ):
        """Run a lightweight reflector on correct samples to tag which bullets helped."""
        cor_prompt_texts = [prompt_texts[i] for i in correct_indices]
        cor_response_texts = [response_texts[i] for i in correct_indices]
        cor_playbook_keys = [playbook_keys[i] for i in correct_indices]

        success_prompts = []
        for j, (prompt, response) in enumerate(zip(cor_prompt_texts, cor_response_texts)):
            pb = self._playbooks[cor_playbook_keys[j]]["playbook"]
            success_prompt = self.success_reflector_prompt_template.format(
                prompt=prompt,
                response=_remove_thinking_trace(response),
                playbook=pb,
            )
            success_prompts.append(success_prompt)

        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        success_messages = np.array(
            [[{"role": "user", "content": p}] for p in success_prompts], dtype=object
        )
        success_batch = DataProto.from_dict(non_tensors={"raw_prompt": success_messages})
        success_batch.meta_info = {"validate": False}

        _check_prompt_lengths(success_prompts, tokenizer, max_prompt_length, label="success reflector prompt")

        success_batch_padded, pad_size = pad_dataproto_to_divisor(success_batch, num_workers)
        num_correct = len(correct_indices)
        print(f"[PlaybookCU] Running success tagging for {num_correct} correct samples...")
        success_output_padded = async_rollout_manager.generate_sequences(success_batch_padded)
        success_output = unpad_dataproto(success_output_padded, pad_size)

        success_reflection_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in success_output.batch["responses"]
        ]

        # Extract bullet tags and update counts
        success_bullet_tags = [_extract_bullet_tags(r, use_json_mode=True) for r in success_reflection_texts]
        for reflection, tags in zip(success_reflection_texts, success_bullet_tags):
            if random.random() < 1 / 8:
                print(f"[PlaybookCU] Success tagging preview: {reflection}...")
                print(f"[PlaybookCU] Success bullet tags: {tags}")

        tags_by_key: Dict[str, list] = defaultdict(list)
        for j, tags in enumerate(success_bullet_tags):
            if tags:
                tags_by_key[cor_playbook_keys[j]].append(tags)

        total_tagged = 0
        for pk, tag_lists in tags_by_key.items():
            for tags in tag_lists:
                self._update_bullet_counts_and_staleness(pk, tags)
                total_tagged += len(tags)

        print(f"[PlaybookCU] Success tagging complete: {total_tagged} bullet tags from {num_correct} correct samples")

    # ------------------------------------------------------------------
    # Bullet staleness tracking
    # ------------------------------------------------------------------

    def _update_bullet_counts_and_staleness(self, playbook_key: str, tags: list):
        """Update bullet helpful/harmful counts and record last-used step."""
        state = self._playbooks[playbook_key]
        state["playbook"] = update_bullet_counts(state["playbook"], tags)
        current_step = state["update_count"]
        for tag in tags:
            if isinstance(tag, dict):
                bullet_id = tag.get("id") or tag.get("bullet", "")
                tag_value = tag.get("tag", "neutral")
                if bullet_id and tag_value in ("helpful", "harmful"):
                    state["bullet_last_used"][bullet_id] = current_step

    def _record_new_bullet(self, playbook_key: str, bullet_id: str):
        """Record the creation step for a newly added bullet."""
        state = self._playbooks[playbook_key]
        state["bullet_last_used"][bullet_id] = state["update_count"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_stats(self, keys=None) -> dict:
        """Aggregate playbook stats across all (or given) playbook keys.

        Returns both totals and per-playbook averages for meaningful logging.
        """
        if keys is None:
            keys = self._playbooks.keys()
        per_pb_bullets = []
        total_high_performing = 0
        total_problematic = 0
        total_unused = 0
        for k in keys:
            if k not in self._playbooks:
                continue
            s = get_playbook_stats(self._playbooks[k]["playbook"])
            per_pb_bullets.append(s["total_bullets"])
            total_high_performing += s["high_performing"]
            total_problematic += s["problematic"]
            total_unused += s["unused"]
        n = len(per_pb_bullets) or 1
        return {
            "num_playbooks": len(per_pb_bullets),
            "total_bullets": sum(per_pb_bullets),
            "avg_bullets": sum(per_pb_bullets) / n,
            "max_bullets_single": max(per_pb_bullets) if per_pb_bullets else 0,
            "min_bullets_single": min(per_pb_bullets) if per_pb_bullets else 0,
            "high_performing": total_high_performing,
            "problematic": total_problematic,
            "unused": total_unused,
        }

    def get_summary(self) -> str:
        """Return a human-readable summary for logging."""
        stats = self._aggregate_stats()
        lines = [
            f"PlaybookContextUpdater (mode={self.playbook_mode})",
            f"  Total playbooks:   {stats['num_playbooks']}",
            f"  Total bullets:     {stats['total_bullets']}",
            f"  Avg bullets/pb:    {stats['avg_bullets']:.1f}",
            f"  Max bullets (one): {stats['max_bullets_single']}",
            f"  Min bullets (one): {stats['min_bullets_single']}",
            f"  High performing:   {stats['high_performing']}",
            f"  Problematic:       {stats['problematic']}",
            f"  Unused:            {stats['unused']}",
        ]
        if self.playbook_mode == "per_example":
            sizes = [(k, get_playbook_stats(s["playbook"])["total_bullets"]) for k, s in self._playbooks.items()]
            sizes.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"  Top playbooks by bullet count:")
            for k, n in sizes[:5]:
                lines.append(f"    {k}: {n} bullets")
        return "\n".join(lines)
