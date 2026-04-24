# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import re
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from ...context_updater import ACEContextUpdater, PlaybookContextUpdater
from ...context_updater.prompts import TEACHER_PROMPT as _DEFAULT_TEACHER_PROMPT
from ...context_updater.prompts import STUDENT_PROMPT as _DEFAULT_STUDENT_PROMPT
from ...teacher import TeacherClient


def _get_context_updater_cfg_value(self_distillation_cfg, nested_key: str, legacy_key: str, default):
    if self_distillation_cfg is None:
        return default
    nested_cfg = self_distillation_cfg.get("context_updater", None)
    if nested_cfg is not None:
        return nested_cfg.get(nested_key, default)
    return self_distillation_cfg.get(legacy_key, default)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        assert not self.config.reward_model.launch_reward_fn_async, "Asynchronous reward function is currently not supported in RayPPOTrainer."

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ACE context updater (off by default — enabled via self_distillation.context_updater.enabled)
        sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        self.use_context_updater = _get_context_updater_cfg_value(
            sd_cfg,
            nested_key="enabled",
            legacy_key="use_context_updater",
            default=False,
        )
        self.context_updater = PlaybookContextUpdater(config) if self.use_context_updater else None
        if self.use_context_updater:
            ctx_cfg = config.actor_rollout_ref.actor.get("self_distillation", {}).get("context_updater", None)
            self.cu_teacher_prompt_template = PlaybookContextUpdater._resolve_prompt_template(
                ctx_cfg, "cu_teacher_prompt_file", "cu_teacher_prompt_template", _DEFAULT_TEACHER_PROMPT,
            )
            # Student rollout playbook: inject playbook snapshot into student prompt
            self.use_playbook_in_student_rollout = _get_context_updater_cfg_value(
                sd_cfg, nested_key="use_playbook_in_student_rollout",
                legacy_key="use_playbook_in_student_rollout", default=False,
            )
            self.student_prompt_template = PlaybookContextUpdater._resolve_prompt_template(
                ctx_cfg, "student_prompt_file", "student_prompt_template", _DEFAULT_STUDENT_PROMPT,
            )
        else:
            self.cu_teacher_prompt_template = _DEFAULT_TEACHER_PROMPT
            self.use_playbook_in_student_rollout = False
            self.student_prompt_template = _DEFAULT_STUDENT_PROMPT

        _teacher_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", {}).get("teacher", None)
        if _teacher_cfg and _teacher_cfg.get("enabled", False):
            self.teacher_client = TeacherClient(
                server_ip=_teacher_cfg.server_ip,
                server_port=_teacher_cfg.server_port,
                n_server_workers=_teacher_cfg.n_server_workers,
                max_tokens=_teacher_cfg.max_tokens,
                temperature=_teacher_cfg.temperature,
                only_response=True,
            )
            self.n_teacher_workers = _teacher_cfg.n_server_workers
        else:
            self.teacher_client = None
            self.n_teacher_workers = 0

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _log_reprompt_data(
        self, batch: DataProto, timing_raw: dict, reprompt_data_dir: str
    ):
        """Log reprompt texts to disk as JSONL for all samples.

        Token-level log probs and distillation loss statistics are already
        dumped by dp_actor.py (controlled by ``token_loss_dump_n``).

        Args:
            batch: The batch containing reprompt data (must have teacher_input_ids).
            timing_raw: Timing information for profiling.
            reprompt_data_dir: Directory path to save the reprompt data.
        """
        if "teacher_input_ids" not in batch.batch:
            return

        with marked_timer("dump_reprompt_data", timing_raw, color="green"):
            teacher_input_ids = batch.batch["teacher_input_ids"]
            responses = batch.batch["responses"]
            reprompt_len = teacher_input_ids.shape[1] - responses.shape[1]
            reprompt_ids = teacher_input_ids[:, :reprompt_len]

            reprompt_texts = self.tokenizer.batch_decode(reprompt_ids, skip_special_tokens=True)
            prompt_texts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sd_mask = batch.batch["self_distillation_mask"].cpu().tolist()

            os.makedirs(reprompt_data_dir, exist_ok=True)
            filename = os.path.join(reprompt_data_dir, f"{self.global_steps}.jsonl")

            lines = []
            n = len(reprompt_texts)
            for i in range(n):
                entry = {
                    "reprompt": reprompt_texts[i],
                    "prompt": prompt_texts[i],
                    "response": response_texts[i],
                    "score": scores[i],
                    "self_distillation_mask": sd_mask[i],
                    "step": self.global_steps,
                }
                if "uid" in batch.non_tensor_batch:
                    entry["uid"] = str(batch.non_tensor_batch["uid"][i])
                lines.append(json.dumps(entry, ensure_ascii=False))

            with open(filename, "w") as f:
                f.write("\n".join(lines) + "\n")

            print(f"Dumped reprompt data to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _log_playbook(self, playbook: str, step: int):
        """Log the current playbook text to the configured logger (wandb only)."""
        if "wandb" in self.config.trainer.logger:
            import wandb
            if wandb.run is not None:
                wandb.log({"playbook/text": wandb.Html(f"<pre>{playbook}</pre>")}, step=step)

    def _log_playbook_summary(self, context_updater: "PlaybookContextUpdater", step: int):
        """Log playbook summary stats for PlaybookContextUpdater (supports both modes)."""
        stats = context_updater._aggregate_stats()
        if "wandb" in self.config.trainer.logger:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "playbook/num_playbooks": stats["num_playbooks"],
                    "playbook/total_bullets": stats["total_bullets"],
                    "playbook/avg_bullets": stats["avg_bullets"],
                    "playbook/max_bullets_single": stats["max_bullets_single"],
                    "playbook/high_performing": stats["high_performing"],
                    "playbook/problematic": stats["problematic"],
                    "playbook/unused": stats["unused"],
                }, step=step)
                # In global mode, also log the full playbook text
                if context_updater.playbook_mode == "global":
                    wandb.log({"playbook/text": wandb.Html(f"<pre>{context_updater.playbook}</pre>")}, step=step)

    def _log_teacher_prompt(self, messages: list, step: int, sample_idx: int = 0):
        """Log a sample teacher prompt to wandb under self_distillation/teacher_prompt."""
        if "wandb" in self.config.trainer.logger:
            import wandb
            if wandb.run is not None:
                sample = self.tokenizer.apply_chat_template(
                    messages[sample_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                wandb.log({"self_distillation/teacher_prompt": wandb.Html(f"<pre>{sample}</pre>")}, step=step)

    def _log_teacher_feedback(self, batch: DataProto, teacher_feedback_list: list, step: int,
                              env_feedback_list: list | None = None):
        """Log one teacher feedback sample to wandb for debugging."""
        if "wandb" not in self.config.trainer.logger:
            return
        import wandb
        if wandb.run is None:
            return

        teacher_cfg = self.config.actor_rollout_ref.actor.self_distillation.teacher
        batch_size = batch.batch.batch_size[0]
        prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        env_fb_list = env_feedback_list or [None] * batch_size

        for i in range(batch_size):
            if teacher_feedback_list[i] is not None:
                env_fb = env_fb_list[i] or ""
                feedback_prompt = teacher_cfg.feedback_prompt_template.format(
                    prompt=prompts[i], response=responses[i], feedback=env_fb,
                )
                wandb.log({
                    "teacher/feedback_prompt": wandb.Html(f"<pre>{feedback_prompt}</pre>"),
                    "teacher/feedback_response": wandb.Html(f"<pre>{teacher_feedback_list[i]}</pre>"),
                }, step=step)
                break

    def _compute_reward_legacy(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores, so it will not fail into this function.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        # Populate extra_info["truncated"] for each sample so reward functions can use it.
        # A response is truncated if it contains no EOS token in its valid (non-padding) portion.
        eos_token_id = self.tokenizer.eos_token_id
        for i in range(len(batch)):
            data_item = batch[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = batch.non_tensor_batch.get("extra_info", None)
            if extra_info is not None and i < len(extra_info):
                if not isinstance(extra_info[i], dict):
                    extra_info[i] = {}
                extra_info[i]["truncated"] = not (valid_response_ids == eos_token_id).any().item()

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid
    
    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        # Case 1: complete <think>...</think> block in response
        out_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        # Case 2: <think> was in the prompt, response starts with thinking content
        out_text = re.sub(r'^.*?</think>\s*', '', out_text, flags=re.DOTALL)
        return out_text
    
    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
        buffered_solution: Optional[str] = None,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        self_is_successful = False
        if dont_reprompt_on_self_success:
            self_is_successful = idx in solution_idxs
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            # Fall back to buffered solution from a previous step,
            # but not if the current rollout is already correct
            if buffered_solution is not None and not self_is_successful:
                if remove_thinking_from_demonstration:
                    buffered_solution = self._remove_thinking_trace(buffered_solution)
                return buffered_solution
            return None
        solution_idx = solution_idxs[0]  # taking the first successful demonstration effectively selects a random one
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str

    def _submit_teacher_feedback(
        self,
        batch: DataProto,
        env_feedback_list: list | None = None,
        acc_list: list | None = None,
    ) -> tuple:
        """Non-blocking submit of teacher feedback requests."""
        teacher_cfg = self.config.actor_rollout_ref.actor.self_distillation.teacher
        batch_size = batch.batch.batch_size[0]

        feedback_on_correct = teacher_cfg.get("feedback_on_correct", False)
        if acc_list is not None and not feedback_on_correct:
            request_indices = [i for i in range(batch_size) if not acc_list[i]]
        else:
            request_indices = list(range(batch_size))

        if not request_indices:
            return [], [], 1

        prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        env_fb = env_feedback_list or [None] * batch_size

        tmpl = teacher_cfg.feedback_prompt_template
        request_texts = [
            tmpl.format(prompt=prompts[i], response=responses[i], feedback=env_fb[i] or "")
            for i in request_indices
        ]

        encoded = self.tokenizer(
            request_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=teacher_cfg.get("max_feedback_prompt_len", 4096),
        )
        input_ids = encoded["input_ids"]
        attn = encoded["attention_mask"].bool()
        n_submit = len(request_indices)
        ids_list = [input_ids[i][attn[i]].tolist() for i in range(n_submit)]

        mbs = max(1, n_submit // self.n_teacher_workers)
        futures = []
        for start in range(0, n_submit, mbs):
            futures.append(self.teacher_client.submit(ids_list[start : start + mbs]))

        return request_indices, futures, mbs

    def _resolve_teacher_feedback(
        self,
        teacher_futures: tuple,
        batch_size: int,
    ) -> list:
        """Blocking wait for teacher feedback futures and decode responses."""
        request_indices, futures, mbs = teacher_futures
        result = [None] * batch_size
        n_failed = 0
        n_succeeded = 0

        for fut_idx, future in enumerate(futures):
            start = fut_idx * mbs
            try:
                teacher_responses, _, _ = future.result()
                for j, resp_ids in enumerate(teacher_responses):
                    batch_idx = request_indices[start + j]
                    token_ids = resp_ids.tolist() if hasattr(resp_ids, "tolist") else resp_ids
                    result[batch_idx] = self._remove_thinking_trace(self.tokenizer.decode(token_ids, skip_special_tokens=True))
                    n_succeeded += 1
            except Exception as e:
                n_batch_in_future = min(mbs, len(request_indices) - start)
                n_failed += n_batch_in_future
                print(f"[teacher_feedback] future {fut_idx} failed ({n_batch_in_future} samples lost): {e}")

        n_requested = len(request_indices)
        if n_requested > 0:
            failure_rate = n_failed / n_requested
            if "wandb" in self.config.trainer.logger:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "teacher/request_failure_rate": failure_rate,
                        "teacher/n_requested": n_requested,
                        "teacher/n_failed": n_failed,
                        "teacher/n_succeeded": n_succeeded,
                    }, step=self.global_steps)
            if n_failed > 0:
                print(f"[teacher_feedback] {n_failed}/{n_requested} samples failed ({failure_rate:.1%})")

        return result
    
    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
        timing_raw: Optional[dict] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode not in ("sdpo", "rlsd", "srpo"):
            return None
        
        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        # Use original (pre-playbook-injection) raw_prompt for teacher/reflection prompts
        _raw_key = "original_raw_prompt" if "original_raw_prompt" in batch.non_tensor_batch else "raw_prompt"
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch[_raw_key]]
        batch_size = batch.batch.batch_size[0]

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        # Extract teacher feedback if available and teacher is enabled
        teacher_feedback_list: list[Any] = [None] * batch_size
        if self_distillation_cfg.teacher.enabled and reward_extra_infos_dict is not None:
            raw_teacher_feedback = reward_extra_infos_dict.get("teacher_feedback", [])
            for i in range(min(len(raw_teacher_feedback), batch_size)):
                if raw_teacher_feedback[i] and isinstance(raw_teacher_feedback[i], str) and raw_teacher_feedback[i].strip():
                    teacher_feedback_list[i] = raw_teacher_feedback[i]
            num_valid = sum(1 for fb in teacher_feedback_list if fb is not None)
            print(f"[Teacher] Extracted {num_valid}/{batch_size} teacher feedbacks (step={self.global_steps})")

        # --- Context Update (only when use_context_updater is enabled) ---
        previous_trials = None
        reflection_list = None
        playbook_stats = None
        # Extract stable example IDs for per-example playbook support
        example_ids = None
        if "extra_info" in batch.non_tensor_batch:
            example_ids = [info["index"] for info in batch.non_tensor_batch["extra_info"]]
        if self.use_context_updater:
            print(f"[ACE] Updating model weights before context update (step={self.global_steps})...")
            self.checkpoint_manager.update_weights()

            print(f"[ACE] Running context update (step={self.global_steps})...")
            _timing = timing_raw if timing_raw is not None else {}
            with marked_timer("context_update", _timing, color="olive"):
                acc_list = None
                if reward_extra_infos_dict is not None:
                    acc_list = reward_extra_infos_dict.get("acc", None)
                context_update_results = self.context_updater.update(
                    batch, self.async_rollout_manager, self.tokenizer, feedback_list,
                    teacher_feedback_list=teacher_feedback_list,
                    acc_list=acc_list,
                    example_ids=example_ids,
                )
            previous_trials = context_update_results.get("response_texts", None)
            reflection_list = context_update_results.get("reflection_texts", None)
            playbook_stats = context_update_results.get("final_stats", None)
            print(f"[ACE] Context update complete (step={self.global_steps}, took {_timing.get('context_update', 0):.1f}s).")
            if isinstance(self.context_updater, PlaybookContextUpdater):
                self._log_playbook_summary(self.context_updater, self.global_steps)
            else:
                self._log_playbook(self.context_updater.playbook, self.global_steps)

            self.checkpoint_manager.sleep_replicas()

            # Sync student playbook snapshot after context update if sync frequency is met
            if (self.use_playbook_in_student_rollout
                    and isinstance(self.context_updater, PlaybookContextUpdater)
                    and self.context_updater.should_sync_student()):
                self.context_updater.sync_student_playbooks()
                print(f"[ACE] Synced student playbook (step={self.global_steps})")

        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)

        # Per-sample correctness mask for SRPO routing
        correctness_mask = None
        if loss_mode == "srpo":
            seq_scores = reward_tensor.sum(dim=-1).detach()
            correctness_mask = (seq_scores >= self_distillation_cfg.success_reward_threshold).float().to(device)

        # Compute per-sample success-rate weights when enabled.
        success_rate_weights = None
        if self_distillation_cfg.get("success_rate_weighting", False):
            uids = batch.non_tensor_batch["uid"]
            uid_counts: dict[Any, int] = defaultdict(int)
            for uid in uids:
                uid_counts[uid] += 1
            sr_alpha = self_distillation_cfg.get("success_rate_alpha", 1.0)
            sr_beta = self_distillation_cfg.get("success_rate_beta", 1.0)
            raw_weights = []
            for i in range(batch_size):
                uid = uids[i]
                n_success = len(success_by_uid[uid])
                sr = n_success / uid_counts[uid]
                is_success = i in success_by_uid[uid]
                if is_success:
                    raw_weights.append((1.0 - sr) ** sr_alpha)
                else:
                    raw_weights.append(sr ** sr_beta)
            raw_weights_t = torch.tensor(raw_weights, dtype=torch.float32, device=device)
            mean_w = raw_weights_t.mean()
            if mean_w > 1e-8:
                success_rate_weights = raw_weights_t / mean_w
            else:
                success_rate_weights = torch.ones(batch_size, dtype=torch.float32, device=device)

        # Store successful trials into the solution buffer (context updater path only).
        # Store raw text — thinking trace removal is handled downstream by _get_solution.
        if self.use_context_updater and self.context_updater is not None and example_ids is not None:
            for uid, idxs in success_by_uid.items():
                if idxs:
                    eid = example_ids[idxs[0]]
                    self.context_updater.store_solution(eid, response_texts[idxs[0]])

        solution_strs = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                buffered_solution=(
                    self.context_updater.get_buffered_solution(example_ids[i])
                    if self.use_context_updater and self.context_updater is not None and example_ids is not None
                    else None
                ),
            )
            for i in range(batch_size)
        ]

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch[_raw_key][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            has_teacher_feedback = teacher_feedback_list[i] is not None
            include_previous_attempt = self_distillation_cfg.get("include_previous_attempt", False)
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
            teacher_feedback_only_without_solution = self_distillation_cfg.get("teacher_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
            use_teacher_feedback = has_teacher_feedback and (not teacher_feedback_only_without_solution or not has_solution)

            # ACE-specific flags (off by default)
            use_reflection = (
                self.use_context_updater
                and reflection_list is not None
                and reflection_list[i] is not None
                and _get_context_updater_cfg_value(
                    self_distillation_cfg,
                    nested_key="use_reflection_in_teacher_prompt",
                    legacy_key="use_reflection_in_teacher_prompt",
                    default=False,
                )
            )
            # Resolve playbook for this example
            _example_playbook = ""
            if self.use_context_updater and example_ids is not None:
                _example_playbook = self.context_updater.get_playbook(example_ids[i])
            elif self.use_context_updater and hasattr(self.context_updater, "playbook"):
                _example_playbook = self.context_updater.playbook
            use_playbook = (
                self.use_context_updater
                and _example_playbook != PlaybookContextUpdater.get_empty_playbook()
                and _get_context_updater_cfg_value(
                    self_distillation_cfg,
                    nested_key="use_playbook_in_teacher_prompt",
                    legacy_key="use_playbook_in_teacher_prompt",
                    default=False,
                )
            )

            # When context updater is active, use TEACHER_PROMPT; otherwise use reprompt_template
            if self.use_context_updater:
                # build solution section
                solution_section = ""
                if has_solution:
                    solution_section = self_distillation_cfg.solution_template.format(
                        successful_previous_attempt=solution_strs[i]
                    )

                # build feedback section, for context updater, you may choose to not include feedback in the teacher prompt even when env feedback is present, controlled by use_feedback_in_teacher_prompt
                feedback_section = ""
                if use_feedback and _get_context_updater_cfg_value(
                    self_distillation_cfg,
                    nested_key="use_feedback_in_teacher_prompt",
                    legacy_key="use_feedback_in_teacher_prompt",
                    default=False,
                ):
                    feedback_section = self_distillation_cfg.feedback_template.format(
                        feedback_raw=feedback_list[i]
                    )

                # build teacher feedback section
                teacher_feedback_section = ""
                if use_teacher_feedback:
                    teacher_feedback_section = self_distillation_cfg.teacher_feedback_template.format(
                        teacher_feedback_raw=teacher_feedback_list[i]
                    )
                
                use_previous_trial_in_teacher_prompt = _get_context_updater_cfg_value(
                    self_distillation_cfg,
                    nested_key="use_previous_trial_in_teacher_prompt",
                    legacy_key="use_previous_trial_in_teacher_prompt",
                    default=False,
                ) and previous_trials is not None and previous_trials[i] is not None

                use_solution_in_teacher_prompt = (
                    has_solution
                    and _get_context_updater_cfg_value(
                        self_distillation_cfg,
                        nested_key="use_solution_in_teacher_prompt",
                        legacy_key="use_solution_in_teacher_prompt",
                        default=False,
                    )
                )

                if use_feedback or has_solution or use_reflection:
                    format_kwargs = dict(
                        prompt=prompt_texts[i],
                        previous_trial=self._remove_thinking_trace(previous_trials[i]) if use_previous_trial_in_teacher_prompt else "",
                        feedback=feedback_section,
                        teacher_feedback=teacher_feedback_section,
                        reflection=self._remove_thinking_trace(reflection_list[i]) if use_reflection else "",
                        playbook=_example_playbook if use_playbook else "",
                    )
                    if "{solution}" in self.cu_teacher_prompt_template:
                        format_kwargs["solution"] = solution_section if use_solution_in_teacher_prompt else ""
                    reprompt_text = self.cu_teacher_prompt_template.format(**format_kwargs)
                else:
                    reprompt_text = prompt_texts[i]
            else:
                # Original behavior: use config-driven reprompt_template
                # build previous attempt section (only when feedback or teacher feedback is present)
                previous_attempt_section = ""
                previous_attempt_str = response_texts[i]
                if self_distillation_cfg.get("remove_thinking_from_demonstration", False):
                    previous_attempt_str = self._remove_thinking_trace(previous_attempt_str)
                if include_previous_attempt and (use_feedback or use_teacher_feedback):
                    previous_attempt_section = self_distillation_cfg.previous_attempt_template.format(
                        previous_attempt_raw=previous_attempt_str
                    )

                # build solution section
                solution_section = ""
                if has_solution:
                    solution_section = self_distillation_cfg.solution_template.format(
                        successful_previous_attempt=solution_strs[i]
                    )

                # build feedback section
                feedback_section = ""
                if use_feedback:
                    feedback_section = self_distillation_cfg.feedback_template.format(
                        feedback_raw=feedback_list[i]
                    )

                # build teacher feedback section (independent of env feedback)
                teacher_feedback_section = ""
                if use_teacher_feedback:
                    teacher_feedback_section = self_distillation_cfg.teacher_feedback_template.format(
                        teacher_feedback_raw=teacher_feedback_list[i]
                    )

                # combine all sections into the reprompt
                if use_feedback or use_teacher_feedback or has_solution:
                    reprompt_text = self_distillation_cfg.reprompt_template.format(
                        prompt=prompt_texts[i],
                        previous_attempt=previous_attempt_section,
                        solution=solution_section,
                        feedback=feedback_section,
                        teacher_feedback=teacher_feedback_section,
                    )
                else:
                    reprompt_text = prompt_texts[i]

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]
        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        teacher_feedback_only_without_solution = self_distillation_cfg.get("teacher_feedback_only_without_solution", False)
        if self.use_context_updater:
            use_feedback_in_teacher_prompt = _get_context_updater_cfg_value(
                self_distillation_cfg,
                nested_key="use_feedback_in_teacher_prompt",
                legacy_key="use_feedback_in_teacher_prompt",
                default=False,
            )
            feedback_used = [
                feedback_list[i] is not None and use_feedback_in_teacher_prompt
                for i in range(batch_size)
            ]
            teacher_feedback_used = [
                teacher_feedback_list[i] is not None
                for i in range(batch_size)
            ]
        else:
            feedback_used = [
                feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
                for i in range(batch_size)
            ]
            teacher_feedback_used = [
                teacher_feedback_list[i] is not None and (not teacher_feedback_only_without_solution or solution_strs[i] is None)
                for i in range(batch_size)
            ]

        # ACE-specific usage tracking
        use_reflection_in_teacher_prompt = _get_context_updater_cfg_value(
            self_distillation_cfg,
            nested_key="use_reflection_in_teacher_prompt",
            legacy_key="use_reflection_in_teacher_prompt",
            default=False,
        )
        reflection_used = [
            bool(self.use_context_updater and reflection_list is not None
            and reflection_list[i] and use_reflection_in_teacher_prompt)
            for i in range(batch_size)
        ]
        use_playbook_in_teacher_prompt = _get_context_updater_cfg_value(
            self_distillation_cfg,
            nested_key="use_playbook_in_teacher_prompt",
            legacy_key="use_playbook_in_teacher_prompt",
            default=False,
        )
        playbook_used = []
        for i in range(batch_size):
            if self.use_context_updater and example_ids is not None:
                pb = self.context_updater.get_playbook(example_ids[i])
            elif self.use_context_updater and hasattr(self.context_updater, "playbook"):
                pb = self.context_updater.playbook
            else:
                pb = ""
            playbook_used.append(
                self.use_context_updater
                and pb != PlaybookContextUpdater.get_empty_playbook()
                and use_playbook_in_teacher_prompt
            )

        # self_distillation_mask is True if sample has a solution OR feedback is used (i.e., will get a reprompted message)
        # a correct sample is always masked because the solution string is none
        self_distillation_mask = torch.tensor(
            [
                solution_strs[i] is not None or feedback_used[i] or teacher_feedback_used[i] or reflection_used[i]
                for i in range(batch_size)
            ],
            dtype=torch.float32,
            device=device
        )

        # Log a reprompted sample (with feedback/reflection) to wandb; fall back to index 0
        reprompted_indices = self_distillation_mask.nonzero(as_tuple=False)
        log_idx = reprompted_indices[0].item() if len(reprompted_indices) > 0 else 0
        self._log_teacher_prompt(messages, self.global_steps, sample_idx=log_idx)

        reprompt_lens = teacher_prompt["attention_mask"].sum(dim=1).float()

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_teacher_feedback_available = sum(1 for f in teacher_feedback_list if f is not None)
        num_with_teacher_feedback_used = sum(1 for f in teacher_feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/teacher_feedback_available_fraction": num_with_teacher_feedback_available / batch_size,
            "self_distillation/teacher_feedback_used_fraction": num_with_teacher_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
            "self_distillation/reprompt_len_mean": reprompt_lens.mean().item(),
            "self_distillation/reprompt_len_max": reprompt_lens.max().item(),
            "self_distillation/reprompt_truncated_fraction": (reprompt_lens == self_distillation_cfg.max_reprompt_len).float().mean().item(),
            "self_distillation/playbook_used_fraction": sum(playbook_used) / batch_size,
            "self_distillation/reflection_used_fraction": sum(reflection_used) / batch_size,
        }
        if playbook_stats is not None:
            metrics.update({
                "playbook/total_bullets": playbook_stats.get("total_bullets", 0),
                "playbook/high_performing": playbook_stats.get("high_performing", 0),
                "playbook/problematic": playbook_stats.get("problematic", 0),
                "playbook/unused": playbook_stats.get("unused", 0),
            })
        if correctness_mask is not None:
            z_sdpo = (1.0 - correctness_mask) * self_distillation_mask
            metrics["srpo/grpo_sample_frac"] = (1.0 - z_sdpo).mean().item()
            metrics["srpo/sdpo_sample_frac"] = z_sdpo.mean().item()
        tensors = {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }
        if correctness_mask is not None:
            tensors["correctness_mask"] = correctness_mask
        if success_rate_weights is not None:
            tensors["success_rate_weights"] = success_rate_weights
            metrics["self_distillation/sr_weight_min"] = success_rate_weights.min().item()
            metrics["self_distillation/sr_weight_max"] = success_rate_weights.max().item()
            metrics["self_distillation/sr_weight_std"] = success_rate_weights.std().item()
        return DataProto.from_dict(tensors=tensors), metrics

    @staticmethod
    def _compute_train_batch_metrics(
        reward_extra_infos_dict: dict[str, list], prefix: str
    ) -> dict[str, float]:
        """Compute mean of scalar fields in reward_extra_infos_dict and return under prefix."""
        metrics = {}
        for key, values in reward_extra_infos_dict.items():
            if isinstance(values, list) and len(values) > 0 and isinstance(values[0], str):
                continue  # skip string fields
            try:
                arr = np.array(values, dtype=float)
                metrics[f"{prefix}/{key}/mean"] = float(np.mean(arr))
            except (ValueError, TypeError):
                pass
        return metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _inject_playbook_into_batch(self, batch: DataProto) -> None:
        """Inject the student playbook snapshot into batch prompts before generation.

        Modifies raw_prompt in-place. Tokenization happens downstream in generate_sequences
        (via agent loop's apply_chat_template). Stores the original raw_prompt as
        'original_raw_prompt' for teacher/reflection prompt construction.
        """
        if not self.use_playbook_in_student_rollout:
            return
        if not self.use_context_updater or self.context_updater is None:
            return
        if "raw_prompt" not in batch.non_tensor_batch:
            return

        # Extract example IDs for per-example playbook lookup
        example_ids = None
        if "extra_info" in batch.non_tensor_batch:
            example_ids = [info["index"] for info in batch.non_tensor_batch["extra_info"]]

        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        batch_size = len(raw_prompts)

        # Collect playbooks and check if any are non-empty
        empty_pb = PlaybookContextUpdater.get_empty_playbook()
        playbooks = []
        any_playbook = False
        for i in range(batch_size):
            eid = example_ids[i] if example_ids is not None else "__global__"
            pb = self.context_updater.get_student_playbook(eid)
            playbooks.append(pb)
            if pb != empty_pb:
                any_playbook = True

        if not any_playbook:
            return  # All playbooks empty, skip injection

        # Save original raw_prompt for teacher/reflection prompt construction
        batch.non_tensor_batch["original_raw_prompt"] = raw_prompts.copy()

        # Build new messages with playbook injected via student prompt template
        new_messages = []
        for i in range(batch_size):
            system_messages = raw_prompts[i][:-1]
            original_prompt_text = raw_prompts[i][-1]["content"]

            if playbooks[i] != empty_pb:
                student_text = self.student_prompt_template.format(
                    playbook=playbooks[i], prompt=original_prompt_text,
                )
            else:
                student_text = original_prompt_text

            new_messages.append(system_messages + [{"role": "user", "content": student_text}])

        batch.non_tensor_batch["raw_prompt"] = np.array(new_messages, dtype=object)

        n_injected = sum(1 for pb in playbooks if pb != empty_pb)
        print(f"[StudentPlaybook] Injected playbook into {n_injected}/{batch_size} samples")

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        if not self.use_reward_loop:
            batch_reward = self.rm_wg.compute_rm_score(batch)
        else:
            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
            batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # The invocation of the reward function is agnostic to whether a reward model is used.
            # Decisions about when (e.g., training vs. validation) and whether to invoke the reward model
            # are delegated to user-defined reward functions.
            # if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            #     return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            if not self.use_reward_loop:
                reward_tensor, reward_extra_info = self._compute_reward_legacy(
                    test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
                )
            else:
                reward_tensor = test_batch.batch["rm_scores"]
                reward_extra_keys = test_batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: test_batch.non_tensor_batch[key] for key in reward_extra_keys}

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _update_best_val(self, val_metrics: dict, step: int) -> dict:
        """Track best validation performance so far.

        Looks for val-core keys and updates the best seen value for each.
        Returns a dict of best-val metrics to be logged.
        """
        if not hasattr(self, "_best_val"):
            self._best_val: dict[str, tuple[float, int]] = {}

        best_metrics: dict[str, float] = {}
        for key, value in val_metrics.items():
            if not key.startswith("val-core/"):
                continue
            prev_best, _ = self._best_val.get(key, (float("-inf"), -1))
            if value >= prev_best:
                self._best_val[key] = (value, step)
            best_val, best_step = self._best_val[key]
            best_key = key.replace("val-core/", "val-best/")
            best_metrics[best_key] = best_val
            best_metrics[best_key + "/step"] = best_step

        if best_metrics:
            pprint(f"[step {step}] Best validation so far: {best_metrics}")
        return best_metrics

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm and not self.use_reward_loop:
            raise RuntimeError("Reward model worker group is not supported, please set use_reward_loop=True")

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager

            # initalize reward loop manager
            # reward model (colocate or standalone): get resource_pool
            # no reward model: resource_pool = None
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(
                config=self.config,
                rm_resource_pool=resource_pool,
            )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = self.use_reward_loop and (
            not self.use_rm or self.config.reward_model.enable_resource_pool
        )
        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout

        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save context updater (playbook) state
        if self.use_context_updater and self.context_updater is not None:
            context_updater_path = os.path.join(local_global_step_folder, "context_updater.json")
            import json
            with open(context_updater_path, "w") as f:
                json.dump(self.context_updater.state_dict(), f, indent=2)
            print(f"Saved context updater state to {context_updater_path}")

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # Skip restoring dataloader state at exact epoch boundaries.
            # StatefulDataLoader saves an "exhausted" state at the end of an epoch,
            # which causes the next `for batch in dataloader` to yield nothing.
            steps_per_epoch = len(self.train_dataloader)
            if steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0:
                print(
                    f"Checkpoint at epoch boundary (step {self.global_steps}), "
                    f"skipping dataloader state restore to start fresh epoch"
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # load context updater (playbook) state
        if self.use_context_updater and self.context_updater is not None:
            context_updater_path = os.path.join(global_step_folder, "context_updater.json")
            if os.path.exists(context_updater_path):
                import json
                with open(context_updater_path, "r") as f:
                    context_updater_state = json.load(f)
                self.context_updater.load_state_dict(context_updater_state)
                print(f"Restored context updater state from {context_updater_path}")
            else:
                print(f"Warning: No context updater state found at {context_updater_path}, starting with empty playbook")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        batch.meta_info["default_local_dir"] = self.config.trainer.default_local_dir
        batch.meta_info["global_steps"] = self.global_steps
        batch.meta_info["end_think_token_id"] = self.tokenizer.convert_tokens_to_ids("</think>")
        batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            best_val_metrics = self._update_best_val(val_metrics, step=self.global_steps)
            val_metrics.update(best_val_metrics)
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Inject student playbook into prompts before generation
                self._inject_playbook_into_batch(batch)

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            if not self.use_reward_loop:
                                reward_baseline_tensor = self._compute_reward_legacy(
                                    batch, reward_fn=self.reward_fn, sum_reward=True
                                )
                            else:
                                reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # teacher_futures will be submitted after reward so env feedback can be included.
                    teacher_futures = None

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # Compute or extract reward_tensor and reward_extra_infos_dict for training
                        if not self.use_reward_loop:
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
                                    batch, reward_fn=self.reward_fn, reward_for_val=False
                                )
                        else:
                            reward_tensor = batch.batch["rm_scores"]
                            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                            reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}

                    if reward_extra_infos_dict:
                        metrics.update(
                            self._compute_train_batch_metrics(reward_extra_infos_dict, prefix="train/pre_update")
                        )
                    
                    # Kick off teacher feedback request now that env feedback is available.
                    # Overlaps with old_log_prob and ref_log_prob GPU computation below.
                    if self.teacher_client is not None:
                        env_fb = list(reward_extra_infos_dict.get("feedback", [None] * batch.batch.batch_size[0]))
                        acc_list = reward_extra_infos_dict.get("acc", None)
                        teacher_futures = self._submit_teacher_feedback(
                            batch,
                            env_feedback_list=env_fb,
                            acc_list=acc_list,
                        )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    if self.teacher_client is not None:
                        if teacher_futures is None:
                            env_fb = list(reward_extra_infos_dict.get("feedback", [None] * batch.batch.batch_size[0]))
                            acc_list = reward_extra_infos_dict.get("acc", None)
                            teacher_futures = self._submit_teacher_feedback(
                                batch,
                                env_feedback_list=env_fb,
                                acc_list=acc_list,
                            )
                        with marked_timer("teacher_feedback_wait", timing_raw):
                            reward_extra_infos_dict["teacher_feedback"] = self._resolve_teacher_feedback(
                                teacher_futures,
                                batch.batch.batch_size[0],
                            )
                            self._log_teacher_feedback(
                                batch,
                                reward_extra_infos_dict["teacher_feedback"],
                                self.global_steps,
                                env_feedback_list=env_fb,
                            )

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict, timing_raw=timing_raw)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_raw_metrics = actor_output.meta_info["metrics"]
                        def _pop_non_scalar(metrics, key):
                            val = metrics.pop(key, None)
                            if isinstance(val, list):
                                val = [v for v in val if v is not None]
                                if not val:
                                    return None
                            return val or None
                        sd_token_hist = _pop_non_scalar(actor_raw_metrics, "actor/sd_token_dist")
                        sd_token_by_pos = _pop_non_scalar(actor_raw_metrics, "actor/sd_token_by_pos")
                        sd_weight_by_pos = _pop_non_scalar(actor_raw_metrics, "actor/sd_weight_by_pos")
                        sd_effective_loss_by_pos = _pop_non_scalar(actor_raw_metrics, "actor/sd_effective_loss_by_pos")
                        student_entropy_by_pos = _pop_non_scalar(actor_raw_metrics, "actor/student_entropy_by_pos")
                        teacher_entropy_by_pos = _pop_non_scalar(actor_raw_metrics, "actor/teacher_entropy_by_pos")
                        actor_output_metrics = reduce_metrics(actor_raw_metrics)
                        if "wandb" in self.config.trainer.logger:
                            import wandb
                            if wandb.run is not None:
                                if sd_token_hist is not None:
                                    # sd_token_hist may be a list-of-lists (one per DP worker); flatten it
                                    if sd_token_hist and isinstance(sd_token_hist[0], (list, tuple)):
                                        flat = [v for sublist in sd_token_hist for v in sublist]
                                    else:
                                        flat = sd_token_hist
                                    actor_output_metrics["actor/sd_token_dist"] = wandb.Histogram(
                                        sequence=flat
                                    )
                                if sd_token_by_pos is not None:
                                    # sd_token_by_pos is [values_list, positions_list] or list-of-such from DP workers
                                    if sd_token_by_pos and isinstance(sd_token_by_pos[0], (list, tuple)) and len(sd_token_by_pos) == 2 and isinstance(sd_token_by_pos[0][0], (int, float)):
                                        all_vals, all_pos = np.array(sd_token_by_pos[0]), np.array(sd_token_by_pos[1])
                                    else:
                                        # list-of-[vals, pos] from multiple DP workers
                                        all_vals = np.concatenate([np.array(pair[0]) for pair in sd_token_by_pos])
                                        all_pos = np.concatenate([np.array(pair[1]) for pair in sd_token_by_pos])
                                    # Bin by position and compute mean sd_token per bin
                                    max_pos = int(all_pos.max()) if len(all_pos) > 0 else 0
                                    n_bins = min(max_pos, 64)
                                    if n_bins > 0:
                                        bin_edges = np.linspace(1, max_pos + 1, n_bins + 1)
                                        bin_indices = np.digitize(all_pos, bin_edges) - 1
                                        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                                        bin_means = np.full(n_bins, np.nan)
                                        bin_stds = np.full(n_bins, np.nan)
                                        for b in range(n_bins):
                                            mask = bin_indices == b
                                            if mask.any():
                                                bin_means[b] = all_vals[mask].mean()
                                                bin_stds[b] = all_vals[mask].std()
                                        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
                                        # Log as a line plot table
                                        table = wandb.Table(
                                            data=[[int(c), float(m), float(s)] for c, m, s in zip(bin_centers, bin_means, bin_stds) if not np.isnan(m)],
                                            columns=["position", "mean_sd_loss", "std_sd_loss"],
                                        )
                                        actor_output_metrics["actor/sd_token_by_position"] = wandb.plot.line(
                                            table, "position", "mean_sd_loss",
                                            title="SD Token Loss vs Response Position",
                                        )
                                if sd_weight_by_pos is not None:
                                    # sd_weight_by_pos is [weights_list, positions_list] or list-of-such from DP workers
                                    if sd_weight_by_pos and isinstance(sd_weight_by_pos[0], (list, tuple)) and len(sd_weight_by_pos) == 2 and isinstance(sd_weight_by_pos[0][0], (int, float)):
                                        all_weights, all_pos_w = np.array(sd_weight_by_pos[0]), np.array(sd_weight_by_pos[1])
                                    else:
                                        all_weights = np.concatenate([np.array(pair[0]) for pair in sd_weight_by_pos])
                                        all_pos_w = np.concatenate([np.array(pair[1]) for pair in sd_weight_by_pos])
                                    max_pos_w = int(all_pos_w.max()) if len(all_pos_w) > 0 else 0
                                    n_bins_w = min(max_pos_w, 64)
                                    if n_bins_w > 0:
                                        bin_edges_w = np.linspace(1, max_pos_w + 1, n_bins_w + 1)
                                        bin_indices_w = np.digitize(all_pos_w, bin_edges_w) - 1
                                        bin_indices_w = np.clip(bin_indices_w, 0, n_bins_w - 1)
                                        bin_means_w = np.full(n_bins_w, np.nan)
                                        for b in range(n_bins_w):
                                            mask = bin_indices_w == b
                                            if mask.any():
                                                bin_means_w[b] = all_weights[mask].mean()
                                        bin_centers_w = ((bin_edges_w[:-1] + bin_edges_w[1:]) / 2).astype(int)
                                        table_w = wandb.Table(
                                            data=[[int(c), float(m)] for c, m in zip(bin_centers_w, bin_means_w) if not np.isnan(m)],
                                            columns=["position", "mean_sd_weight"],
                                        )
                                        actor_output_metrics["actor/sd_weight_by_position"] = wandb.plot.line(
                                            table_w, "position", "mean_sd_weight",
                                            title="SD Token Weight vs Response Position",
                                        )
                                if sd_effective_loss_by_pos is not None:
                                    if sd_effective_loss_by_pos and isinstance(sd_effective_loss_by_pos[0], (list, tuple)) and len(sd_effective_loss_by_pos) == 2 and isinstance(sd_effective_loss_by_pos[0][0], (int, float)):
                                        all_eff, all_pos_eff = np.array(sd_effective_loss_by_pos[0]), np.array(sd_effective_loss_by_pos[1])
                                    else:
                                        all_eff = np.concatenate([np.array(pair[0]) for pair in sd_effective_loss_by_pos])
                                        all_pos_eff = np.concatenate([np.array(pair[1]) for pair in sd_effective_loss_by_pos])
                                    max_pos_eff = int(all_pos_eff.max()) if len(all_pos_eff) > 0 else 0
                                    n_bins_eff = min(max_pos_eff, 64)
                                    if n_bins_eff > 0:
                                        bin_edges_eff = np.linspace(1, max_pos_eff + 1, n_bins_eff + 1)
                                        bin_indices_eff = np.digitize(all_pos_eff, bin_edges_eff) - 1
                                        bin_indices_eff = np.clip(bin_indices_eff, 0, n_bins_eff - 1)
                                        bin_means_eff = np.full(n_bins_eff, np.nan)
                                        bin_stds_eff = np.full(n_bins_eff, np.nan)
                                        for b in range(n_bins_eff):
                                            mask = bin_indices_eff == b
                                            if mask.any():
                                                bin_means_eff[b] = all_eff[mask].mean()
                                                bin_stds_eff[b] = all_eff[mask].std()
                                        bin_centers_eff = ((bin_edges_eff[:-1] + bin_edges_eff[1:]) / 2).astype(int)
                                        table_eff = wandb.Table(
                                            data=[[int(c), float(m), float(s)] for c, m, s in zip(bin_centers_eff, bin_means_eff, bin_stds_eff) if not np.isnan(m)],
                                            columns=["position", "mean_effective_loss", "std_effective_loss"],
                                        )
                                        actor_output_metrics["actor/sd_effective_loss_by_position"] = wandb.plot.line(
                                            table_eff, "position", "mean_effective_loss",
                                            title="SD Effective Loss (loss*weight) vs Response Position",
                                        )
                                for ent_key, ent_data, ent_col, ent_title in [
                                    ("actor/student_entropy_by_position", student_entropy_by_pos, "mean_student_entropy", "Student Entropy vs Response Position"),
                                    ("actor/teacher_entropy_by_position", teacher_entropy_by_pos, "mean_teacher_entropy", "Teacher Entropy vs Response Position"),
                                ]:
                                    if ent_data is not None:
                                        if ent_data and isinstance(ent_data[0], (list, tuple)) and len(ent_data) == 2 and isinstance(ent_data[0][0], (int, float)):
                                            all_ent_vals, all_ent_pos = np.array(ent_data[0]), np.array(ent_data[1])
                                        else:
                                            all_ent_vals = np.concatenate([np.array(pair[0]) for pair in ent_data])
                                            all_ent_pos = np.concatenate([np.array(pair[1]) for pair in ent_data])
                                        max_pos_e = int(all_ent_pos.max()) if len(all_ent_pos) > 0 else 0
                                        n_bins_e = min(max_pos_e, 64)
                                        if n_bins_e > 0:
                                            bin_edges_e = np.linspace(1, max_pos_e + 1, n_bins_e + 1)
                                            bin_indices_e = np.clip(np.digitize(all_ent_pos, bin_edges_e) - 1, 0, n_bins_e - 1)
                                            bin_means_e = np.full(n_bins_e, np.nan)
                                            for b in range(n_bins_e):
                                                mask = bin_indices_e == b
                                                if mask.any():
                                                    bin_means_e[b] = all_ent_vals[mask].mean()
                                            bin_centers_e = ((bin_edges_e[:-1] + bin_edges_e[1:]) / 2).astype(int)
                                            table_e = wandb.Table(
                                                data=[[int(c), float(m)] for c, m in zip(bin_centers_e, bin_means_e) if not np.isnan(m)],
                                                columns=["position", ent_col],
                                            )
                                            actor_output_metrics[ent_key] = wandb.plot.line(
                                                table_e, "position", ent_col, title=ent_title,
                                            )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Log reprompt texts if enabled
                    reprompt_data_dir = self.config.trainer.get("reprompt_data_dir", None)
                    if reprompt_data_dir:
                        self._log_reprompt_data(batch, timing_raw, reprompt_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        best_val_metrics = self._update_best_val(val_metrics, step=self.global_steps)
                        val_metrics.update(best_val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
