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
StreamRayPPOTrainer extended with critic/value model support for PPO with GAE.

Based on stream_trainer.py with these additions:
1. Critic worker group creation and management
2. Value computation for GAE advantage estimation
3. Critic model updates during training
4. Critic checkpointing (save/load)
"""

import os
import uuid
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.reward import compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.workers.config import FSDPEngineConfig
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from .stream_trainer import StreamRayPPOTrainer, compute_advantage, compute_response_mask, apply_kl_penalty


class StreamRayPPOTrainerWithCritic(StreamRayPPOTrainer):
    """Stream trainer with critic/value model for PPO with GAE advantages.

    Inherits all functionality from StreamRayPPOTrainer and adds:
    - Critic worker group lifecycle (create, init, checkpoint)
    - _compute_values / _update_critic methods
    - GAE-compatible training loop
    """

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
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        self.use_critic = need_critic(self.config)

    # ------------------------------------------------------------------
    # Worker initialization — must override entirely because critic needs
    # to be in resource_pool_to_cls before the colocated spawn pass.
    # ------------------------------------------------------------------
    def init_workers(self):
        """Initialize distributed workers including critic worker group."""
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

        # spawn colocated worker groups
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
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

        # initialize critic
        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
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
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # actor/rollout must be last so vllm can estimate kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # reward loop manager
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        # async rollout manager
        self.async_rollout_mode = True

        from verl.utils.import_utils import load_class_from_fqn
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        enable_agent_reward_loop = self.use_reward_loop and (
            not self.use_rm or self.config.reward_model.enable_resource_pool
        )
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        self.checkpoint_manager.sleep_replicas()

    # ------------------------------------------------------------------
    # Critic compute / update
    # ------------------------------------------------------------------
    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
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

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
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
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    # ------------------------------------------------------------------
    # Checkpoint save/load — extend parent to include critic
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int | None = None):
        super()._save_checkpoint(step=step)

        if not self.use_critic:
            return

        save_step = step if step is not None else self.global_steps
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{save_step}"
        )
        critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
        critic_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{save_step}", str(Role.Critic)
            )
        )
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        self.critic_wg.save_checkpoint(
            critic_local_path, critic_remote_path, save_step, max_ckpt_to_keep=max_critic_ckpt_to_keep
        )

    def _load_checkpoint(self):
        super()._load_checkpoint()

        if not self.use_critic or self.global_steps == 0:
            return

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))

        if os.path.exists(critic_path):
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
            print(f"Loaded critic checkpoint from {critic_path}")
        else:
            print(f"Warning: No critic checkpoint at {critic_path}, critic starts from random init")

    # ------------------------------------------------------------------
    # Profiling — extend parent to include critic
    # ------------------------------------------------------------------
    def _start_profiling(self, do_profile: bool) -> None:
        super()._start_profiling(do_profile)
        if do_profile and self.use_critic:
            self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        super()._stop_profiling(do_profile)
        if do_profile and self.use_critic:
            self.critic_wg.stop_profile()

    # ------------------------------------------------------------------
    # fit — full override to insert critic compute/update into inner loop
    # Based on StreamRayPPOTrainer.fit (stream_trainer.py)
    # ------------------------------------------------------------------
    def fit(self):
        """Training loop with critic support for GAE advantage estimation."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        forget_cfg = self.config.trainer.get("forget_eval", {})
        forget_freq = forget_cfg.get("eval_freq", 0)

        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

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

        _max_k = self.config.trainer.get("max_updates_per_batch", 1)
        progress_bar = tqdm(
            total=len(self.train_dataloader) * _max_k,
            initial=self.global_steps,
            desc="Training Progress",
        )

        self.global_steps += 1
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Critic warmup: number of steps to train only the critic before actor updates begin
        critic_warmup_steps = self.config.trainer.get("critic_warmup_steps", 0)

        for batch_idx, batch_dict in enumerate(self.train_dataloader, start=self._resumed_batch_idx):
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

            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            self._inject_playbook_into_batch(batch)

            base_batch = deepcopy(batch)

            prev_scores = None
            max_k = self.config.trainer.get("max_updates_per_batch", 1)
            min_k = self.config.trainer.get("min_updates_per_batch", 1)
            threshold = self.config.trainer.get("early_stop_improvement_threshold", 0.0)

            is_last_batch = (batch_idx + 1) >= len(self.train_dataloader)
            inner_stopped_early = False

            for update_iter in range(max_k):
                gen_batch = self._get_gen_batch(deepcopy(base_batch))
                gen_batch.meta_info["global_steps"] = self.global_steps

                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

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

                    batch = base_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    teacher_futures = None

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

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

                    if self.teacher_client is not None:
                        env_fb = list(reward_extra_infos_dict.get("feedback", [None] * batch.batch.batch_size[0]))
                        acc_list = reward_extra_infos_dict.get("acc", None)
                        teacher_futures = self._submit_teacher_feedback(
                            batch,
                            env_feedback_list=env_fb,
                            acc_list=acc_list,
                        )

                    # Operating Mode Selection
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            from verl.trainer.ppo.core_algos import agg_loss
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
                                from verl.utils.debug.metrics import calculate_debug_metrics
                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # === CRITIC: compute values for GAE ===
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    if self.teacher_client is not None:
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

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    reprompt_data_dir = self.config.trainer.get("reprompt_data_dir", None)
                    if reprompt_data_dir:
                        self._log_reprompt_data(batch, timing_raw, reprompt_data_dir)

                    # === CRITIC: update critic ===
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # update actor (gated by critic warmup)
                    in_critic_warmup = self.use_critic and self.global_steps <= critic_warmup_steps
                    if not in_critic_warmup:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_raw_metrics = actor_output.meta_info["metrics"]
                        # Pop histogram data before reduce_metrics (it only handles scalars)
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
                                    if sd_token_hist and isinstance(sd_token_hist[0], (list, tuple)) and len(sd_token_hist) == 2 and isinstance(sd_token_hist[0][0], (int, float)):
                                        all_vals, all_pos = np.array(sd_token_hist[0]), np.array(sd_token_hist[1])
                                    else:
                                        all_vals = np.concatenate([np.array(pair[0]) for pair in sd_token_hist])
                                        all_pos = np.concatenate([np.array(pair[1]) for pair in sd_token_hist])
                                    if len(all_vals) > 0:
                                        actor_output_metrics["actor/sd_token_id_dist"] = wandb.Histogram(all_vals.tolist())
                                        max_pos = int(all_pos.max()) if len(all_pos) > 0 else 0
                                        n_bins = min(max_pos, 64)
                                        if n_bins > 0:
                                            bin_edges = np.linspace(1, max_pos + 1, n_bins + 1)
                                            bin_indices = np.digitize(all_pos, bin_edges) - 1
                                            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                                            bin_means = np.full(n_bins, np.nan)
                                            for b in range(n_bins):
                                                mask = bin_indices == b
                                                if mask.any():
                                                    bin_means[b] = all_vals[mask].mean()
                                            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
                                            table = wandb.Table(
                                                data=[[int(c), float(m)] for c, m in zip(bin_centers, bin_means) if not np.isnan(m)],
                                                columns=["position", "mean_sd_loss"],
                                            )
                                            actor_output_metrics["actor/sd_loss_by_position"] = wandb.plot.line(
                                                table, "position", "mean_sd_loss",
                                                title="SD Token Loss vs Response Position",
                                            )
                                if sd_weight_by_pos is not None:
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
                    else:
                        metrics["training/critic_warmup"] = 1.0

                    # early stop
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    n_prompts = len(base_batch.batch)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
                    scores_per_prompt = scores.reshape(n_prompts, rollout_n).mean(-1)

                    if forget_freq > 0 and update_iter == 0:
                        self.past_batch_buffer.append(
                            (batch_idx, deepcopy(base_batch), scores_per_prompt.copy())
                        )

                    if prev_scores is not None and update_iter + 1 >= min_k:
                        improved_frac = float((scores_per_prompt > prev_scores).mean())
                        metrics["training/improved_frac"] = improved_frac
                        if improved_frac < threshold:
                            inner_stopped_early = True
                    prev_scores = scores_per_prompt

                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights()

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

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/batch_idx": batch_idx,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                is_last_inner = (update_iter + 1 >= max_k) or inner_stopped_early
                if not is_last_inner:
                    logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                last_inner_step = self.global_steps
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if inner_stopped_early:
                    break

            # ----------------------------------------------------------------
            # OUTER LOOP
            # ----------------------------------------------------------------
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )

            if self.config.trainer.test_freq > 0 and (
                is_last_batch
                or (batch_idx + 1) % self.config.trainer.test_freq == 0
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    best_val_metrics = self._update_best_val(val_metrics, step=last_inner_step)
                    val_metrics.update(best_val_metrics)
                metrics.update(val_metrics)

            if forget_freq > 0 and (
                is_last_batch
                or (batch_idx + 1) % forget_freq == 0
            ):
                with marked_timer("forget_eval", timing_raw, color="cyan"):
                    forget_metrics = self._eval_on_past_batches()
                metrics.update(forget_metrics)

            logger.log(data=metrics, step=last_inner_step)

            if self.config.trainer.save_freq > 0 and (
                is_last_batch
                or (batch_idx + 1) % self.config.trainer.save_freq == 0
                or esi_close_to_expiration
            ):
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint(step=last_inner_step)

            if is_last_batch:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                progress_bar.close()
                return

            if hasattr(self.train_dataset, "on_batch_end"):
                self.train_dataset.on_batch_end(batch=batch)
