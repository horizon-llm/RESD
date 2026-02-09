"""
ACE Trainer
"""

import os
import time
import uuid
from typing import Optional
from pprint import pprint
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

from selfevolve.ace.utils import load_initial_playbook, extract_bullet_tags, extract_answer, extract_bullet_ids
from selfevolve.ace.prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT, REFLECTOR_PROMPT_NO_GT, CURATOR_PROMPT_NO_GT
from selfevolve.ace.playbook_utils import extract_playbook_bullets, update_bullet_counts, get_playbook_stats

WorkerType = type[Worker]


class ACETrainer(RayPPOTrainer):
    """
    ACE Trainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False
        self.use_json_mode = config.trainer.get("use_json_mode", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.params_dtype = PrecisionType.to_dtype("bfloat16")
    
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

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

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset:
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

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

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
    
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        # Build Ray classes per pool
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Rollout group
        rollout_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
        )
        resource_pool_to_cls[rollout_pool]["rollout"] = rollout_cls

        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        
        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            time.sleep(20)  # avoid port conflict
        
        self.rollout_wg = all_wg["rollout"]

        # Initialize both groups
        self.rollout_wg.init_model()
    
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
    
    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _load_playbook(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        playbook_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(playbook_folder):
            working_dir = os.getcwd()
            playbook_folder = os.path.join(working_dir, playbook_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.playbook = self._initialize_empty_playbook()
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

        self.playbook = load_initial_playbook(os.path.join(global_step_folder, "playbook.txt"))

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        gen_batch = batch.pop(
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch.meta_info["global_steps"] = self.global_steps
        return gen_batch

    def _apply_generation_template(self, batch: DataProto, reflections: list = None) -> DataProto:        
        # format prompt with playbook
        extra_infos = batch.non_tensor_batch.get("extra_info", {})
        questions = extra_infos.get("question", [])
        contexts = extra_infos.get("context", [])

        generator_prompts = []
        for question, context, reflection in zip(questions, contexts, reflections or ["(empty)"] * len(questions)):
            generator_prompts.append(GENERATOR_PROMPT.format(
                self.playbook, reflection, question, context
            ))
        generator_messages = [[{"role": "user", "content": prompt}] for prompt in generator_prompts]
        new_batch = deepcopy(batch)
        new_batch.non_tensor_batch["raw_prompt"] = np.array(generator_messages, dtype=object)
        return new_batch
    
    def _apply_reflection_template(self, batch: DataProto, no_ground_truth: bool = False) -> DataProto:
        extra_infos = batch.non_tensor_batch.get("extra_info", {})
        questions = extra_infos.get("question", [])
        targets = extra_infos.get("target", []) if not no_ground_truth else None
        responses = extra_infos.get("generation", [])
        predicted_answers = extra_infos.get("extracted_answer", [])
        environment_feedbacks = extra_infos.get("environment_feedback", ["Predicted answer does not match ground truth."] * len(questions))
        bullet_ids = extra_infos.get("bullet_id", [])

        # Get bullets for reflector
        playbook_bullets = extract_playbook_bullets(
            self.playbook, bullet_ids
        )
        
        reflector_prompts = []
        for question, target, response, predicted_answer, feedback, bullets in zip(questions, targets, responses, predicted_answers, environment_feedbacks, playbook_bullets):
            if no_ground_truth:
                prompt = REFLECTOR_PROMPT_NO_GT.format(
                    question,
                    response,
                    predicted_answer,
                    feedback,
                    bullets,
                )
            else:
                prompt = REFLECTOR_PROMPT.format(
                    question,
                    response,
                    predicted_answer,
                    target,
                    feedback,
                    bullets,
                )
            reflector_prompts.append(prompt)
        reflector_messages = [[{"role": "user", "content": prompt}] for prompt in reflector_prompts]
        new_batch = deepcopy(batch)
        new_batch.non_tensor_batch["raw_prompt"] = np.array(reflector_messages, dtype=object)
        return new_batch

    def _apply_curation_template(self, batch: DataProto, no_ground_truth: bool = False) -> DataProto:
        extra_infos = batch.non_tensor_batch.get("extra_info", {})

        playbook_stats = get_playbook_stats(self.playbook)

        recent_reflection = extra_infos['reflection_results'][-1]['reflection']
        context = extra_infos['context'][-1] # only need the context, not the sample
        total_playbook_token_budget = self.config.trainer.get("total_playbook_token_budget", 80000)
        
        if no_ground_truth:
            curation_prompt = CURATOR_PROMPT_NO_GT.format(
                recent_reflection=recent_reflection,
                context=context,
                playbook_stats=playbook_stats,
                total_playbook_token_budget=total_playbook_token_budget,
            )
        else:
            curation_prompt = CURATOR_PROMPT.format(
                recent_reflection=recent_reflection,
                context=context,
                playbook_stats=playbook_stats,
                total_playbook_token_budget=total_playbook_token_budget,
            )
        
        curation_messages = [[{"role": "user", "content": curation_prompt}]] * len(batch.batch)
        # copy the batch
        new_batch = deepcopy(batch)

        new_batch.non_tensor_batch["raw_prompt"] = np.array(curation_messages, dtype=object)
        return new_batch
    
    def _curate(self, batch: DataProto) -> DataProto:
        gen_batch = self._apply_curation_template(batch)
        gen_outputs = self.rollout_wg.generate_sequences(gen_batch)
        for data_item in gen_outputs:
            # decode responses
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            curation_response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Check for empty response error
            if curation_response.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
                print(f"⏭️  Skipping curator operation due to empty response")
                return current_playbook

    def _iterative_reflection(self, batch: DataProto) -> DataProto:
        max_num_rounds = self.config.trainer.get("max_reflection_rounds", 3)
        batch_size = len(batch.batch)

        # Track which samples are still incorrect using original indices
        active_indices = torch.arange(batch_size)
        # Clone the batch so we can write back results per-round
        result_batch = batch

        # save results in each round
        reflect_results = [[]] * batch_size

        for round_num in range(max_num_rounds):
            if len(active_indices) == 0:
                break  # All samples resolved

            # Subset to only the still-incorrect samples
            active_mask = torch.zeros(batch_size, dtype=torch.bool)
            active_mask[active_indices] = True
            active_batch = batch[active_mask]

            # Reflect on incorrect samples
            gen_batch = self._apply_reflection_template(active_batch)
            gen_outputs = self.rollout_wg.generate_sequences(gen_batch)

            batch_reflections = []
            batch_bullet_tags = []
            for data_item in gen_outputs:
                # decode responses
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                reflection = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                batch_reflections.append(reflection)

                # extract bullet tags
                bullet_tags = extract_bullet_tags(reflection, use_json_mode=self.use_json_mode)
                batch_bullet_tags.append(bullet_tags)

                if bullet_tags:
                    self.playbook = update_bullet_counts(self.playbook, bullet_tags)

            # Regenerate with reflection
            gen_batch = self._generate(active_batch, reflections=batch_reflections)

            # Check correctness
            reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
                gen_batch, reward_fn=self.reward_fn, reward_for_val=False
            )
            is_correct = reward_extra_infos_dict["acc"]

            # Write back results for all active samples into result_batch
            for key in gen_batch.batch.keys():
                result_batch.batch[key][active_indices] = gen_batch.batch[key]
            active_indices_np = active_indices.numpy()
            for key in gen_batch.non_tensor_batch.keys():
                result_batch.non_tensor_batch[key][active_indices_np] = gen_batch.non_tensor_batch[key]
            
            for idx, reflection in zip(active_indices, batch_reflections):
                reflect_results[idx] = {
                    "round": round_num,
                    "reflection": reflection,
                    "bullet_tags": batch_bullet_tags[idx],
                    "gen_result": gen_batch.non_tensor_batch["extra_info"]["gen_results"][idx]
                }

            # Remove newly correct samples from the active set
            if is_correct is not None:
                # is_correct is a bool tensor over the active subset
                still_incorrect = ~is_correct
                active_indices = active_indices[still_incorrect]
        
        result_batch.non_tensor_batch['extra_info']['reflection_results'] = np.array(reflect_results, dtype=object)

        return result_batch
    
    def _generate(self, batch: DataProto, reflections: list = None) -> DataProto:
        gen_batch = self._apply_generation_template(batch, reflections=reflections)
        gen_outputs = self.rollout_wg.generate_sequences(gen_batch)
        gen_batch = gen_batch.union(gen_outputs)

        all_responses = []
        all_bullet_ids = []
        all_extracted_answers = []
        for data_item in gen_outputs:
            # decode responses
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            all_responses.append(response_str)

            # extract bullet ids
            bullet_ids = extract_bullet_ids(response_str, use_json_mode=self.use_json_mode)
            all_bullet_ids.append(bullet_ids)

            # Extract answer
            extracted_answer = extract_answer(response_str, use_json_mode=self.use_json_mode)
            all_extracted_answers.append(extracted_answer)
        
        gen_results = []
        for response, bullet_ids, extracted_answer in zip(all_responses, all_bullet_ids, all_extracted_answers):
            gen_results.append({
                "response": response,
                "bullet_id": bullet_ids,
                "extracted_answer": extracted_answer,
            })
        gen_batch.non_tensor_batch["extra_info"]["gen_results"] = gen_results
        return gen_batch

    def _train_single_batch(self, batch: DataProto):
        """
        Train a single step of ACE, the original implementation can only take one sample at a time. Here we modify it to support batch training.

        Args:
            batch (DataProto): The input batch for the training step.

        Returns:
            dict: A dictionary containing the metrics for the training step.
        """
        # STEP 1: Initial generation (pre-train)
        gen_batch = self._generate(batch)

        # Extract answer and check correctness
        reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
            gen_batch, reward_fn=self.reward_fn, reward_for_val=False
        )
        is_correct = reward_extra_infos_dict.get("acc", None)
        pre_train_answer = reward_extra_infos_dict.get("pred", None)

        # STEP 2: Reflection and regeneration
        # select incorrect samples for reflection
        incorrect_mask = ~is_correct if is_correct is not None else torch.zeros(len(batch.batch), dtype=torch.bool)
        incorrect_indices = torch.where(incorrect_mask)[0]
        if incorrect_mask.any():
            incorrect_batch = gen_batch[incorrect_mask]
            reflected_batch = self._iterative_reflection(incorrect_batch)

            for key in reflected_batch.batch.keys():
                gen_batch.batch[key][incorrect_mask] = reflected_batch.batch[key]
            
            incorrect_indices_np = incorrect_indices.numpy()
            for key in reflected_batch.non_tensor_batch.keys():
                gen_batch.non_tensor_batch[key][incorrect_indices_np] = reflected_batch.non_tensor_batch[key]
        else:
            # For correct answers - still run reflector to tag helpful bullets
            correct_batch = gen_batch[~incorrect_mask]

            correct_batch = self._apply_reflection_template(correct_batch)
            reflect_outputs = self.rollout_wg.generate_sequences(correct_batch)
            correct_batch = correct_batch.union(reflect_outputs)

            reflections = self.tokenizer.batch_decode(correct_batch.batch['responses'], skip_special_tokens=True)
            bullet_tags = extract_bullet_tags(reflections, use_json_mode=self.use_json_mode)

            if bullet_tags:
                self.playbook = update_bullet_counts(self.playbook, bullet_tags)
        
        # STEP 3: Curator - Periodically update playbook
        if self.global_steps % self.config.trainer.curation_interval == 0:
            print(f"\n--- Running Curator at step {self.global_steps} ---")

            self._curate(gen_batch)
        
        # STEP 4: Post-curator generation
        gen_batch = self._generate(gen_batch)
    
    def fit(self):
        """
        The training loop of ACE

        reference from `verl/trainer/ppo/ray_trainer.py` since it supports resume training
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

        self._load_playbook()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
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

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.rollout_wg.start_profiling()
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                self._train_single_batch(gen_batch)