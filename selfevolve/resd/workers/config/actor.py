# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.config.optimizer import OptimizerConfig

__all__ = ["TeacherFeedbackConfig", "ContextUpdaterConfig", "SelfDistillationConfig", "PolicyLossConfig", "RouterReplayConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


@dataclass
class TeacherFeedbackConfig(BaseConfig):
    """Configuration for the teacher feedback server.

    Args:
        enabled (bool): Whether the teacher feedback server is enabled.
        server_ip (str): IP address of the teacher feedback server.
        server_port (int): Port of the teacher feedback server.
        n_server_workers (int): Number of parallel server workers.
        max_tokens (int): Max tokens for teacher feedback generation.
        temperature (float): Sampling temperature for teacher feedback generation.
        max_feedback_prompt_len (int): Max tokens for the feedback prompt sent to the teacher.
        feedback_on_correct (bool): Whether to request teacher feedback for correctly solved samples
            (seq reward >= success_reward_threshold). Default False — only incorrect samples get feedback.
        feedback_prompt_template (str): Template for querying the teacher.
            Available variables: {prompt}, {response}, {feedback}.
    """

    enabled: bool = False
    server_ip: str = "127.0.0.1"
    server_port: int = 15555
    n_server_workers: int = 1
    max_tokens: int = 16384
    temperature: float = 0.7
    max_feedback_prompt_len: int = 4096
    feedback_on_correct: bool = False
    feedback_prompt_template: str = (
        "Here is a student's response to a question.\n"
        "Question: {prompt}\n"
        "Response: {response}\n"
        "Environment feedback: {feedback}\n"
        "Please inspect the model response and any thinking process before that, and provide concise "
        "feedback on errors and how to improve. Don't answer the question directly. Focus on pointing "
        "out the mistakes in the thinking and response.\n"
    )


@dataclass
class ContextUpdaterConfig(BaseConfig):
    """Configuration for ACE context updater.

    Args:
        enabled (bool): Whether to enable ACE context updater during self-distillation.
        playbook_mode (str): Playbook strategy. Options:
            - "global": single shared playbook across all examples (default, mirrors legacy ACE behavior).
            - "per_example": each training example gets a dedicated playbook, keyed by extra_info["index"].
        concise_frequency (Optional[int]): Frequency (in context updates) to concise the playbook.
        max_bullets (Optional[int]): Maximum number of playbook bullets to keep; None disables bullet-count trigger.
        concise_method (str): Playbook concise strategy. Options: "reset", "prioritized".
        use_reflection_in_teacher_prompt (bool): Whether to include ACE reflection in teacher prompt.
        use_playbook_in_teacher_prompt (bool): Whether to include ACE playbook in teacher prompt.
        use_feedback_in_teacher_prompt (bool): Whether to include environment feedback in teacher prompt.
        use_previous_trial_in_teacher_prompt (bool): Whether to include the student's previous trial in teacher prompt.
        use_solution_in_teacher_prompt (bool): Whether to include successful solutions in teacher prompt.
            Requires the template to contain a {solution} placeholder. Default False.
        tag_correct_samples (bool): Whether to run a lightweight success reflector on correct samples
            to tag which playbook bullets contributed to success, reinforcing their helpful counts.
            Default False — only incorrect samples update bullet counts.
        deduplicate_rollouts (bool): When rollout.n > 1, only run the curator and success tagging
            on one sample per unique example_id to avoid redundant playbook operations. Default False.
        reflector_prompt_template (Optional[str]): Template for the reflector prompt.
            Available variables: {prompt}, {response}, {feedback}, {teacher_feedback}, {playbook}.
            If null, uses the built-in default from prompts/ace_reflector.py.
        success_reflector_prompt_template (Optional[str]): Template for the success reflector prompt.
            Available variables: {prompt}, {response}, {playbook}.
            If null, uses the built-in default from prompts/ace_reflector.py.
        curator_prompt_template (Optional[str]): Template for the curator prompt.
            Available variables: {playbook_stats}, {recent_reflection}, {current_playbook}, {prompt}.
            If null, uses the built-in default from prompts/ace_curator.py.
        reflector_prompt_file (Optional[str]): Path to a text file containing the reflector prompt template.
            Takes precedence over reflector_prompt_template. If null, falls back to reflector_prompt_template.
        success_reflector_prompt_file (Optional[str]): Path to a text file containing the success reflector
            prompt template. Takes precedence over success_reflector_prompt_template.
        curator_prompt_file (Optional[str]): Path to a text file containing the curator prompt template.
            Takes precedence over curator_prompt_template. If null, falls back to curator_prompt_template.
        cu_teacher_prompt_template (Optional[str]): Template for the context-updater teacher (generator) prompt
            used during reprompting. Available variables: {playbook}, {prompt}, {previous_trial}, {feedback},
            {reflection}, {teacher_feedback}. If null, uses the built-in default from prompts/ace_generator.py.
        cu_teacher_prompt_file (Optional[str]): Path to a text file containing the context-updater teacher prompt
            template. Takes precedence over cu_teacher_prompt_template.
        use_solution_buffer (bool): Whether to cache successful response texts across training steps.
            When batch_size=1 there are no in-batch peers; the buffer lets the trainer re-use a
            successful trial from a previous step as the demonstration. Default False.
        concise_after_curation (bool): Whether to run an additional concise pass immediately after
            the curator adds new bullets, ensuring total bullet count stays within max_bullets.
            Without this, the pre-curation concise may leave room but the curator can push the
            count back over the limit. Default False.
    """

    enabled: bool = False
    playbook_mode: str = "global"
    concise_frequency: Optional[int] = 4
    max_bullets: Optional[int] = None
    concise_method: str = "reset"
    tag_correct_samples: bool = False
    deduplicate_rollouts: bool = False
    use_solution_buffer: bool = False
    concise_after_curation: bool = False
    use_reflection_in_teacher_prompt: bool = True
    use_playbook_in_teacher_prompt: bool = True
    use_feedback_in_teacher_prompt: bool = True
    use_previous_trial_in_teacher_prompt: bool = True
    use_solution_in_teacher_prompt: bool = False
    reflector_prompt_template: Optional[str] = None
    success_reflector_prompt_template: Optional[str] = None
    curator_prompt_template: Optional[str] = None
    reflector_prompt_file: Optional[str] = None
    success_reflector_prompt_file: Optional[str] = None
    curator_prompt_file: Optional[str] = None
    cu_teacher_prompt_template: Optional[str] = None
    cu_teacher_prompt_file: Optional[str] = None
    use_playbook_in_student_rollout: bool = False
    student_playbook_sync_frequency: Optional[int] = None
    student_prompt_template: Optional[str] = None
    student_prompt_file: Optional[str] = None

    def __post_init__(self):
        valid_playbook_modes = ["global", "per_example"]
        if self.playbook_mode not in valid_playbook_modes:
            raise ValueError(
                "self_distillation.context_updater.playbook_mode must be one of "
                f"{valid_playbook_modes}, got {self.playbook_mode}"
            )
        if self.concise_frequency is not None and self.concise_frequency <= 0:
            raise ValueError(
                "self_distillation.context_updater.concise_frequency must be a positive integer when set, got "
                f"{self.concise_frequency}"
            )
        if self.max_bullets is not None and self.max_bullets <= 0:
            raise ValueError(
                "self_distillation.context_updater.max_bullets must be a positive integer when set, got "
                f"{self.max_bullets}"
            )
        valid_concise_methods = ["reset", "prioritized", "staleness"]
        if self.concise_method not in valid_concise_methods:
            raise ValueError(
                "self_distillation.context_updater.concise_method must be one of "
                f"{valid_concise_methods}, got {self.concise_method}"
            )


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode == "sdpo".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation. Mutually exclusive with distillation_top_p.
        distillation_top_p (Optional[float]): If set, use nucleus (top-p) sampling for distillation. Mutually exclusive with distillation_topk.
        distillation_max_k (Optional[int]): Maximum number of tokens to keep when using top-p (memory cap). Only used with distillation_top_p.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k/top-p distillation.
        distillation_token_selector (str): Who determines the token support set. Options: "student", "teacher", "union".
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        teacher_prob_min_ratio (Optional[float]): Lower bound clamp on teacher probability relative to student probability.
            If set, enforces p_teacher >= teacher_prob_min_ratio * p_student.
        teacher_prob_max_ratio (Optional[float]): Upper bound clamp on teacher probability relative to student probability.
            If set, enforces p_teacher <= teacher_prob_max_ratio * p_student.
        entropy_diff_filter_ratio (Optional[float]): [Deprecated, use entropy_filter_ratio + entropy_filter_criterion]
            Fraction of tokens to keep per sequence based on (teacher_entropy - student_entropy). None disables.
        entropy_filter_ratio (Optional[float]): Fraction of tokens to keep per sequence based on entropy_filter_criterion. None disables.
        entropy_filter_criterion (str): Which entropy criterion to use for filtering. Options:
            - "diff": keep tokens with high (teacher_entropy - student_entropy). Teacher uncertain, student confident.
            - "abs_diff": keep tokens with high |teacher_entropy - student_entropy|. Biggest entropy gap regardless of sign.
            - "teacher_low": keep tokens with LOW teacher entropy. Teacher is confident (high-quality signal).
            - "teacher_high": keep tokens with HIGH teacher entropy. Teacher is uncertain (diverse signal).
            - "student_high": keep tokens with HIGH student entropy. Student is confused (needs learning).
            - "student_low": keep tokens with LOW student entropy. Student is confident.
            - "ratio": keep tokens with high (teacher_entropy / student_entropy). Relative uncertainty ratio.
        position_weighting_enabled (bool): Whether to enable position-wise weighting for self-distillation loss.
        position_weighting_beta (float): Linear ramp slope for position-wise weighting in (0, +inf).
        remove_thinking_in_loss (bool): Whether to remove <think>...</think> tokens from loss computation.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_previous_attempt (bool): Whether to include the student's own previous (failed) attempt in the reprompt.
        previous_attempt_template (str): Template for the previous attempt section. Uses {previous_attempt_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        teacher_feedback_only_without_solution (bool): If True, only use teacher feedback when no solution is available.
        teacher_feedback_template (str): Template for the teacher feedback section. Uses {teacher_feedback_raw} placeholder.
        teacher (TeacherFeedbackConfig): Configuration for the teacher feedback server.
        context_updater (Optional[ContextUpdaterConfig]): Nested ACE context updater configuration.
        use_context_updater (bool): [Legacy] Whether to enable ACE context updater during self-distillation.
        concise_frequency (Optional[int]): [Legacy] Frequency (in context updates) to concise the playbook.
        max_bullets (Optional[int]): [Legacy] Maximum number of playbook bullets to keep.
        concise_method (str): [Legacy] Playbook concise strategy. Options: "reset", "prioritized".
        use_reflection_in_teacher_prompt (bool): [Legacy] Whether to include ACE reflection in teacher prompt.
        use_playbook_in_teacher_prompt (bool): [Legacy] Whether to include ACE playbook in teacher prompt.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_top_p: Optional[float] = None
    distillation_max_k: Optional[int] = None
    distillation_add_tail: bool = True
    distillation_token_selector: str = "student"
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    teacher_prob_min_ratio: Optional[float] = None
    teacher_prob_max_ratio: Optional[float] = None
    entropy_diff_filter_ratio: Optional[float] = None  # deprecated, use entropy_filter_ratio + entropy_filter_criterion
    entropy_filter_ratio: Optional[float] = None
    entropy_filter_criterion: str = "diff"
    entropy_gt_filter: bool = False  # only keep tokens where teacher_entropy > student_entropy
    position_weighting_enabled: bool = False
    position_weighting_beta: float = 1.0
    remove_thinking_in_loss: bool = False
    reprompt_template: str = (
        "{prompt}{previous_attempt}{solution}{feedback}{teacher_feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_previous_attempt: bool = False
    previous_attempt_template: str = (
        "\n"
        "The following is your previous attempt:\n\n"
        "{previous_attempt_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_feedback_only_without_solution: bool = False
    teacher_feedback_template: str = (
        "\n"
        "The following is feedback from the teacher model:\n\n"
        "{teacher_feedback_raw}"
    )
    teacher: TeacherFeedbackConfig = field(default_factory=TeacherFeedbackConfig)
    context_updater: Optional[ContextUpdaterConfig] = None

    # RLSD (RLVR with Self-Distillation) config — used when policy_loss.loss_mode == "rlsd"
    rlsd_lambda_init: float = 0.5       # initial mixing coefficient for token-level credit
    rlsd_lambda_final: float = 0.0      # final mixing coefficient (decays linearly)
    rlsd_lambda_warmdown_steps: int = 50  # steps over which lambda decays
    rlsd_epsilon_w: float = 0.2         # clip range for evidence weights w_t

    # Success-rate weighting: weight per-token distillation loss by overall batch success rate.
    # Success samples get (1-sr)^alpha, failure samples get sr^beta, batch-normalized to mean 1.
    success_rate_weighting: bool = False
    success_rate_alpha: float = 1.0     # exponent for success sample weights: (1-sr)^alpha
    success_rate_beta: float = 1.0      # exponent for failure sample weights: sr^beta

    # SRPO (Sample-Routed Policy Optimization) config — used when policy_loss.loss_mode == "srpo"
    srpo_entropy_beta: float = 1.0      # β for DW-SDPO entropy weighting: w = exp(-β * H_teacher)

    # Legacy flat context-updater fields kept for backward compatibility.
    use_context_updater: bool = False
    concise_frequency: Optional[int] = 4
    max_bullets: Optional[int] = None
    concise_method: str = "reset"
    use_reflection_in_teacher_prompt: bool = False
    use_playbook_in_teacher_prompt: bool = False

    def get_context_updater_enabled(self) -> bool:
        return self.context_updater.enabled if self.context_updater is not None else self.use_context_updater

    def get_context_updater_concise_frequency(self) -> Optional[int]:
        return self.context_updater.concise_frequency if self.context_updater is not None else self.concise_frequency

    def get_context_updater_max_bullets(self) -> Optional[int]:
        return self.context_updater.max_bullets if self.context_updater is not None else self.max_bullets

    def get_context_updater_concise_method(self) -> str:
        return self.context_updater.concise_method if self.context_updater is not None else self.concise_method

    def get_use_reflection_in_teacher_prompt(self) -> bool:
        return (
            self.context_updater.use_reflection_in_teacher_prompt
            if self.context_updater is not None
            else self.use_reflection_in_teacher_prompt
        )

    def get_use_playbook_in_teacher_prompt(self) -> bool:
        return (
            self.context_updater.use_playbook_in_teacher_prompt
            if self.context_updater is not None
            else self.use_playbook_in_teacher_prompt
        )

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.distillation_top_p is not None and not (0.0 < self.distillation_top_p <= 1.0):
            raise ValueError(
                f"self_distillation.distillation_top_p must be in (0, 1], got {self.distillation_top_p}"
            )
        if self.distillation_topk is not None and self.distillation_top_p is not None:
            raise ValueError(
                "self_distillation.distillation_topk and distillation_top_p are mutually exclusive"
            )
        if self.distillation_max_k is not None and self.distillation_max_k <= 0:
            raise ValueError(
                f"self_distillation.distillation_max_k must be a positive integer, got {self.distillation_max_k}"
            )
        valid_token_selectors = ["student", "teacher", "union"]
        if self.distillation_token_selector not in valid_token_selectors:
            raise ValueError(
                f"self_distillation.distillation_token_selector must be one of "
                f"{valid_token_selectors}, got {self.distillation_token_selector}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.teacher_prob_min_ratio is not None and not (0.0 < self.teacher_prob_min_ratio <= 1.0):
            raise ValueError(
                f"self_distillation.teacher_prob_min_ratio must be in (0, 1], got {self.teacher_prob_min_ratio}"
            )
        if self.teacher_prob_max_ratio is not None and self.teacher_prob_max_ratio <= 0.0:
            raise ValueError(
                f"self_distillation.teacher_prob_max_ratio must be positive, got {self.teacher_prob_max_ratio}"
            )
        if (
            self.teacher_prob_min_ratio is not None
            and self.teacher_prob_max_ratio is not None
            and self.teacher_prob_min_ratio > self.teacher_prob_max_ratio
        ):
            raise ValueError(
                "self_distillation.teacher_prob_min_ratio must be <= teacher_prob_max_ratio, got "
                f"{self.teacher_prob_min_ratio} > {self.teacher_prob_max_ratio}"
            )
        valid_entropy_filter_criteria = ["diff", "abs_diff", "teacher_low", "teacher_high", "student_high", "student_low", "ratio"]
        if self.entropy_filter_criterion not in valid_entropy_filter_criteria:
            raise ValueError(
                f"self_distillation.entropy_filter_criterion must be one of "
                f"{valid_entropy_filter_criteria}, got {self.entropy_filter_criterion}"
            )
        if self.entropy_filter_ratio is not None and not (0.0 < self.entropy_filter_ratio <= 1.0):
            raise ValueError(
                f"self_distillation.entropy_filter_ratio must be in (0, 1], got {self.entropy_filter_ratio}"
            )
        if self.entropy_diff_filter_ratio is not None and not (0.0 < self.entropy_diff_filter_ratio <= 1.0):
            raise ValueError(
                f"self_distillation.entropy_diff_filter_ratio must be in (0, 1], got {self.entropy_diff_filter_ratio}"
            )
        if self.position_weighting_beta <= 0.0:
            raise ValueError(
                "self_distillation.position_weighting_beta must be > 0, got "
                f"{self.position_weighting_beta}"
            )
        if self.context_updater is None:
            if self.concise_frequency is not None and self.concise_frequency <= 0:
                raise ValueError(
                    "self_distillation.concise_frequency must be a positive integer when set, got "
                    f"{self.concise_frequency}"
                )
            if self.max_bullets is not None and self.max_bullets <= 0:
                raise ValueError(
                    "self_distillation.max_bullets must be a positive integer when set, got "
                    f"{self.max_bullets}"
                )
            valid_concise_methods = ["reset", "prioritized", "staleness"]
            if self.concise_method not in valid_concise_methods:
                raise ValueError(
                    "self_distillation.concise_method must be one of "
                    f"{valid_concise_methods}, got {self.concise_method}"
                )

@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)
    token_loss_dump_n: int = 0

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
