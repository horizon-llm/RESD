"""
ACE Batch System

Extends ACE with parallel generation and reflection for offline training.
Curation remains sequential since it modifies the shared playbook.

Usage:
    from ace import ACEBatch

    ace_system = ACEBatch(
        api_provider="vllm",
        generator_model="model-name",
        reflector_model="model-name",
        curator_model="model-name",
        batch_size=10,
        batch_workers=10
    )

    results = ace_system.run(
        mode='offline',
        train_samples=train_data,
        val_samples=val_data,
        test_samples=test_data,
        data_processor=processor,
        config=config
    )
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from selfevolve.ace import ACE
from selfevolve.ace.playbook_utils import (
    extract_playbook_bullets, update_bullet_counts, get_playbook_stats,
    apply_curator_operations
)
from selfevolve.ace.logger import log_bullet_usage
from selfevolve.ace.utils import extract_answer, count_tokens, evaluate_test_set


class ACEBatch(ACE):
    """
    Batch version of ACE with parallel generation, reflection, and curation.

    Extends ACE to process training samples in batches:
    - Generation runs in parallel across samples in a batch
    - Reflection (+ re-generation rounds) runs in parallel
    - Curation runs in parallel: all curator LLM calls fire against the same
      playbook snapshot, then all resulting ADD operations are merged and applied
      in one pass — so samples within a batch have no visibility into each
      other's additions
    - Post-curator generation runs in parallel

    Only affects offline training. Online training and eval_only
    use the original sequential implementation from ACE.
    """

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        batch_size: int = 10,
        batch_workers: int = 10
    ):
        """
        Initialize the ACEBatch system.

        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer
            batch_size: Number of samples to process in parallel per batch
            batch_workers: Number of parallel threads per batch
        """
        super().__init__(
            api_provider=api_provider,
            generator_model=generator_model,
            reflector_model=reflector_model,
            curator_model=curator_model,
            max_tokens=max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=bulletpoint_analyzer_threshold
        )
        self.batch_size = batch_size
        self.batch_workers = batch_workers

    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor: "selfevolve.ace.data_processor.DataProcessor",
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Batch offline training with parallel generation/reflection.

        Overrides ACE._offline_train to process samples in batches.
        Within each batch:
        1. Initial generation runs in parallel
        2. Reflection + re-generation runs in parallel
        3. Bullet count updates applied sequentially
        4. Curation runs sequentially
        5. Post-curator generation runs in parallel
        """
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']

        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        print(f"Total epochs: {num_epochs}")
        print(f"Train samples per epoch: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} batches")
        print(f"Batch size: {self.batch_size}")
        print(f"Batch workers: {self.batch_workers}\n")

        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs} (BATCH MODE)")
            print(f"{'='*60}")

            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []

            # Process in batches
            batch_counter = 0
            for batch_start in range(0, len(train_samples), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(train_samples))
                batch = train_samples[batch_start:batch_end]
                # Steps are 1-indexed to match original
                batch_steps = list(range(batch_start + 1, batch_end + 1))

                print(f"\n{'='*60}")
                print(f"BATCH: Steps {batch_steps[0]}-{batch_steps[-1]}/{len(train_samples)}")
                print(f"{'='*60}")

                # Process batch with parallel gen/reflect + sequential curate
                t0 = time.perf_counter()
                batch_results = self._train_batch(
                    batch_samples=batch,
                    data_processor=data_processor,
                    epoch=epoch,
                    batch_steps=batch_steps,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(train_samples)
                )
                batch_time = time.perf_counter() - t0
                print(f"Batch processing time: {batch_time:.2f} seconds")

                # Collect results
                for i, result in enumerate(batch_results):
                    step = batch_steps[i]
                    target = batch[i].get("target", "")

                    epoch_answers_pre_train.append(result['pre_train_answer'])
                    epoch_targets_pre_train.append(target)
                    epoch_answers_post_train.append(result['post_train_answer'])
                    epoch_targets_post_train.append(target)

                    pre_train_post_train_result = {
                        "epoch": epoch,
                        "step": step,
                        "target": target,
                        "batch_time": batch_time,
                        **result['tracking_dict'],
                        **result['latency']
                    }
                    pre_train_post_train_results.append(pre_train_post_train_result)

                    # Log per-step metrics to wandb
                    global_step = (epoch - 1) * len(train_samples) + step
                    self._wandb_log({
                        "train/step_pre_correct": int(result['tracking_dict']['pre_train_result']['is_correct']),
                        "train/step_post_correct": int(result['tracking_dict']['post_train_result']['is_correct']),
                        "playbook/num_tokens": result['tracking_dict']['post_train_result']['playbook_num_tokens'],
                        "playbook/length": result['tracking_dict']['post_train_result']['playbook_length'],
                        "epoch": epoch,
                    }, step=global_step)

                # Save pre_train_post_train_results incrementally
                pre_train_post_train_results_path = os.path.join(
                    save_path, "pre_train_post_train_results.json"
                )
                with open(pre_train_post_train_results_path, "w") as f:
                    json.dump(pre_train_post_train_results, f, indent=2)

                # Save intermediate playbook at checkpoint steps within this batch
                for step in batch_steps:
                    if step % save_steps == 0:
                        intermediate_path = os.path.join(
                            playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                        )
                        with open(intermediate_path, "w") as f:
                            f.write(self.playbook)

                # Periodic evaluation based on accumulated batch count
                batch_counter += 1
                if batch_counter % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, BATCH {batch_counter} (STEP {batch_steps[-1]})")
                    print(f"{'='*40}")

                    # Compute training accuracies
                    pre_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )

                    # Validation evaluation
                    val_results = {}
                    val_error_log = {}
                    if val_samples:
                        t0 = time.perf_counter()
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, self.playbook,
                            val_samples, self.max_tokens, log_dir,
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )
                        val_time = time.perf_counter() - t0

                    # Count unparsable generations
                    pre_train_unparsable = sum(1 for a in epoch_answers_pre_train if a == "No final answer found")
                    post_train_unparsable = sum(1 for a in epoch_answers_post_train if a == "No final answer found")

                    result = {
                        "epoch": epoch,
                        "batch": batch_counter,
                        "step": batch_steps[-1],
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy,
                            "pre_train_unparsable": pre_train_unparsable,
                            "post_train_unparsable": post_train_unparsable
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook),
                        "val_time": val_time if val_samples else None
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "batch": batch_counter,
                        "step": batch_steps[-1],
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"New best accuracy: {best_accuracy:.3f}")

                    # Log evaluation metrics to wandb
                    global_step = (epoch - 1) * len(train_samples) + batch_steps[-1]
                    eval_log = {
                        "eval/pre_train_accuracy": pre_train_accuracy,
                        "eval/post_train_accuracy": post_train_accuracy,
                        "eval/pre_train_unparsable": pre_train_unparsable,
                        "eval/post_train_unparsable": post_train_unparsable,
                        "eval/best_val_accuracy": best_accuracy,
                    }
                    if val_results:
                        eval_log["eval/val_accuracy"] = val_results.get("accuracy", 0)
                        eval_log["eval/val_correct"] = val_results.get("correct", 0)
                        eval_log["eval/val_no_answer"] = val_results.get("no_answer", 0)
                    stats = get_playbook_stats(self.playbook)
                    eval_log["playbook/total_bullets"] = stats.get("total_bullets", 0)
                    eval_log["playbook/high_performing"] = stats.get("high_performing", 0)
                    eval_log["playbook/problematic"] = stats.get("problematic", 0)
                    eval_log["playbook/unused"] = stats.get("unused", 0)
                    self._wandb_log(eval_log, step=global_step)

                    # Save results
                    results_path = os.path.join(save_path, "train_results.json")
                    with open(results_path, "w") as f:
                        json.dump({
                            "best_accuracy": best_accuracy,
                            "results": results,
                        }, f, indent=2)

                    error_logs_path = os.path.join(save_path, "val_results.json")
                    with open(error_logs_path, "w") as f:
                        json.dump(error_logs, f, indent=2)

            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

        # Save training results
        results_path = os.path.join(save_path, "train_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_accuracy": best_accuracy,
                "results": results,
            }, f, indent=2)

        pre_train_post_train_results_path = os.path.join(
            save_path, "pre_train_post_train_results.json"
        )
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)

        # Save final playbook
        final_playbook_path = os.path.join(save_path, "final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)

        # Save best playbook
        best_playbook_path = os.path.join(save_path, "best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)

        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE (BATCH MODE)")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy}

    def _train_batch(
        self,
        batch_samples: List[Dict[str, Any]],
        data_processor,
        epoch: int,
        batch_steps: List[int],
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of samples with parallel gen/reflect/curate.

        Phases:
        1. Parallel initial generation (all samples use same playbook snapshot)
        2. Parallel reflection + re-generation rounds (each sample independent)
        3. Sequential bullet count updates (applied to shared playbook)
        4. Parallel curation: all curator LLM calls use the same snapshot;
           collected operations are merged and applied in one pass
        5. Parallel post-curator generation (all samples use updated playbook)

        Args:
            batch_samples: List of sample dicts for this batch
            data_processor: Data processor for evaluation
            epoch: Current epoch number
            batch_steps: List of 1-indexed step numbers for each sample
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in dataset

        Returns:
            List of result dicts, one per sample, each containing
            pre_train_answer, post_train_answer, and tracking_dict
        """
        max_num_rounds = config_params['max_num_rounds']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        # Snapshot playbook for parallel phases
        playbook_snapshot = self.playbook

        # =====================================================================
        # Phase 1: Parallel initial generation
        # =====================================================================
        print(f"\n--- Phase 1: Parallel Initial Generation "
              f"({len(batch_samples)} samples) ---")
        gen_results = self._parallel_generate(
            samples=batch_samples,
            playbook=playbook_snapshot,
            data_processor=data_processor,
            use_json_mode=use_json_mode,
            log_dir=log_dir,
            step_ids=[f"train_e_{epoch}_s_{s}" for s in batch_steps],
            suffix="_gen_initial"
        )

        for i, gen_result in enumerate(gen_results):
            print(f"  Step {batch_steps[i]}: "
                  f"correct={gen_result['is_correct']}")

        # =====================================================================
        # Phase 2: Parallel reflection + re-generation
        # =====================================================================
        print(f"\n--- Phase 2: Parallel Reflection + Re-generation ---")
        reflect_results = self._parallel_reflect_and_regenerate(
            samples=batch_samples,
            gen_results=gen_results,
            playbook=playbook_snapshot,
            data_processor=data_processor,
            max_num_rounds=max_num_rounds,
            no_ground_truth=no_ground_truth,
            use_json_mode=use_json_mode,
            log_dir=log_dir,
            step_ids=[f"train_e_{epoch}_s_{s}" for s in batch_steps]
        )

        # =====================================================================
        # Phase 3: Sequential bullet count updates + logging
        # =====================================================================
        print(f"\n--- Phase 3: Sequential Bullet Count Updates ---")
        for i, (sample, gen_result, ref_result) in enumerate(
            zip(batch_samples, gen_results, reflect_results)
        ):
            step = batch_steps[i]

            # Log initial bullet usage (matches original behavior)
            log_bullet_usage(
                usage_log_path, epoch, step, sample,
                gen_result['bullet_ids'],
                playbook=self.playbook,
                is_correct=gen_result['is_correct']
            )

            # Apply all collected bullet tags from this sample's reflection
            for tags in ref_result['all_bullet_tags']:
                if tags:
                    self.playbook = update_bullet_counts(self.playbook, tags)

            # For correct samples, log again with reflection content
            # (matches original behavior where correct samples get a second log)
            if gen_result['is_correct']:
                log_bullet_usage(
                    usage_log_path, epoch, step, sample,
                    gen_result['bullet_ids'],
                    playbook=self.playbook,
                    reflection_content=ref_result['reflection_content'],
                    is_correct=gen_result['is_correct']
                )

        # =====================================================================
        # Phase 4: Parallel curation
        # =====================================================================
        print(f"\n--- Phase 4: Parallel Curation ---")
        curation_time, curated_steps = self._parallel_curate(
            batch_samples=batch_samples,
            reflect_results=reflect_results,
            batch_steps=batch_steps,
            epoch=epoch,
            total_samples=total_samples,
            config_params=config_params,
            log_dir=log_dir
        )

        # =====================================================================
        # Phase 5: Parallel post-curator generation
        # =====================================================================
        print(f"\n--- Phase 5: Parallel Post-Curator Generation ---")
        post_gen_results = self._parallel_generate(
            samples=batch_samples,
            playbook=self.playbook,
            data_processor=data_processor,
            use_json_mode=use_json_mode,
            log_dir=log_dir,
            step_ids=[f"train_e_{epoch}_s_{s}" for s in batch_steps],
            suffix="_post_curate"
        )

        # =====================================================================
        # Combine results
        # =====================================================================
        combined = []
        for i in range(len(batch_samples)):
            combined.append({
                'pre_train_answer': gen_results[i]['final_answer'],
                'post_train_answer': post_gen_results[i]['final_answer'],
                'tracking_dict': {
                    'pre_train_result': {
                        'final_answer': gen_results[i]['final_answer'],
                        'is_correct': gen_results[i]['is_correct'],
                        'playbook_num_tokens': count_tokens(playbook_snapshot),
                        'playbook_length': len(playbook_snapshot)
                    },
                    'post_train_result': {
                        'final_answer': post_gen_results[i]['final_answer'],
                        'is_correct': post_gen_results[i]['is_correct'],
                        'playbook_num_tokens': count_tokens(self.playbook),
                        'playbook_length': len(self.playbook)
                    }
                },
                "latency": {
                    "initial_generation_time": gen_results[i]['gen_time'],
                    "reflection_time": reflect_results[i]['reflection_time'],
                    "curation_time": curation_time if batch_steps[i] in curated_steps else None,
                    "post_curator_generation_time": post_gen_results[i]['gen_time']
                }
            })

        return combined

    def _parallel_generate(
        self,
        samples: List[Dict[str, Any]],
        playbook: str,
        data_processor,
        use_json_mode: bool,
        log_dir: str,
        step_ids: List[str],
        suffix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Run generation in parallel for a batch of samples.

        Args:
            samples: List of sample dicts
            playbook: Playbook string (read-only, shared across workers)
            data_processor: Data processor for correctness checking
            use_json_mode: Whether to use JSON mode
            log_dir: Directory for logging
            step_ids: List of step ID prefixes for each sample
            suffix: Suffix to append to call_id (e.g., "_gen_initial")

        Returns:
            List of result dicts in same order as input samples
        """
        def generate_single(args):
            i, sample, step_id = args
            try:
                t0 = time.perf_counter()
                gen_response, bullet_ids, call_info = self.generator.generate(
                    question=sample.get("question", ""),
                    playbook=playbook,
                    context=sample.get("context", ""),
                    reflection="(empty)",
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}{suffix}",
                    log_dir=log_dir
                )
                gen_time = time.perf_counter() - t0

                final_answer = extract_answer(gen_response)
                target = sample.get("target", "")
                is_correct = data_processor.answer_is_correct(final_answer, target)

                return {
                    'index': i,
                    'gen_response': gen_response,
                    'bullet_ids': bullet_ids,
                    'final_answer': final_answer,
                    'is_correct': is_correct,
                    'call_info': call_info,
                    'gen_time': gen_time,
                    'success': True
                }
            except Exception as e:
                print(f"Error generating for sample {i}: {e}")
                return {
                    'index': i,
                    'gen_response': '',
                    'bullet_ids': [],
                    'final_answer': 'ERROR',
                    'is_correct': False,
                    'call_info': {},
                    'success': False
                }

        args_list = [
            (i, sample, step_id)
            for i, (sample, step_id) in enumerate(zip(samples, step_ids))
        ]

        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
            future_to_idx = {
                executor.submit(generate_single, args): args[0]
                for args in args_list
            }
            for future in as_completed(future_to_idx):
                result = future.result()
                results[result['index']] = result

        return results

    def _parallel_reflect_and_regenerate(
        self,
        samples: List[Dict[str, Any]],
        gen_results: List[Dict[str, Any]],
        playbook: str,
        data_processor,
        max_num_rounds: int,
        no_ground_truth: bool,
        use_json_mode: bool,
        log_dir: str,
        step_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Run reflection and re-generation rounds in parallel for a batch.

        Each sample processes independently using a local copy of the playbook
        snapshot. Bullet count updates during reflection rounds only affect the
        local copy; the global playbook is updated later in the sequential phase.

        Args:
            samples: List of sample dicts
            gen_results: List of generation results from Phase 1
            playbook: Playbook snapshot (read-only base for local copies)
            data_processor: Data processor for correctness checking
            max_num_rounds: Maximum reflection rounds for incorrect answers
            no_ground_truth: Whether to skip ground truth in reflection
            use_json_mode: Whether to use JSON mode
            log_dir: Directory for logging
            step_ids: List of step ID prefixes for each sample

        Returns:
            List of reflection result dicts in same order as input samples
        """
        def reflect_single(args):
            i, sample, gen_result, step_id = args
            try:
                question = sample.get("question", "")
                target = sample.get("target", "")
                context = sample.get("context", "")

                gen_response = gen_result['gen_response']
                bullet_ids = gen_result['bullet_ids']
                final_answer = gen_result['final_answer']
                is_correct = gen_result['is_correct']

                # Each worker gets its own playbook copy for local updates
                local_playbook = playbook

                all_bullet_tags = []
                reflection_content = "(empty)"

                if not is_correct:
                    # Incorrect: iterate reflection rounds
                    for round_num in range(max_num_rounds):
                        print(f"  Step {step_id}: "
                              f"Reflection round {round_num + 1}/{max_num_rounds}")

                        playbook_bullets = extract_playbook_bullets(
                            local_playbook, bullet_ids
                        )

                        t0 = time.perf_counter()
                        reflection_content, bullet_tags, _ = \
                            self.reflector.reflect(
                                question=question,
                                reasoning_trace=gen_response,
                                predicted_answer=final_answer,
                                ground_truth=(target
                                              if not no_ground_truth else None),
                                environment_feedback=(
                                    "Predicted answer does not match "
                                    "ground truth"),
                                bullets_used=playbook_bullets,
                                use_ground_truth=not no_ground_truth,
                                use_json_mode=use_json_mode,
                                call_id=f"{step_id}_round_{round_num}",
                                log_dir=log_dir
                            )
                        reflection_time = time.perf_counter() - t0

                        all_bullet_tags.append(bullet_tags)

                        # Update local playbook copy
                        if bullet_tags:
                            local_playbook = update_bullet_counts(
                                local_playbook, bullet_tags
                            )

                        # Re-generate with reflection
                        gen_response, bullet_ids, _ = \
                            self.generator.generate(
                                question=question,
                                playbook=local_playbook,
                                context=context,
                                reflection=reflection_content,
                                use_json_mode=use_json_mode,
                                call_id=(f"{step_id}_post_reflect"
                                         f"_round_{round_num}"),
                                log_dir=log_dir
                            )

                        final_answer = extract_answer(gen_response)

                        if data_processor.answer_is_correct(
                            final_answer, target
                        ):
                            print(f"  Step {step_id}: Corrected after "
                                  f"reflection round {round_num + 1}!")
                            is_correct = True
                            break
                else:
                    # Correct: still reflect to tag helpful bullets
                    playbook_bullets = extract_playbook_bullets(
                        local_playbook, bullet_ids
                    )

                    t0 = time.perf_counter()
                    reflection_content, bullet_tags, _ = \
                        self.reflector.reflect(
                            question=question,
                            reasoning_trace=gen_response,
                            predicted_answer=final_answer,
                            ground_truth=(target
                                          if not no_ground_truth else None),
                            environment_feedback=(
                                "Predicted answer matches ground truth"),
                            bullets_used=playbook_bullets,
                            use_ground_truth=not no_ground_truth,
                            use_json_mode=use_json_mode,
                            call_id=f"{step_id}_reflect_on_correct",
                            log_dir=log_dir
                        )
                    reflection_time = time.perf_counter() - t0

                    all_bullet_tags.append(bullet_tags)

                return {
                    'index': i,
                    'reflection_content': reflection_content,
                    'all_bullet_tags': all_bullet_tags,
                    'final_answer': final_answer,
                    'is_correct': is_correct,
                    'success': True,
                    'reflection_time': reflection_time
                }
            except Exception as e:
                print(f"Error reflecting for sample {i}: {e}")
                return {
                    'index': i,
                    'reflection_content': "(empty)",
                    'all_bullet_tags': [],
                    'final_answer': gen_result['final_answer'],
                    'is_correct': gen_result['is_correct'],
                    'success': False,
                    'reflection_time': 0.0
                }

        args_list = [
            (i, sample, gen_result, step_id)
            for i, (sample, gen_result, step_id)
            in enumerate(zip(samples, gen_results, step_ids))
        ]

        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
            future_to_idx = {
                executor.submit(reflect_single, args): args[0]
                for args in args_list
            }
            for future in as_completed(future_to_idx):
                result = future.result()
                results[result['index']] = result

        return results

    def _parallel_curate(
        self,
        batch_samples: List[Dict[str, Any]],
        reflect_results: List[Dict[str, Any]],
        batch_steps: List[int],
        epoch: int,
        total_samples: int,
        config_params: Dict[str, Any],
        log_dir: str
    ) -> Tuple[float, set]:
        """
        Run curation in parallel for all eligible samples, then merge operations.

        All curator LLM calls fire against the same playbook snapshot, so no
        sample within the batch can see another's additions.  After all calls
        complete, the collected ADD operations are applied together in one pass
        with a single call to apply_curator_operations, which assigns fresh
        bullet IDs sequentially.

        Args:
            batch_samples: List of sample dicts for this batch
            reflect_results: Reflection results from Phase 2
            batch_steps: 1-indexed step numbers for each sample
            epoch: Current epoch number
            total_samples: Total number of training samples
            config_params: Configuration parameters dictionary
            log_dir: Path for logging directory

        Returns:
            Tuple of (curation_wall_time, curated_steps) where
            curation_wall_time is the wall-clock seconds for the parallel phase
            (0.0 if no samples were eligible) and curated_steps is the set of
            step numbers that triggered curation.
        """
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        no_ground_truth = config_params['no_ground_truth']
        use_json_mode = config_params['use_json_mode']

        # Determine which samples are eligible for curation this batch
        eligible = [
            (i, batch_samples[i], reflect_results[i], batch_steps[i])
            for i in range(len(batch_samples))
            if batch_steps[i] % curator_frequency == 0
        ]
        curated_steps = {step for _, _, _, step in eligible}

        if not eligible:
            return 0.0, curated_steps

        # All calls share the same snapshot — no intra-batch visibility
        playbook_snapshot = self.playbook
        stats = get_playbook_stats(playbook_snapshot)

        def curate_single(args):
            _, sample, ref_result, step = args
            try:
                # We only need the operations list; ignore the returned playbook
                # since IDs are assigned later during the merged apply step.
                _, _, operations, _ = self.curator.curate(
                    current_playbook=playbook_snapshot,
                    recent_reflection=ref_result['reflection_content'],
                    question_context=sample.get("context", ""),
                    current_step=step,
                    total_samples=total_samples,
                    token_budget=token_budget,
                    playbook_stats=stats,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"train_e_{epoch}_s_{step}",
                    log_dir=log_dir,
                    next_global_id=self.next_global_id  # read-only here
                )
                return operations
            except Exception as e:
                print(f"Error in curator for step {step}: {e}")
                return []

        t0 = time.perf_counter()
        per_sample_ops: List[Optional[List]] = [None] * len(eligible)
        with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
            future_to_idx = {
                executor.submit(curate_single, args): j
                for j, args in enumerate(eligible)
            }
            for future in as_completed(future_to_idx):
                j = future_to_idx[future]
                per_sample_ops[j] = future.result()
        curation_time = time.perf_counter() - t0

        # Merge all operations and apply in a single pass so IDs are assigned
        # sequentially and no sample's bullets collide with another's.
        all_operations = [op for ops in per_sample_ops if ops for op in ops]
        if all_operations:
            self.playbook, self.next_global_id = apply_curator_operations(
                self.playbook, all_operations, self.next_global_id
            )

        if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
            print(f"  Running BulletpointAnalyzer "
                  f"(threshold={self.bulletpoint_analyzer_threshold})...")
            self.playbook = self.bulletpoint_analyzer.analyze(
                playbook=self.playbook,
                threshold=self.bulletpoint_analyzer_threshold,
                merge=True
            )

        print(f"  Parallel curation complete: {len(eligible)} curator calls, "
              f"{len(all_operations)} operations applied in {curation_time:.2f}s")

        return curation_time, curated_steps
