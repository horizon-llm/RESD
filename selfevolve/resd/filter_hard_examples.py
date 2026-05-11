"""Evaluate a model on reasoning tasks and filter to hard examples (pass@k == 0).

Supports different reward functions via --reward_function_path / --reward_function_name.

Usage (sudoku / sokoban - reasoning_gym dispatcher):
    python -m selfevolve.resd.filter_hard_examples \
        --model Qwen/Qwen3-4B-Thinking-2507 \
        --data selfevolve/resd/datasets/sudoku/train.parquet \
        --output selfevolve/resd/datasets/sudoku/train_hard.parquet \
        --k 8 --enable_thinking

Usage (manufactoria - dedicated scorer):
    python -m selfevolve.resd.filter_hard_examples \
        --model Qwen/Qwen3-4B-Thinking-2507 \
        --data selfevolve/resd/datasets/manufactoria/train_-1.parquet \
        --output selfevolve/resd/datasets/manufactoria/train_hard.parquet \
        --reward_function_path selfevolve/resd/feedback/manufactoria.py \
        --k 8 --enable_thinking
"""

import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model and filter to hard examples (pass@k == 0)")
    parser.add_argument("--model", type=str, required=True, help="Model name or path for vLLM")
    parser.add_argument("--data", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file (hard examples only)")
    parser.add_argument("--k", type=int, default=8, help="Number of samples per problem for pass@k")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=25600)
    parser.add_argument("--max_model_len", type=int, default=74752)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode in chat template")
    parser.add_argument("--pass_threshold", type=float, default=1.0,
                        help="Score threshold to count as 'pass' (default: 1.0 = exact match)")
    parser.add_argument("--reward_function_path", type=str, default=None,
                        help="Path to reward function module (default: reasoning_gym_games dispatcher)")
    parser.add_argument("--reward_function_name", type=str, default="compute_score",
                        help="Name of the scoring function in the module (default: compute_score)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="Optional path to save per-example results as JSONL")
    return parser.parse_args()


def load_data(data_path: str):
    """Load parquet dataset and return as list of dicts."""
    ds = datasets.load_dataset("parquet", data_files=data_path, split="train")
    return ds


def build_prompts(ds, tokenizer, enable_thinking: bool):
    """Build tokenized prompts from dataset using the model's chat template."""
    prompts = []
    for example in ds:
        messages = example["prompt"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(text)
    return prompts


def load_reward_function(reward_function_path, reward_function_name):
    """Dynamically load a reward function from a file path."""
    if reward_function_path is None:
        # Default: use reasoning_gym_games dispatcher.
        # Load directly via importlib to avoid triggering feedback/__init__.py
        # which has deep relative imports that fail outside the full package.
        rg_path = Path(__file__).resolve().parent / "feedback" / "reasoning_gym_games" / "__init__.py"
        spec = importlib.util.spec_from_file_location("reasoning_gym_games", str(rg_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.compute_score

    path = Path(reward_function_path).resolve()
    spec = importlib.util.spec_from_file_location("reward_module", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, reward_function_name)


def score_responses(responses_per_example, ds, reward_fn):
    """Score all responses using the provided scoring function."""
    # Detect whether the scoring function takes data_source as first arg.
    # reasoning_gym_games.compute_score: (data_source, solution_str, ground_truth, ...)
    # manufactoria.compute_score:        (solution_str, ground_truth, ...)
    sig = inspect.signature(reward_fn)
    params = list(sig.parameters.keys())
    has_data_source = "data_source" in params

    results = []
    for idx, responses in enumerate(tqdm(responses_per_example, desc="Scoring")):
        example = ds[idx]
        ground_truth = example["reward_model"]["ground_truth"]
        extra_info = example["extra_info"]
        data_source = example["data_source"]

        scores = []
        for resp in responses:
            if has_data_source:
                result = reward_fn(
                    data_source=data_source,
                    solution_str=resp,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    sparse_rewards=False,
                )
            else:
                result = reward_fn(
                    solution_str=resp,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    sparse_rewards=False,
                )
            scores.append(result)

        results.append({
            "index": idx,
            "scores": [s["score"] for s in scores],
            "accs": [s["acc"] for s in scores],
            "max_score": max(s["score"] for s in scores),
            "max_acc": max(s["acc"] for s in scores),
            "mean_score": sum(s["score"] for s in scores) / len(scores),
        })

    return results


def write_parquet_large(ds, path: str, rows_per_group: int = 32):
    """Write dataset to parquet with large string/list types to avoid overflow."""
    def _to_large(field: pa.Field) -> pa.Field:
        t = field.type
        if pa.types.is_string(t):
            return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
        if pa.types.is_binary(t):
            return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
        if pa.types.is_list(t):
            return pa.field(field.name, pa.large_list(_to_large(pa.field("item", t.value_type)).type),
                            field.nullable, field.metadata)
        if pa.types.is_struct(t):
            return pa.field(field.name,
                pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
                field.nullable, field.metadata)
        return field

    def _large_schema(schema: pa.Schema) -> pa.Schema:
        return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])

    tbl: pa.Table = ds.flatten_indices().data.table
    tbl = tbl.cast(_large_schema(tbl.schema))
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


def main():
    args = parse_args()

    # 1. Load data
    print(f"Loading data from {args.data}")
    ds = load_data(args.data)
    print(f"Loaded {len(ds)} examples")

    # 2. Initialize vLLM
    print(f"Loading model {args.model}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # 3. Build prompts
    print("Building prompts...")
    prompts = build_prompts(ds, tokenizer, args.enable_thinking)
    print(f"Built {len(prompts)} prompts")

    # 4. Generate responses
    print(f"Generating {args.k} responses per example...")
    outputs = llm.generate(prompts, sampling_params)

    # Organize: list of lists of response strings
    responses_per_example = []
    for output in outputs:
        responses = [o.text for o in output.outputs]
        responses_per_example.append(responses)

    # 5. Score all responses
    print("Scoring responses...")
    reward_fn = load_reward_function(args.reward_function_path, args.reward_function_name)
    results = score_responses(responses_per_example, ds, reward_fn)

    # 6. Print summary statistics
    total = len(results)
    num_pass = sum(1 for r in results if r["max_acc"] >= args.pass_threshold)
    num_fail = total - num_pass
    print(f"\n{'='*60}")
    print(f"Results Summary (pass@{args.k}, threshold={args.pass_threshold}):")
    print(f"  Total examples:  {total}")
    print(f"  pass@{args.k} > 0:     {num_pass} ({num_pass/total*100:.1f}%)")
    print(f"  pass@{args.k} == 0:    {num_fail} ({num_fail/total*100:.1f}%) <- hard examples")
    print(f"  Mean max_score:  {sum(r['max_score'] for r in results)/total:.4f}")
    print(f"  Mean mean_score: {sum(r['mean_score'] for r in results)/total:.4f}")
    print(f"{'='*60}\n")

    # 7. Optionally save detailed results
    if args.save_results:
        print(f"Saving detailed results to {args.save_results}")
        with open(args.save_results, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # 8. Filter to hard examples (pass@k == 0)
    hard_indices = [r["index"] for r in results if r["max_acc"] < args.pass_threshold]

    if not hard_indices:
        print("No hard examples found! All examples were solved at least once.")
        return

    print(f"Filtering to {len(hard_indices)} hard examples...")
    hard_ds = ds.select(hard_indices)

    # 9. Write filtered parquet
    print(f"Writing hard examples to {args.output}")
    write_parquet_large(hard_ds, args.output)
    print(f"Done! Wrote {len(hard_ds)} hard examples to {args.output}")


if __name__ == "__main__":
    main()
