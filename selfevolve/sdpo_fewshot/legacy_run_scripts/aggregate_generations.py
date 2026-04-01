#!/usr/bin/env python3
"""
Aggregate metrics from saved JSONL generation files produced by test_lcbv6.sh.

Usage:
  # Single file
  python selfevolve/sdpo_fewshot/aggregate_generations.py path/to/0.jsonl

  # Whole checkpoint directory (picks up all *.jsonl inside)
  python selfevolve/sdpo_fewshot/aggregate_generations.py val_generations/kayleexl_models_livecodebench/

  # Glob across all checkpoints, print a summary table
  python selfevolve/sdpo_fewshot/aggregate_generations.py val_generations/kayleexl_models_livecodebench/*/val_generations/0.jsonl
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Per-file aggregation
# ---------------------------------------------------------------------------

def aggregate_file(path: Path, tokenizer=None) -> dict | None:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return None

    # Batch-tokenize all outputs at once if tokenizer is provided
    if tokenizer is not None:
        print(f"[tokenize] encoding {len(rows)} outputs...", file=sys.stderr)
        token_lengths = [
            len(ids) for ids in tokenizer(
                [r["output"] for r in rows],
                add_special_tokens=False,
            )["input_ids"]
        ]
    else:
        token_lengths = None

    # Group samples by problem using a hash of the prompt text.
    # With val_kwargs.n=4 and shuffle=False, consecutive rows share the same
    # prompt, but hashing is safer against any ordering variation.
    uid2accs: dict[str, list[float]] = defaultdict(list)
    uid2lengths: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        uid = hashlib.md5(row["input"].encode()).hexdigest()
        acc = float(row.get("acc", row.get("score", 0.0)))
        uid2accs[uid].append(acc)
        if token_lengths is not None:
            uid2lengths[uid].append(float(token_lengths[i]))

    n_problems = len(uid2accs)
    all_accs = list(uid2accs.values())
    n_samples_per_problem = [len(v) for v in all_accs]
    n = n_samples_per_problem[0] if len(set(n_samples_per_problem)) == 1 else None

    metrics = {
        "n_problems": n_problems,
        "n_samples": len(rows),
        "n_per_problem": n if n is not None else f"uneven ({min(n_samples_per_problem)}-{max(n_samples_per_problem)})",
    }

    # --- prob_acc: binary per problem (1.0 iff ALL test cases passed, i.e. acc == 1) ---
    # --- tc_acc:   raw acc value per sample (proportion of test cases passed) ----------
    #
    # mean@4: mean of the 4 sample values per problem, then mean across problems
    # best@4: max  of the 4 sample values per problem, then mean across problems
    # (mirrors process_validation_metrics in ray_trainer.py)

    for metric_name, per_sample_fn in [
        ("prob_acc", lambda a: 1.0 if a >= 1.0 else 0.0),
        ("tc_acc",   lambda a: a),
    ]:
        vals_mean4 = [np.mean([per_sample_fn(a) for a in accs]) for accs in all_accs]
        vals_best4 = [np.max( [per_sample_fn(a) for a in accs]) for accs in all_accs]
        metrics[f"{metric_name}/mean@4"] = float(np.mean(vals_mean4))
        metrics[f"{metric_name}/best@4"] = float(np.mean(vals_best4))

    if uid2lengths:
        all_lengths = list(uid2lengths.values())
        metrics["response_length/mean@4"] = float(np.mean([np.mean(l) for l in all_lengths]))
        metrics["response_length/best@4"] = float(np.mean([np.max(l)  for l in all_lengths]))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_paths(args: list[str]) -> list[Path]:
    paths = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            found = sorted(p.rglob("*.jsonl"))
            if not found:
                print(f"[warn] no .jsonl files found under {p}", file=sys.stderr)
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            # treat as glob pattern
            import glob
            matched = [Path(x) for x in glob.glob(arg)]
            if not matched:
                print(f"[warn] no match for {arg}", file=sys.stderr)
            paths.extend(sorted(matched))
    return paths


def label_for(path: Path) -> str:
    """Make a short human-readable label from the file path."""
    parts = path.parts
    # Try to use the checkpoint directory name
    for i, part in enumerate(parts):
        if part == "val_generations" and i > 0:
            return parts[i - 1]
    return str(path)


def print_table(rows: list[tuple[str, dict]]) -> None:
    if not rows:
        return

    # Collect all metric keys in a stable order
    key_order = ["n_problems", "n_samples", "n_per_problem",
                 "prob_acc/mean@4", "prob_acc/best@4",
                 "tc_acc/mean@4",   "tc_acc/best@4",
                 "response_length/mean@4", "response_length/best@4"]
    present_keys = []
    for k in key_order:
        if any(k in r for _, r in rows):
            present_keys.append(k)

    label_w = max(len(label) for label, _ in rows)
    col_w = 12

    header = f"{'model':<{label_w}}" + "".join(f"  {k:>{col_w}}" for k in present_keys)
    print(header)
    print("-" * len(header))
    for label, metrics in rows:
        row = f"{label:<{label_w}}"
        for k in present_keys:
            v = metrics.get(k, "")
            if isinstance(v, float):
                row += f"  {v:>{col_w}.4f}"
            else:
                row += f"  {str(v):>{col_w}}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics from generation JSONL files.")
    parser.add_argument("paths", nargs="+", help="Files, directories, or glob patterns")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a table")
    parser.add_argument(
        "--tokenizer",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace tokenizer to use for computing response token lengths "
             "(default: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print(f"[tokenizer] loading {args.tokenizer} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    paths = collect_paths(args.paths)
    if not paths:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in paths:
        label = label_for(path)
        metrics = aggregate_file(path, tokenizer=tokenizer)
        if metrics is None:
            print(f"[warn] {path} is empty, skipping", file=sys.stderr)
            continue
        results.append((label, metrics, path))

        # Save metrics.json next to the .jsonl file
        import json as _json
        out_path = path.parent / "metrics.json"
        with open(out_path, "w") as f:
            _json.dump(metrics, f, indent=2)
        print(f"[saved] {out_path}", file=sys.stderr)

    if args.json:
        import json as _json
        print(_json.dumps({label: m for label, m, _ in results}, indent=2))
    else:
        print_table([(label, m) for label, m, _ in results])


if __name__ == "__main__":
    main()
