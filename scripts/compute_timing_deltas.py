"""Compute per-step timing deltas from a downloaded wandb run's history.jsonl.

The stream trainer accumulates timing_s/* metrics across inner iterations
(max_updates_per_batch). This script recovers the true per-step wall-clock
time by computing deltas, resetting at each batch boundary.

Usage:
    python scripts/compute_timing_deltas.py wandb_run_download/grpo_stream_bouncingsim_easy/uwaks8q8
    python scripts/compute_timing_deltas.py wandb_run_download/grpo_stream_bouncingsim_easy/uwaks8q8 --updates-per-batch 8
    python scripts/compute_timing_deltas.py wandb_run_download/grpo_stream_bouncingsim_easy/uwaks8q8 --csv output.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_timing_deltas(history_path: str, updates_per_batch: int) -> pd.DataFrame:
    with open(history_path) as f:
        data = [json.loads(line) for line in f]

    all_keys = set()
    for d in data:
        all_keys.update(d.keys())
    timing_keys = sorted(
        k for k in all_keys
        if k.startswith("timing_s/") or k.startswith("timing_per_token_ms/")
    )

    rows = []
    prev = {k: 0.0 for k in timing_keys}
    timing_step_count = 0

    for d in data:
        step = d.get("_step")
        if step is None:
            continue

        has_timing = any(d.get(k) is not None for k in timing_keys)
        if not has_timing:
            continue

        is_batch_start = (timing_step_count % updates_per_batch == 0)
        timing_step_count += 1

        row = {"step": step}
        for k in timing_keys:
            v = d.get(k)
            if v is None:
                row[k] = None
                continue

            if is_batch_start:
                delta = v
            else:
                delta = v - prev.get(k, 0.0)
            prev[k] = v
            row[k] = delta

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute per-step timing deltas from wandb history.")
    parser.add_argument("run_dir", help="Path to downloaded wandb run directory (containing history.jsonl and config.json)")
    parser.add_argument("--updates-per-batch", type=int, default=None,
                        help="Number of inner updates per batch (default: read from config.json max_updates_per_batch)")
    parser.add_argument("--csv", type=str, default="auto",
                        help="Save results to CSV file (default: <run_dir>/timing_deltas.csv, use 'none' to disable)")
    parser.add_argument("--keys", type=str, nargs="+", default=None,
                        help="Only show these timing keys (substring match)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.jsonl"
    config_path = run_dir / "config.json"

    if not history_path.exists():
        raise FileNotFoundError(f"history.jsonl not found in {run_dir}")

    updates_per_batch = args.updates_per_batch
    if updates_per_batch is None:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            updates_per_batch = config.get("trainer", {}).get("max_updates_per_batch", 1)
            print(f"Detected max_updates_per_batch={updates_per_batch} from config.json")
        else:
            updates_per_batch = 1
            print("No config.json found, assuming updates_per_batch=1 (no accumulation)")

    df = compute_timing_deltas(str(history_path), updates_per_batch)

    if args.keys:
        cols = ["step"] + [c for c in df.columns if c != "step" and any(k in c for k in args.keys)]
        df = df[[c for c in cols if c in df.columns]]

    csv_path = args.csv
    if csv_path == "auto":
        csv_path = str(run_dir / "timing_deltas.csv")
    elif csv_path.lower() == "none":
        csv_path = None

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
