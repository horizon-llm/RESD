#!/usr/bin/env python3
"""Gather LiveCodeBench results from per-method JSONL files and print a summary table.

Columns reported (matching process_validation_metrics conventions):
  - prob_acc:  problem-level accuracy (score: sparse, all-or-nothing)
  - tc_acc:    test-case accuracy (acc: partial)
  - response_length: response length in tokens
  - mean@N:    per-problem mean over N samples, averaged across problems
  - best@N:    per-problem max over N samples, averaged across problems

Usage:
    python scripts/gather_lcb_results.py kayleexl_models_livecodebench/
    python scripts/gather_lcb_results.py kayleexl_models_livecodebench/ --format csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(records: list[dict]) -> dict:
    uid_to_samples: dict[str, list[dict]] = defaultdict(list)
    has_response_length = "response_length" in records[0]

    for r in records:
        uid = r.get("input", "")
        uid_to_samples[uid].append(r)

    n_problems = len(uid_to_samples)
    n_per_problem = [len(v) for v in uid_to_samples.values()]
    n = max(n_per_problem) if n_per_problem else 0
    n_samples = sum(n_per_problem)

    prob_acc_means, prob_acc_bests = [], []
    tc_acc_means, tc_acc_bests = [], []
    resp_len_means, resp_len_bests = [], []

    for samples in uid_to_samples.values():
        scores = [float(s["score"]) for s in samples]
        accs = [float(s["acc"]) for s in samples]
        prob_acc_means.append(np.mean(scores))
        prob_acc_bests.append(np.max(scores))
        tc_acc_means.append(np.mean(accs))
        tc_acc_bests.append(np.max(accs))
        if has_response_length:
            lengths = [float(s["response_length"]) for s in samples]
            resp_len_means.append(np.mean(lengths))
            resp_len_bests.append(np.min(lengths))

    metrics = {
        "n_problems": n_problems,
        "n_samples": n_samples,
        "n_per_problem": n,
        f"prob_acc/mean@{n}": np.mean(prob_acc_means) * 100,
        f"prob_acc/best@{n}": np.mean(prob_acc_bests) * 100,
        f"tc_acc/mean@{n}": np.mean(tc_acc_means) * 100,
        f"tc_acc/best@{n}": np.mean(tc_acc_bests) * 100,
    }
    if has_response_length:
        metrics[f"response_length/mean@{n}"] = np.mean(resp_len_means)
        metrics[f"response_length/best@{n}"] = np.mean(resp_len_bests)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Root directory containing per-method subdirectories")
    parser.add_argument("--format", choices=["table", "csv", "tsv"], default="table")
    args = parser.parse_args()

    root = args.results_dir
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue

        jsonl_files = sorted(method_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Warning: no JSONL files in {method_dir.name}, skipping", file=sys.stderr)
            continue

        all_records = []
        for jf in jsonl_files:
            all_records.extend(load_jsonl(jf))

        if not all_records:
            continue

        metrics = compute_metrics(all_records)
        rows.append((method_dir.name, metrics))

    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    all_keys = []
    for _, m in rows:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    header = ["model"] + all_keys

    if args.format == "table":
        col_widths = [max(len(header[0]), max(len(r[0]) for r in rows))]
        for k in all_keys:
            fmt = ".0f" if k.startswith("n_") else ".1f"
            vals = [f"{m[k]:{fmt}}" if k in m else "-" for _, m in rows]
            col_widths.append(max(len(k), max(len(v) for v in vals)))

        def fmt_row(cells):
            return " | ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, col_widths)))

        print(fmt_row(header))
        print("-+-".join("-" * w for w in col_widths))
        for name, m in rows:
            cells = [name]
            for k in all_keys:
                fmt = ".0f" if k.startswith("n_") else ".1f"
                cells.append(f"{m[k]:{fmt}}" if k in m else "-")
            print(fmt_row(cells))
    else:
        sep = "," if args.format == "csv" else "\t"
        print(sep.join(header))
        for name, m in rows:
            cells = [name]
            for k in all_keys:
                fmt = ".0f" if k.startswith("n_") else ".1f"
                cells.append(f"{m[k]:{fmt}}" if k in m else "")
            print(sep.join(cells))


if __name__ == "__main__":
    main()
