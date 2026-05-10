#!/usr/bin/env python3
"""Gather IFEval results from per-checkpoint JSONL files and print a summary table.

Columns reported:
  - strict_prompt_acc: prompt-level strict accuracy (all instructions followed)
  - loose_prompt_acc:  prompt-level loose accuracy
  - strict_inst_acc:   instruction-level strict accuracy
  - loose_inst_acc:    instruction-level loose accuracy

Usage:
    python scripts/gather_ifeval_results.py checkpoints/ifeval_eval/
    python scripts/gather_ifeval_results.py checkpoints/ifeval_eval/ --format csv
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

    for r in records:
        uid = r.get("input", "")
        uid_to_samples[uid].append(r)

    n_problems = len(uid_to_samples)
    n_per_problem = [len(v) for v in uid_to_samples.values()]
    n = max(n_per_problem) if n_per_problem else 0
    n_samples = sum(n_per_problem)

    strict_prompt_means, strict_prompt_bests = [], []
    loose_prompt_means, loose_prompt_bests = [], []
    strict_inst_means, strict_inst_bests = [], []
    loose_inst_means, loose_inst_bests = [], []

    for samples in uid_to_samples.values():
        sp = [float(s.get("strict_prompt_acc", s.get("acc", 0))) for s in samples]
        lp = [float(s.get("loose_prompt_acc", 0)) for s in samples]
        si = [float(s.get("strict_inst_acc", s.get("score", 0))) for s in samples]
        li = [float(s.get("loose_inst_acc", 0)) for s in samples]

        strict_prompt_means.append(np.mean(sp))
        strict_prompt_bests.append(np.max(sp))
        loose_prompt_means.append(np.mean(lp))
        loose_prompt_bests.append(np.max(lp))
        strict_inst_means.append(np.mean(si))
        strict_inst_bests.append(np.max(si))
        loose_inst_means.append(np.mean(li))
        loose_inst_bests.append(np.max(li))

    metrics = {
        "n_problems": n_problems,
        "n_samples": n_samples,
        "n_per_problem": n,
        f"strict_prompt/mean@{n}": np.mean(strict_prompt_means) * 100,
        f"strict_prompt/best@{n}": np.mean(strict_prompt_bests) * 100,
        f"loose_prompt/mean@{n}": np.mean(loose_prompt_means) * 100,
        f"loose_prompt/best@{n}": np.mean(loose_prompt_bests) * 100,
        f"strict_inst/mean@{n}": np.mean(strict_inst_means) * 100,
        f"strict_inst/best@{n}": np.mean(strict_inst_bests) * 100,
        f"loose_inst/mean@{n}": np.mean(loose_inst_means) * 100,
        f"loose_inst/best@{n}": np.mean(loose_inst_bests) * 100,
    }

    return metrics


def find_jsonl_files(root: Path) -> list[tuple[str, list[Path]]]:
    """Walk root and find directories containing JSONL files, returning (label, files) pairs."""
    results = []
    for d in sorted(root.rglob("*")):
        if not d.is_dir():
            continue
        jsonl_files = sorted(d.glob("*.jsonl"))
        if jsonl_files:
            label = str(d.relative_to(root))
            results.append((label, jsonl_files))
    if not results:
        jsonl_files = sorted(root.glob("*.jsonl"))
        if jsonl_files:
            results.append((root.name, jsonl_files))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Root directory containing result subdirectories")
    parser.add_argument("--format", choices=["table", "csv", "tsv"], default="table")
    args = parser.parse_args()

    root = args.results_dir
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    entries = find_jsonl_files(root)
    if not entries:
        print("No JSONL result files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for label, jsonl_files in entries:
        all_records = []
        for jf in jsonl_files:
            all_records.extend(load_jsonl(jf))
        if not all_records:
            continue
        metrics = compute_metrics(all_records)
        rows.append((label, metrics))

    if not rows:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    all_keys = []
    for _, m in rows:
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    header = ["checkpoint"] + all_keys

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
