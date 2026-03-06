import argparse
import json
import os
from typing import Any

from datasets import load_dataset

def _format_options(options: Any) -> str:
    """Format MedQA options into a readable multi-line string."""
    if isinstance(options, dict):
        # Prefer canonical MCQ order.
        preferred = ["A", "B", "C", "D"]
        keys = [k for k in preferred if k in options]
        if not keys:
            keys = sorted(options.keys(), key=lambda x: str(x))
        return "\n".join([f"{k}: {options[k]}" for k in keys])
    if isinstance(options, list):
        return "\n".join([f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)])
    return str(options)


def _normalize_answer(answer_idx: Any) -> str:
    if isinstance(answer_idx, str):
        value = answer_idx.strip().upper()
        if value in {"A", "B", "C", "D"}:
            return value
        if value.isdigit():
            n = int(value)
            if 0 <= n <= 3:
                return chr(65 + n)
            if 1 <= n <= 4:
                return chr(64 + n)
    if isinstance(answer_idx, int):
        if 0 <= answer_idx <= 3:
            return chr(65 + answer_idx)
        if 1 <= answer_idx <= 4:
            return chr(64 + answer_idx)
    return str(answer_idx)


def _build_prompt(question: str, options: Any) -> str:
    question = "" if question is None else str(question).strip()
    options_block = _format_options(options)
    return question + "\n\n" + options_block + "\nPlease reason step by step."


def _build_record(example: dict[str, Any], idx: int) -> dict[str, Any]:
    question = example.get("question", "")
    options = example.get("options", {})
    answer = _normalize_answer(example.get("answer_idx", "-"))
    return {
        "idx": idx,
        "kind": "mcq",
        "dataset": "MedQA",
        "prompt": _build_prompt(question, options),
        "answer": answer,
        "elo": '-',
        "tests":'-',
        "description": question,
    }


def _sample_split(split_name: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split_name)
    if sample_size > len(ds):
        raise ValueError(f"Requested {sample_size} from {split_name}, but only {len(ds)} available.")
    ds = ds.shuffle(seed=seed).select(range(sample_size))
    return [dict(row) for row in ds]


def _write_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_medqa(output_dir: str, train_size: int = 500, test_size: int = 100, seed: int = 42) -> None:
    train_raw = _sample_split("train", train_size, seed)
    test_raw = _sample_split("test", test_size, seed)

    train_records = [_build_record(ex, idx=i) for i, ex in enumerate(train_raw)]
    train_num=len(train_records)
    test_records = [_build_record(ex, idx=i+train_num) for i, ex in enumerate(test_raw)]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)

    print(f"Saved train: {len(train_records)} -> {train_path}")
    print(f"Saved test : {len(test_records)} -> {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MedQA dataset in SDPO format.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/hal245/op_distill/my_scripts/sdpo_dataset/health",
        help="Directory to save train.jsonl and test.jsonl",
    )
    parser.add_argument("--train_size", type=int, default=500, help="Number of train samples to keep.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    convert_medqa(
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
