import argparse
import json
import os
from typing import Any


def _build_record(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example.get("prompt", "")
    return {
        "idx": example.get("key"),
        "kind": "IF",
        "dataset": "IFeval",
        "answer": {
            "prompt": prompt,
            "instruction_id_list": example.get("instruction_id_list", []),
            "kwargs": example.get("kwargs", []),
        },
        "elo": "-",
        "prompt": prompt,
        "tests": "-",
        "description": "-",
    }


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _write_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_ifeval(input_path: str, output_dir: str, test_size: int = 100) -> None:
    raw = _read_jsonl(input_path)
    records = [_build_record(x) for x in raw]

    # Deterministic split: first `test_size` examples for test, remainder for train.
    test_records = records[:test_size]
    train_records = records[test_size:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)

    print(f"Loaded {len(records)} samples from {input_path}")
    print(f"Saved train: {len(train_records)} -> {train_path}")
    print(f"Saved test : {len(test_records)} -> {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IFEval jsonl to SDPO IF jsonl format.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/hal245/op_distill/IFeval/instruction_following_eval/data/input_data.jsonl",
        help="Path to IFEval input_data.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/hal245/op_distill/SDPO/datasets/ifeval",
        help="Directory to save train.jsonl/test.jsonl",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of samples in test split.",
    )
    args = parser.parse_args()
    convert_ifeval(args.input_path, args.output_dir, args.test_size)


if __name__ == "__main__":
    main()
