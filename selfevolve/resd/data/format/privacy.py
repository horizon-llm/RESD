import argparse
import json
import os
from typing import Any

from datasets import load_dataset


def _parse_dataset_item(raw_item: Any) -> dict[str, Any]:
    if isinstance(raw_item, dict):
        return raw_item
    if isinstance(raw_item, str):
        return json.loads(raw_item)
    raise ValueError(f"Unsupported dataset_item type: {type(raw_item).__name__}")


def _format_information(info: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in info.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _build_prompt(user_task: str, information: dict[str, Any]) -> str:
    return (
        "You are an assistant helping the user complete a task.\n"
        "Important: Protect the user's privacy.\n"
        "- Only include information that is necessary to complete the task.\n"
        "- Do NOT reveal sensitive or unrelated personal information "
        "(e.g., document IDs, medical notes, insurance numbers, or private chats).\n"
        "- If such information appears in the context, ignore it unless absolutely required.\n\n"
        f"User Task:\n{user_task.strip()}\n\n"
        f"Information:\n{_format_information(information)}"
    )


def _build_record(example: dict[str, Any], idx: int) -> dict[str, Any]:
    item = _parse_dataset_item(example.get("dataset_item"))
    user_task = str(item.get("user_task", "")).strip()
    information = item.get("information", {})
    annotation = item.get("annotation", {})
    if not isinstance(information, dict):
        information = {}

    return {
        "idx": idx,
        "kind": "privacy",
        "dataset": "contextualIntegritySynthetic",
        "prompt": _build_prompt(user_task, information),
        "answer": json.dumps(annotation, ensure_ascii=False),
        "elo": "-",
        "tests": "-",
        "description": "-",
    }


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_privacy(output_dir: str, train_size: int = 500, test_size: int = 100) -> None:
    ds = load_dataset("huseyinatahaninan/ContextualIntegritySyntheticDataset", split="train")
    records = [_build_record(dict(example), idx=i) for i, example in enumerate(ds)]
    if len(records) < train_size + test_size:
        raise ValueError(
            f"Not enough samples: {len(records)} < train_size({train_size}) + test_size({test_size})"
        )

    test_records = records[:test_size]
    train_records = records[test_size:test_size + train_size]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)

    print(f"Loaded {len(ds)} samples from huseyinatahaninan/ContextualIntegritySyntheticDataset")
    print(f"Saved train: {len(train_records)} -> {train_path}")
    print(f"Saved test : {len(test_records)} -> {test_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ContextualIntegritySyntheticDataset to SDPO-style privacy train.jsonl."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/hal245/op_distill/my_scripts/sdpo_dataset/privacy",
        help="Directory to save converted train.jsonl",
    )
    parser.add_argument("--train_size", type=int, default=500, help="Number of samples in train split.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of samples in test split.")
    args = parser.parse_args()
    convert_privacy(args.output_dir, train_size=args.train_size, test_size=args.test_size)


if __name__ == "__main__":
    main()
