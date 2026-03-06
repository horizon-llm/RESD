import argparse
import json
import os
from typing import Any


def _normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        # Keep deterministic text answer for QA training.
        return " | ".join([str(x).strip() for x in answer if str(x).strip()])
    if answer is None:
        return ""
    return str(answer).strip()


def _build_record(example: dict[str, Any], idx: int) -> dict[str, Any]:
    question = str(example.get("question", "")).strip()
    answer = _normalize_answer(example.get("answer", ""))
    fact = str(example.get("fact", "")).strip()
    return {
        "idx": idx,
        "kind": "qa",
        "dataset": "WikiDYK",
        "prompt": question,
        "answer": answer,
        "elo": "-",
        "test": "-",
        "desription": fact,
    }


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_wikidyk_train(input_path: str, output_dir: str, max_samples: int = 500) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Expected input JSON to be a list of records.")

    if max_samples > 0:
        raw = raw[:max_samples]
    records = [_build_record(example, idx=i) for i, example in enumerate(raw)]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.jsonl")
    _write_jsonl(out_path, records)
    print(f"Loaded {len(raw)} samples from {input_path}")
    print(f"Saved train: {len(records)} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WikiDYK train JSON to SDPO-style train.jsonl.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/hal245/op_distill/WikiDYK/data/wikidyk2022-2025_01082025_gpt-4o_evalv2_pages_formatted_combined_v2_trainqas.json",
        help="Path to WikiDYK train qas JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/hal245/op_distill/my_scripts/sdpo_dataset/WikiDYK",
        help="Directory to save train.jsonl.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum number of training samples to keep. Use -1 for all.",
    )
    args = parser.parse_args()
    convert_wikidyk_train(args.input_path, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
