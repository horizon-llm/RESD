"""
Data preprocessing for competition coding datasets from HuggingFace @competitioncode.
Converts HF datasets into the standard parquet format for VERL training.
"""

import os
import json
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):  return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):  return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):    return pa.field(field.name, pa.large_list(_to_large(pa.field("item", t.value_type)).type),
                                              field.nullable, field.metadata)
    if pa.types.is_struct(t):  return pa.field(field.name,
        pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
        field.nullable, field.metadata)
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    """Cast to LargeString/LargeList and write many small row groups."""
    tbl: pa.Table = ds.data.table
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


def make_map_fn(split: str, data_source_suffix: str):
    def process_fn(example, idx):
        messages = example.pop("messages")
        ground_truth = example.pop("ground_truth")  # list of assertion strings

        extra_info = {
            "split": split,
            "index": f"{example.get('id', idx)}",
            "category": example.get("category", ""),
            "difficulty": example.get("difficulty", ""),
            "problem_name": example.get("problem_name", ""),
        }

        return {
            "data_source": example.get("dataset", "competitioncode") + "-" + data_source_suffix,
            "prompt": messages,
            "ability": "competitioncode",
            "reward_model": {"style": "competitioncode", "ground_truth": json.dumps(ground_truth)},
            "extra_info": extra_info,
        }

    return process_fn


def _map_in_shards(dataset, split: str, num_shards: int, num_proc: int, data_source_suffix: str):
    processed_shards = []
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard = shard.map(function=make_map_fn(split, data_source_suffix), with_indices=True, num_proc=num_proc)
        processed_shards.append(shard)
    return datasets.concatenate_datasets(processed_shards)


def _load_dataset(data_source: str, split_name: str):
    """Load a HF dataset, handling different split naming conventions."""
    try:
        return datasets.load_dataset(data_source, split=split_name)
    except ValueError:
        # Fall back to "default" split (seed datasets use this)
        return datasets.load_dataset(data_source, split="default")


def run_preprocessing(train_data_source, test_data_source, data_source_suffix, num_proc=4, num_data=-1, shuffle=False, seed=42):
    print("Train data source:", train_data_source)
    print("Test data source:", test_data_source)
    train_dataset = _load_dataset(train_data_source, "train")
    test_dataset = _load_dataset(test_data_source, "train")
    if shuffle:
        train_dataset = train_dataset.shuffle(seed=seed)
        test_dataset = test_dataset.shuffle(seed=seed)
    if num_data != -1:
        train_dataset = train_dataset.select(range(min(num_data, len(train_dataset))))

    print(f"Map Datasets {train_dataset.num_rows} train, {test_dataset.num_rows} test")
    num_shards = min(4, (len(train_dataset) // 1000) + 1)
    print(f"Using {num_shards} shards")
    num_proc = min(num_proc, num_shards)
    print(f"Using {num_proc} processes")
    train_ds = _map_in_shards(train_dataset, "train", num_shards=num_shards, num_proc=num_proc, data_source_suffix=data_source_suffix)
    print(train_ds)
    test_ds = _map_in_shards(test_dataset, "test", num_shards=num_shards, num_proc=num_proc, data_source_suffix=data_source_suffix)
    print(test_ds)

    out_train = os.path.join(f"selfevolve/sdpo_fewshot/datasets/competitioncode_{data_source_suffix}", f"train_{num_data}.parquet")
    out_test  = os.path.join(f"selfevolve/sdpo_fewshot/datasets/competitioncode_{data_source_suffix}", "test.parquet")
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    write_rowgrouped_large(train_ds, out_train)
    write_rowgrouped_large(test_ds, out_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess competition coding datasets for VERL training."
    )
    parser.add_argument(
        "--train_data_source", type=str, required=True,
        help="HF dataset name for training (e.g., competitioncode/meet_in_the_middle)."
    )
    parser.add_argument(
        "--test_data_source", type=str, required=True,
        help="HF dataset name for testing (e.g., competitioncode/meet_in_the_middle_seed)."
    )
    parser.add_argument(
        "--num_data", "-n", type=int, default=-1,
        help="Optional limit on number of training samples (default: all).",
    )
    parser.add_argument(
        "--data_source_suffix", type=str, default="",
        help="Suffix appended to the dataset name to form the data_source field."
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=False,
        help="Shuffle the data before processing (default: False)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)."
    )
    args = parser.parse_args()

    run_preprocessing(
        train_data_source=args.train_data_source,
        test_data_source=args.test_data_source,
        data_source_suffix=args.data_source_suffix,
        num_data=args.num_data,
        shuffle=args.shuffle,
        seed=args.seed,
    )
