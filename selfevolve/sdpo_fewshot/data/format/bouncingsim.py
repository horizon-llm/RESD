"""
preprocess functions for the datasets used in paper "RL Grokking Recipe: How RL Unlocks and Transfers New Algorithms in LLMs"
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
    """Cast to LargeString/LargeList and write many small row groups.

    This avoids 32-bit offset overflow in Arrow arrays by casting to
    LargeString/LargeList and writing smaller row groups.
    """
    tbl: pa.Table = ds.data.table
    tbl = tbl.cast(_large_schema(tbl.schema))  # avoid 32-bit offset overflow
    # DO NOT combine_chunks() here — we want smaller arrays per row group
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


def make_map_fn(split: str, data_source_suffix):
    def process_fn(example, idx):
        messages = example.pop("messages")
        extra_info = {
            "split": split,
            "index": f"{example.pop("id")}",
            "difficulty": example.pop("difficulty"),
            
        }

        return {
            "data_source": example.pop("dataset") + "-" + data_source_suffix,
            "prompt": messages,
            "ability": "bouncingsim",
            "reward_model": {"style": "bouncingsim", "ground_truth": json.dumps(example.pop("ground_truth"))},
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


def run_proprocessing(data_source, data_source_suffix, num_proc=4, num_data=-1):
    print("Train data source: ", data_source)
    print("Test data source: ", data_source)
    train_dataset = datasets.load_dataset(data_source, split="train")
    test_dataset = datasets.load_dataset(data_source, split="test")
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

    out_train = os.path.join(f"selfevolve/sdpo_fewshot/datasets/bouncingsim_{data_source_suffix}", f"train_{num_data}.parquet")
    out_test  = os.path.join(f"selfevolve/sdpo_fewshot/datasets/bouncingsim_{data_source_suffix}", "test.parquet")
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    write_rowgrouped_large(train_ds, out_train)
    write_rowgrouped_large(test_ds, out_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce sorted dataset that can be used for training on most relevant questions."
    )
    parser.add_argument(
        "--data_source", type=str,
        help="HF dataset name."
    )
    parser.add_argument(
        "--num_data", "-n",
        type=int,
        default=-1,
        help="Optional limit on number of training samples to process (default: all).",
    )
    parser.add_argument(
        "--data_source_suffix", type=str,
        default="",
        help="Suffix appended to the dataset name to form the data_source field."
    )
    args = parser.parse_args()

    run_proprocessing(data_source=args.data_source, data_source_suffix=args.data_source_suffix, num_data=args.num_data)
