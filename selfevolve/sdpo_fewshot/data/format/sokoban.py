"""Preprocess sokoban dataset with curriculum sorting by solution length."""
import os
import argparse
import datasets
import pyarrow as pa
import pyarrow.parquet as pq


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
    ds = ds.flatten_indices()
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess sokoban dataset with curriculum sorting.")
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Path to input parquet file or directory containing train.parquet")
    parser.add_argument("--num_data", "-n", type=int, default=-1,
                        help="Limit on number of training samples (default: all)")
    parser.add_argument("--sort_by_solution_length", action="store_true",
                        help="Sort examples by ground-truth solution length (ascending) for curriculum learning")
    args = parser.parse_args()

    input_parquet = args.input_parquet
    if os.path.isdir(input_parquet):
        input_parquet = os.path.join(input_parquet, "train.parquet")

    print(f"Loading {input_parquet}")
    ds = datasets.load_dataset("parquet", data_files=input_parquet, split="train")
    print(f"Original size: {len(ds)}")

    if args.sort_by_solution_length:
        sol_lens = [len(row["ground_truth"]) for row in ds["reward_model"]]
        sorted_indices = sorted(range(len(sol_lens)), key=lambda i: sol_lens[i])
        ds = ds.select(sorted_indices)
        sorted_lens = [sol_lens[i] for i in sorted_indices]
        print(f"Sorted by solution length: {sorted_lens[0]} .. {sorted_lens[-1]} (mean={sum(sorted_lens)/len(sorted_lens):.1f})")

    if args.num_data > 0 and args.num_data < len(ds):
        ds = ds.select(range(args.num_data))
    print(f"Final size: {len(ds)}")

    base, ext = os.path.splitext(input_parquet)
    out_path = f"{base}_{args.num_data}{ext}"
    write_rowgrouped_large(ds, out_path)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
