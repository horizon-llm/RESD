import argparse
import json
from pathlib import Path

import wandb

parser = argparse.ArgumentParser(description="Download a W&B run's config, history, files, and artifacts.")
parser.add_argument("run_path", help='W&B run path, e.g. "entity/project/run_id"')
parser.add_argument("--out-dir", default="wandb_run_download", help="Base output directory (default: wandb_run_download). Run files are placed in <out-dir>/<project>/<run_name>.")
parser.add_argument("--use-run-id", action="store_true", help="Use run ID instead of run name for the folder name")
args = parser.parse_args()

RUN_PATH = args.run_path
_, project, run_id = RUN_PATH.split("/")

api = wandb.Api()
run = api.run(RUN_PATH)

folder_name = run_id if args.use_run_id else run.name
OUT_DIR = Path(args.out_dir) / project / folder_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 0. Run metadata (name, id, etc.)
#    For resumed runs, we store all run IDs so the provenance is clear.
# -------------------------
meta_path = OUT_DIR / "run_meta.json"
existing_meta = None
if meta_path.exists():
    with open(meta_path) as f:
        existing_meta = json.load(f)

run_meta = {
    "id": run.id,
    "name": run.name,
    "display_name": getattr(run, "display_name", run.name),
    "url": run.url,
    "state": run.state,
    "created_at": run.created_at,
}

if existing_meta:
    prev_ids = existing_meta.get("all_run_ids", [existing_meta["id"]])
    if run.id not in prev_ids:
        prev_ids.append(run.id)
    run_meta["all_run_ids"] = prev_ids
else:
    run_meta["all_run_ids"] = [run.id]

meta_path.write_text(json.dumps(run_meta, indent=2, default=str))

# -------------------------
# 1. Config + summary (use latest run's config)
# -------------------------
(OUT_DIR / "config.json").write_text(
    json.dumps(dict(run.config), indent=2, default=str)
)

(OUT_DIR / "summary.json").write_text(
    json.dumps(dict(run.summary), indent=2, default=str)
)

# -------------------------
# 2. Full raw history as JSONL
#    For resumed runs, merge by appending new steps beyond the existing max step.
# -------------------------
history_path = OUT_DIR / "history.jsonl"

existing_max_step = -1
if history_path.exists():
    with open(history_path) as f:
        for line in f:
            row = json.loads(line)
            s = row.get("_step", -1)
            if s > existing_max_step:
                existing_max_step = s
    print(f"Existing history found (max _step={existing_max_step}), appending new steps.")

mode = "a" if existing_max_step >= 0 else "w"
new_rows = 0
with open(history_path, mode) as f:
    for row in run.scan_history(page_size=10000):
        step = row.get("_step", -1)
        if step > existing_max_step:
            f.write(json.dumps(row, default=str) + "\n")
            new_rows += 1

if existing_max_step >= 0:
    print(f"Appended {new_rows} new rows (steps > {existing_max_step}).")

# -------------------------
# 3. Download all run files
#    This includes things like media files, tables, logs, uploaded text files, etc.
# -------------------------
files_dir = OUT_DIR / "files"
files_dir.mkdir(exist_ok=True)

for file in run.files():
    if file.name.startswith("artifact/"):
        continue
    print("Downloading file:", file.name)
    file.download(root=str(files_dir), replace=True)

# -------------------------
# 4. Download all output artifacts
# -------------------------
artifacts_dir = OUT_DIR / "logged_artifacts"
artifacts_dir.mkdir(exist_ok=True)

for artifact in run.logged_artifacts():
    safe_name = artifact.name.replace(":", "_").replace("/", "_")
    print("Downloading logged artifact:", artifact.name)
    artifact.download(root=str(artifacts_dir / safe_name))

# -------------------------
# 5. Optional: download input / used artifacts too
#    This may include datasets/checkpoints consumed by the run, so it can be huge.
# -------------------------
used_artifacts_dir = OUT_DIR / "used_artifacts"
used_artifacts_dir.mkdir(exist_ok=True)

for artifact in run.used_artifacts():
    safe_name = artifact.name.replace(":", "_").replace("/", "_")
    print("Downloading used artifact:", artifact.name)
    artifact.download(root=str(used_artifacts_dir / safe_name))

print(f"Downloaded run to: {OUT_DIR.resolve()}")
