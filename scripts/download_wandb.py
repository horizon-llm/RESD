import argparse
import json
from pathlib import Path

import wandb

parser = argparse.ArgumentParser(description="Download aa W&B run's config, history, files, and artifacts.")
parser.add_argument("run_path", help='W&B run path, e.g. "entity/project/run_id"')
parser.add_argument("--out-dir", default="wandb_run_download", help="Base output directory (default: wandb_run_download). Run files are placed in <out-dir>/<project>/<run_id>.")
args = parser.parse_args()

RUN_PATH = args.run_path
_, project, run_id = RUN_PATH.split("/")
OUT_DIR = Path(args.out_dir) / project / run_id

api = wandb.Api()
run = api.run(RUN_PATH)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. Config + summary
# -------------------------
(OUT_DIR / "config.json").write_text(
    json.dumps(dict(run.config), indent=2, default=str)
)

(OUT_DIR / "summary.json").write_text(
    json.dumps(dict(run.summary), indent=2, default=str)
)

# -------------------------
# 2. Full raw history as JSONL
#    Better than CSV for text, dicts, tables, media refs, etc.
# -------------------------
with open(OUT_DIR / "history.jsonl", "w") as f:
    for row in run.scan_history(page_size=10000):
        f.write(json.dumps(row, default=str) + "\n")

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