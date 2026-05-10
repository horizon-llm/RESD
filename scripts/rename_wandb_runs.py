"""Rename downloaded wandb run folders from run_id to run_name.

Looks up run names via the wandb API and renames directories in place.

Usage:
    python scripts/rename_wandb_runs.py wandb_run_download/grpo_stream_bouncingsim_easy
    python scripts/rename_wandb_runs.py wandb_run_download/grpo_stream_bouncingsim_easy --entity your-entity
"""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description="Rename wandb run folders from run_id to run_name.")
    parser.add_argument("project_dir", help="Path to project directory containing run_id folders")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (default: auto-detect from wandb)")
    parser.add_argument("--dry-run", action="store_true", help="Print renames without executing")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {project_dir}")

    project_name = project_dir.name
    api = wandb.Api()
    entity = args.entity or api.default_entity

    for run_dir in sorted(project_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name

        # Skip if already has run_meta.json with name matching folder
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("name") == run_id:
                print(f"  {run_id} — already named correctly")
                continue
            if meta.get("name"):
                new_name = meta["name"]
                new_path = project_dir / new_name
                if new_path.exists():
                    print(f"  {run_id} — target {new_name} already exists, skipping")
                    continue
                if args.dry_run:
                    print(f"  {run_id} -> {new_name} (dry run)")
                else:
                    run_dir.rename(new_path)
                    print(f"  {run_id} -> {new_name}")
                continue

        # No local meta — fetch from API
        try:
            run = api.run(f"{entity}/{project_name}/{run_id}")
        except Exception as e:
            print(f"  {run_id} — failed to fetch: {e}")
            continue

        new_name = run.name
        if new_name == run_id:
            print(f"  {run_id} — name equals id, skipping")
            continue

        # Save run_meta.json for future use
        run_meta = {
            "id": run.id,
            "name": run.name,
            "display_name": getattr(run, "display_name", run.name),
            "url": run.url,
            "state": run.state,
            "created_at": run.created_at,
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, indent=2, default=str)
        )

        new_path = project_dir / new_name
        if new_path.exists():
            print(f"  {run_id} — target {new_name} already exists, skipping")
            continue

        if args.dry_run:
            print(f"  {run_id} -> {new_name} (dry run)")
        else:
            run_dir.rename(new_path)
            print(f"  {run_id} -> {new_name}")


if __name__ == "__main__":
    main()
