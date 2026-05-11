#!/usr/bin/env bash
set -euo pipefail

# Aggregate LCBv6 generation metrics for all checkpoints and use each
# checkpoint's own tokenizer for response length calculation.
#
# Usage:
#   bash selfevolve/resd/aggregate_lcbv6_all_with_ckpt_tokenizer.sh
#
# Optional overrides:
#   S3_BASE=s3://... \
#   LOCAL_BASE=checkpoints/... \
#   VAL_BASE=val_generations/... \
#   SYNC_CHECKPOINTS=1 \
#   SKIP_SUMMARY=1 \
#   bash selfevolve/resd/aggregate_lcbv6_all_with_ckpt_tokenizer.sh
#
# Notes:
# - By default, this script does not sync checkpoints from S3.
# - Set SYNC_CHECKPOINTS=1 if you want to auto-download missing checkpoints.

S3_BASE="${S3_BASE:-s3://shopqa-users/kayleexl/models_livecodebench}"
LOCAL_BASE="${LOCAL_BASE:-checkpoints/kayleexl_models_livecodebench}"
VAL_BASE="${VAL_BASE:-val_generations/kayleexl_models_livecodebench}"
SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

CHECKPOINTS=(
    "AdaptThink-1.5B-delta0.01"
    "AdaptThink-1.5B-delta0.05"
    "AdaptThink-1.5B-delta0.075"
    "AdaptThink-1.5B-delta0.1"
    "DRPO-1.5B"
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "GPQA_grpo_380"
    "GPQA_grpo_outcome_p0.5_0.7_1.4_100"
    "GPQA_w0.5_0.3_max0.15_beta1_theta0.2_110"
    "GPQA_w0.5_0.3_max0.15_beta1_theta0.3_300"
    "GPQA_w0.5_0.3_max0.15_beta1_theta0.3_600"
    "GPQA_w0.5_0.3_max0.15_beta1_theta0.5_720"
    "JET-1.5B"
    "LCR1_1.5B"
    "Laser-D-L1024-1.5B"
    "Laser-D-L2048-1.5B"
    "Laser-D-L4096-1.5B"
    "Laser-DE-L1024-1.5B"
    "Laser-DE-L2048-1.5B"
    "Laser-DE-L4096-1.5B"
    "Laser-L8192-1.5B"
    "Thinkprune-2k"
    "Thinkprune-3k"
    "Thinkprune-4k"
    "Thinkprune-iter2k"
    "pen_w0.5-0.3_max0.15_beta1_theta0.2_600"
    "pen_w0.5-0.3_max0.15_beta1_theta0.2_790"
    "pen_w0.5-0.3_max0.15_beta1_theta0.2_950"
    "pen_w0.5-0.3_max0.15_beta1_theta0.3_660"
    "pen_w0.5-0.3_max0.15_beta1_theta0.3_780"
    "pen_w0.5-0.3_max0.15_beta1_theta0.3_910"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python"
fi

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "========================================"
    echo "[loop] Aggregating checkpoint: ${ckpt}"
    echo "========================================"

    ckpt_path="${LOCAL_BASE}/${ckpt}"
    val_dir="${VAL_BASE}/${ckpt}"

    if [[ ! -d "${ckpt_path}" || -z "$(ls -A "${ckpt_path}" 2>/dev/null || true)" ]]; then
        if [[ "${SYNC_CHECKPOINTS}" == "1" ]]; then
            mkdir -p "${ckpt_path}"
            echo "[sync] ${S3_BASE}/${ckpt}/ -> ${ckpt_path}/"
            aws s3 sync "${S3_BASE}/${ckpt}/" "${ckpt_path}/" --region us-east-1
        else
            echo "[warn] Missing local checkpoint: ${ckpt_path}"
            echo "[warn] Set SYNC_CHECKPOINTS=1 to auto-download from S3. Skipping ${ckpt}."
            continue
        fi
    fi

    if [[ ! -d "${val_dir}" ]]; then
        echo "[warn] Missing generation directory: ${val_dir}. Skipping ${ckpt}."
        continue
    fi

    echo "[run] tokenizer=${ckpt_path} path=${val_dir}"
    "${PYTHON}" selfevolve/resd/aggregate_generations.py \
        "${val_dir}" \
        --tokenizer "${ckpt_path}" \
        "$@"

    echo "[done] ${ckpt}"
    echo

done

if [[ "${SKIP_SUMMARY}" == "1" ]]; then
    exit 0
fi

echo "========================================"
echo "[summary] Aggregated checkpoint table"
echo "========================================"

"${PYTHON}" - <<'PY' "${VAL_BASE}" "${CHECKPOINTS[@]}"
import json
import sys
from pathlib import Path

val_base = Path(sys.argv[1])
checkpoints = sys.argv[2:]

rows = []
for ckpt in checkpoints:
    metric_path = val_base / ckpt / "metrics.json"
    if not metric_path.exists():
        continue
    with open(metric_path) as f:
        metrics = json.load(f)
    rows.append((ckpt, metrics))

if not rows:
    print("[warn] No metrics.json found. Run aggregation first.")
    raise SystemExit(0)

key_order = [
    "n_problems",
    "n_samples",
    "n_per_problem",
    "prob_acc/mean@4",
    "prob_acc/best@4",
    "tc_acc/mean@4",
    "tc_acc/best@4",
    "response_length/mean@4",
    "response_length/best@4",
]

present_keys = [k for k in key_order if any(k in m for _, m in rows)]
label_w = max(len(label) for label, _ in rows)
col_w = 12

header = f"{'model':<{label_w}}" + "".join(f"  {k:>{col_w}}" for k in present_keys)
print(header)
print("-" * len(header))

for label, metrics in rows:
    line = f"{label:<{label_w}}"
    for k in present_keys:
        v = metrics.get(k, "")
        if isinstance(v, float):
            line += f"  {v:>{col_w}.4f}"
        else:
            line += f"  {str(v):>{col_w}}"
    print(line)
PY
