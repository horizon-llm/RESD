#!/usr/bin/env bash
set -euo pipefail

# Run test_lcbv6.sh against every checkpoint under
# s3://shopqa-users/kayleexl/models_livecodebench/

S3_BASE="s3://shopqa-users/kayleexl/models_livecodebench"
LOCAL_BASE="checkpoints/kayleexl_models_livecodebench"

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

clear_gpus() {
    echo "[cleanup] Stopping Ray cluster..."
    ray stop --force 2>/dev/null || true

    echo "[cleanup] Killing lingering GPU processes..."
    # Kill any processes still holding NVIDIA device handles
    if command -v fuser &>/dev/null; then
        fuser -k /dev/nvidia* 2>/dev/null || true
    else
        # Fallback: kill via nvidia-smi
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    fi

    # Brief pause to let the driver release memory
    sleep 5

    echo "[cleanup] GPU memory after cleanup:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
}

for ckpt in "${CHECKPOINTS[@]}"; do
    echo "========================================"
    echo "[loop] Testing checkpoint: ${ckpt}"
    echo "========================================"

    local_path="${LOCAL_BASE}/${ckpt}"
    mkdir -p "${local_path}"

    echo "[sync] ${S3_BASE}/${ckpt}/ -> ${local_path}/"
    aws s3 sync "${S3_BASE}/${ckpt}/" "${local_path}/" --region us-east-1

    # These are plain HuggingFace weights, not training checkpoints.
    # Load via model.path override; do NOT set CHECKPOINT_PATH (that triggers
    # trainer.resume_mode=resume_path which requires "global_step_" in the path).
    bash "${SCRIPT_DIR}/test_lcbv6.sh" \
        actor_rollout_ref.model.path="${local_path}" \
        trainer.validation_data_dir="val_generations/kayleexl_models_livecodebench/${ckpt}" \
        "$@"

    clear_gpus
done
