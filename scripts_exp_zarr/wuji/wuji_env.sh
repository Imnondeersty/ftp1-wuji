# Shared settings for the Wuji/Tianji FTP-1 fine-tuning scripts in this directory.
# Sourced by compute_norm_stats_wuji.sh and train_wuji.sh so that every parameter recorded in
# norm_params_snapshot.json is identical between norm computation and training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ================================ Environment ================================
# `uv` is not on PATH here and the venv's editable install points at a stale /DATA path,
# so call the venv interpreter directly and put the sources on PYTHONPATH.
PYTHON="${REPO_ROOT}/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENPI_DATA_HOME="/wuji-vepfs/wuji-il/YuanZhen/cache/openpi"  # holds big_vision/paligemma_tokenizer.model
export HF_ENDPOINT="https://hf-mirror.com"
export HYDRA_FULL_ERROR=1

# ================================ Paths ================================
# Override both together for a different dataset variant, e.g. head camera only:
#   REPO_ID=wuji_tianji_v1_headcam DATASET_CONFIG=${SCRIPT_DIR}/dataset_wuji_headcam.json
repo_id="${REPO_ID:-wuji_tianji_v1}"
dataset_config_path="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_wuji.json}"
cache_root="/wuji-vepfs/wuji-il/YuanZhen/data/ftp1_cache"
checkpoint_base_dir="${cache_root}/checkpoints"
assets_base_dir="${cache_root}/assets"
pretrained_ckpt="${REPO_ROOT}/ckpts/ftp1_v0426_50kstep"

# ============ Data / normalization (must match between norm stats and training) ============
# Same representation as the pretrained checkpoint (ckpts/ftp1_v0426_50kstep/train_config.json).
action_down_sample_steps=1  # data is 20 Hz -> 32-step action chunk = 1.6 s
proprioception_pose_rep="relative"
action_pose_rep="relative"
proprioception_joint_rep="abs"
action_joint_rep="mix"
norm_type="zscore"
independent_norm_mode="all"
norm_image_tactile_mode="channel_wise"
val_ratio=5  # >= 1 means a fixed number of held-out episodes

COMMON_ARGS=(
  ftp1
  --repo_id="${repo_id}"
  --data.repo-id="${repo_id}"
  --dataset_config_path="${dataset_config_path}"
  --checkpoint_base_dir="${checkpoint_base_dir}"
  --assets_base_dir="${assets_base_dir}"
  --action_down_sample_steps="${action_down_sample_steps}"
  --proprioception_pose_rep="${proprioception_pose_rep}"
  --action_pose_rep="${action_pose_rep}"
  --proprioception_joint_rep="${proprioception_joint_rep}"
  --action_joint_rep="${action_joint_rep}"
  --norm_type="${norm_type}"
  --independent_norm_mode="${independent_norm_mode}"
  --norm_image_tactile_mode="${norm_image_tactile_mode}"
  --use_val_dataset
  --create_train_val_split
  --val_ratio="${val_ratio}"
)
