#!/usr/bin/env bash
# Step 1: compute normalization stats + train/val split for the Wuji domain.
#   bash scripts_exp_zarr/wuji/compute_norm_stats_wuji.sh
# Output: ${assets_base_dir}/ftp1/${repo_id}/WujiTianji/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wuji_env.sh"

"${PYTHON}" scripts/zarr_compute_norm_stats.py "${COMMON_ARGS[@]}" \
  --exp_name=wuji_norm \
  --no-wandb_enabled \
  --batch_size=256 \
  --norm_sample_ratio=1.0 \
  --norm_batch_size=64 \
  --norm_num_workers=16
