#!/usr/bin/env bash
# Per-taxel tactile runs, queued until all 8 GPUs are free, same recipe as A-D (batch 128, 10k steps, ckpt / 2000):
#   E wuji_tianji_ft_v1_taxel          3 cameras + taxels   (compare with A / B)
#   F wuji_tianji_ft_v1_taxel_headcam  head cam  + taxels   (compare with C / D)
# F only starts if E wrote its final checkpoint.
#   setsid nohup bash scripts_exp_zarr/wuji/queue_taxel_runs.sh > /dev/null 2>&1 &
set -uo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache_root="/wuji-vepfs/wuji-il/YuanZhen/data/ftp1_cache"
qlog="${cache_root}/queue_taxel.log"

run() {  # run <exp_name> <repo_id> <dataset_config>
  local exp="$1" log="${cache_root}/train_$1.log"
  bash "${dir}/wait_gpus_free.sh" "${qlog}"
  echo "[queue] $(date '+%F %T') starting ${exp} -> ${log}" >> "${qlog}"
  EXP_NAME="${exp}" REPO_ID="$2" DATASET_CONFIG="$3" SAVE_INTERVAL=2000 bash "${dir}/train_wuji.sh" > "${log}" 2>&1
  local rc=$?
  echo "[queue] $(date '+%F %T') ${exp} exited with ${rc}" >> "${qlog}"
  if [ ! -f "${cache_root}/checkpoints/ftp1/${exp}/9999/model.safetensors" ]; then
    echo "[queue] $(date '+%F %T') ${exp} final checkpoint missing; stopping the queue" >> "${qlog}"
    exit 1
  fi
}

echo "[queue] $(date '+%F %T') pid $$" >> "${qlog}"
run wuji_tianji_ft_v1_taxel wuji_tianji_v1_taxel "${dir}/dataset_wuji_taxel.json"
run wuji_tianji_ft_v1_taxel_headcam wuji_tianji_v1_taxel_headcam "${dir}/dataset_wuji_taxel_headcam.json"
echo "[queue] $(date '+%F %T') all done" >> "${qlog}"
