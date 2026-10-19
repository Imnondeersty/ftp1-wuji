#!/usr/bin/env bash
# Teleop-command action runs (arm / hand / wrist actions from *_cmd, state input still measured), queued until all
# 8 GPUs are free, same recipe as A-F (batch 128, 10k steps, ckpt / 2000):
#   F2 wuji_tianji_ft_v2cmd_taxel_headcam  head cam  + taxels      (compare with F)
#   A2 wuji_tianji_ft_v2cmd                3 cameras + agg forces  (compare with A)
# A2 only starts if F2 wrote its final checkpoint.
#   setsid nohup bash scripts_exp_zarr/wuji/queue_cmd_runs.sh > /dev/null 2>&1 &
set -uo pipefail
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cache_root="/wuji-vepfs/wuji-il/YuanZhen/data/ftp1_cache"
qlog="${cache_root}/queue_cmd.log"

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
run wuji_tianji_ft_v2cmd_taxel_headcam wuji_tianji_v2cmd_taxel_headcam "${dir}/dataset_wuji_taxel_headcam_cmd.json"
run wuji_tianji_ft_v2cmd wuji_tianji_v2cmd "${dir}/dataset_wuji_cmd.json"
echo "[queue] $(date '+%F %T') all done" >> "${qlog}"
