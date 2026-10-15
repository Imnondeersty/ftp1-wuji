#!/usr/bin/env bash
# Step 2: fine-tune FTP-1 (ftp1_v0426_50kstep) on the Wuji domain. Run compute_norm_stats_wuji.sh first.
#   bash scripts_exp_zarr/wuji/train_wuji.sh [extra trainer args, appended last so they override]
# Overrides via env, e.g.: CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=64 EXP_NAME=wuji_try2 bash ...
# Logging: off by default; export SWANLAB_API_KEY=... to log to SwanLab.
# Checkpoints: ${checkpoint_base_dir}/ftp1/${EXP_NAME}/<step>/ (~40 GB each incl. optimizer state).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wuji_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exp_name="${EXP_NAME:-wuji_tianji_ft_v1}"
batch_size="${BATCH_SIZE:-128}"  # global; must be divisible by the number of GPUs
num_train_steps="${NUM_TRAIN_STEPS:-10000}"
save_interval="${SAVE_INTERVAL:-5000}"

TRAIN_ARGS=(
  "${COMMON_ARGS[@]}"
  --exp_name="${exp_name}"
  --pytorch_weight_path="${pretrained_ckpt}"
  # model: same as the pretrained checkpoint
  --model.state_input_mode=adarms
  --model.tactile_expert_variant=gemma_small
  --model.use_tactile_input
  # optimization (UniVTAC fine-tuning recipe)
  --batch_size="${batch_size}"
  --num_train_steps="${num_train_steps}"
  --lr_schedule.warmup_steps=500
  --lr_schedule.peak_lr=5e-5
  --lr_schedule.decay_lr=5e-6
  --lr_decay_till_end
  --optimizer.b1=0.9
  --optimizer.b2=0.95
  --optimizer.eps=1e-8
  --optimizer.weight_decay=1e-10
  --optimizer.clip_gradient_norm=1.0
  --ema_decay=0.99
  --pytorch_training_precision=bfloat16
  --use_torch_compile
  --no-gradient_checkpointing_enable
  # bookkeeping
  --log_interval=50
  --val_interval=1000
  --save_interval="${save_interval}"
  --keep_period="${save_interval}"
  --num_workers=8
  --val_num_workers=2
)
if [ -n "${SWANLAB_API_KEY:-}" ]; then
  export USE_SWANLAB=true SWANLAB_SAVE_DIR="${HOME}/.swanlab" SWANLAB_LOG_DIR="${HOME}/swanlog/${exp_name}"
  mkdir -p "${SWANLAB_SAVE_DIR}" "${SWANLAB_LOG_DIR}"
  TRAIN_ARGS+=(--wandb_enabled)
else
  TRAIN_ARGS+=(--no-wandb_enabled)
fi
TRAIN_ARGS+=("$@")

num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
if [ "${num_gpus}" -gt 1 ]; then
  "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
    scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
else
  "${PYTHON}" scripts/zarr_train_ftp1_pytorch.py "${TRAIN_ARGS[@]}"
fi
