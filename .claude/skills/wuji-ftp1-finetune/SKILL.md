---
name: wuji-ftp1-finetune
description: Fine-tune FTP-1 on Wuji hand 2 + Tianji arm teleop data (ROS2 MCAP from HF Inmodeersty/wuji-tactile) - conversion to FTP-1 zarr, norm stats, training launch, and known pitfalls of this machine/repo.
---

# Wuji / Tianji -> FTP-1 fine-tuning

## Pipeline
1. Raw data (read-only): `/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/` (HF private `Inmodeersty/wuji-tactile`,
   download via `HF_ENDPOINT=https://hf-mirror.com`; huggingface.co is unreachable). Only `ros2_bags/` are usable;
   `ros2_bags_fail/` = failed or tactile-drift episodes.
2. Convert: `.venv/bin/python -I /wuji-vepfs/wuji-il/YuanZhen/data/_tools/convert_wuji_to_zarr.py [--keep-left]`
   -> `/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr/WujiTianji/wuji_task{2,3}+<instruction>+.zarr`
   + `conversion_report.json` (per-bag frames, stream gaps, spike counts). FK lives in `data/_tools/tianji_fk.py`.
   `data/_tools/` is outside the repo (`data/` is gitignored); `scripts_exp_zarr/wuji/data_tools/` is a snapshot of its
   `*.py` (2026-10-15, absolute paths unchanged; still needs `data/_pylibs` and the URDFs in
   `wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/`). Edit one and copy to the other to keep them equal.
3. Norm stats: `bash scripts_exp_zarr/wuji/compute_norm_stats_wuji.sh`
4. Train: `bash scripts_exp_zarr/wuji/train_wuji.sh` (env: `CUDA_VISIBLE_DEVICES`, `BATCH_SIZE`, `NUM_TRAIN_STEPS`,
   `SAVE_INTERVAL`, `EXP_NAME`, `SWANLAB_API_KEY`; extra args are appended and override).
   Shared settings in `scripts_exp_zarr/wuji/wuji_env.sh`; outputs under `/wuji-vepfs/wuji-il/YuanZhen/data/ftp1_cache/`.

## Conversion conventions (v1)
- 20 Hz uniform grid over the overlap of all used streams (header.stamp); joints/tactile linear interp, images nearest.
- Images: centre square crop -> 224x224 RGB; `camera_ego_rgb` = head cam, plus both wrist cams.
- Arm: bag `/{side}_arm/joint_states` is in degrees, order = URDF `Joint1..7_{R,L}`; stored as rad.
  `{side}_wrist_pose` = FK of `{side}_palm_link` (URDF `wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/arm_{side}.urdf`)
  rotated by Rz(-90deg) into the FTP-1 wrist frame (Z to fingertips, Y palmar, X right), in the arm base frame
  (no `camera_ego_pose`; fine because pose reps are relative).
- Hand: 20 joints (rad). Per finger joint1..4 = thumb: cmc_flex, cmc_abd, mcp, ip / others: mcp_flex, mcp_abd, pip, dip.
  `hand_joints_idx = [1,26,2,3, 7,6,8,9, 12,11,13,14, 17,16,18,19, 22,21,23,24]`.
- Tactile (plan A): per-fingertip aggregate force `agg_f{x,y,z}` (N) -> `{side}_tactile_data_fingertips (T,5,3)`,
  type `state`, sensor `WujiHand2`, areas `[0..4]` (loader adds +24 for left). Per-taxel `fx,fy,fz` (fixed-scale,
  dimensionless) are the separate taxel variant below. README spike rule applied at 100 Hz before resampling.
- Cleaning result (`conversion_report.json`): all 102 bags in `ros2_bags/` kept (task2 35 / 11377 frames, task3 67 /
  17835 frames, 24 min at 20 Hz); the 85 `ros2_bags_fail/` bags (7 operator-failed + 78 tactile zero drift) are not
  used. Spike rule dropped 444 tactile samples at 100 Hz (~0.06%; index finger 353, in 26 bags). Worst stream gaps
  0.033 s for cameras; 2 task3 bags (`rosbag_20261008_165457`, `_170814`) have 0.18 s hand / 0.13 s tactile gaps
  (interpolated over). The 78 drift bags have good vision/joints and could feed a no-tactile run or a
  drift-corrected tactile variant (not done).
- Only the right arm/hand is exported by default: task2 left side is static and its left tactile firmware (v2.5.1) is
  broken; task3 left arm moves ~10-15 deg but the left hand never makes contact. Use `--keep-left` to export it.

## Training settings (must match pretrained `ckpts/ftp1_v0426_50kstep`)
`state_input_mode=adarms`, `tactile_expert_variant=gemma_small`, pose reps relative/relative, joint reps abs/mix,
`norm_type=zscore`, `independent_norm_mode=all`. Single domain `WujiTianji` (both tasks, separated by instruction);
5 held-out val episodes. Default 8 GPUs, batch 128, 10k steps, lr 5e-5 (warmup 500, cosine to 5e-6), EMA 0.99.
Checkpoint = ~21 GB (model + optimizer); no automatic cleanup of old checkpoints.

## Runs and ablations (2026-10-12)
- A `wuji_tianji_ft_v1`: full FTP-1 SFT with tactile, ckpt every 2000 steps (final = step 9999).
- B `wuji_tianji_ft_v1_notactile`: identical except `--model.no-use-tactile-input` (no tactile encoder/expert;
  checkpoint tactile keys are reported as "unexpected" and skipped). Queued with
  `scripts_exp_zarr/wuji/queue_after.sh <pid> <A final ckpt dir> <log> -- bash train_wuji.sh ...`, which only starts
  B if A's final checkpoint exists. A vs B is the tactile ablation; compare on the real robot with matched layouts.
- Results (both finished cleanly, ~4h45m each on 8 GPUs): val rmse_total A 0.0290 / B 0.0295 at step 9999
  (best A 0.0288 at 8000); plateau after ~5k steps, no overfitting. Tactile mainly speeds early convergence
  (step 1000: A 0.0349 vs B 0.0389); the final 1.7% gap on 5 val episodes is within noise -> needs real-robot test.
  Kept checkpoints: 6000/8000/9999 for all four runs A-D (2000/4000 deleted 2026-10-13; `/wuji-vepfs/wuji-il` is ~100%
  full). Total 254 GB.
- Offline open-loop eval (2026-10-13, step 9999, 330 val frames, 2 GPUs, ~15 min):
  `scripts_exp_zarr/wuji/offline_eval.py --ckpt-a <A> --ckpt-b <B> --out <dir>` -> summary.json + per-episode plots.
  Mean error over the chunk: wrist 10.7 / 11.0 mm (A/B), rot 2.4 / 2.5 deg, arm 1.12 / 1.15 deg, hand 0.99 / 0.99 deg;
  "hold current pose" baseline 48.6 mm / 12.6 / 5.4 / 3.5 deg. **A is ~insensitive to tactile**: zeroing the tactile
  input moves A's chunk by 0.014 deg (arm) vs 0.22 deg for just resampling the flow noise (contact frames 0.027 vs 0.18).
  So expect A ~= B on the robot for these two tasks unless training forces tactile use, e.g. `--non_tactile_dropout_ratio`
  (masks images + state on contact frames; needs contact thresholds -> recompute norm stats with
  `--no-skip_contact_detection`) or evaluation under visual occlusion.
- Head-camera-only ablation (user request, 2026-10-13): wrist cams give the policy too much contact information, so
  C `wuji_tianji_ft_v1_headcam` (tactile) and D `wuji_tianji_ft_v1_headcam_notactile` use
  `/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr-headcam/` (byte-identical copy of v1 minus the two
  `*_wrist_camera_rgb` arrays; zarr listing order preserved), `REPO_ID=wuji_tianji_v1_headcam`,
  `DATASET_CONFIG=scripts_exp_zarr/wuji/dataset_wuji_headcam.json` (env overrides in wuji_env.sh), and A's
  `train_val_split.json` copied in before computing norm stats, so the split and norm stats match A/B exactly.
  Dropping images is done by removing keys, not by masking: the model uses whatever image keys the sample has.
  One image instead of three: ~1.2 it/s (vs 0.62), ~43 GB/GPU. Deployment for C/D must pass only `camera_ego_rgb_0`.
- Tactile sensitivity A vs C, step 9999, same 40 val frames (20 contact), CPU, `scripts_exp_zarr/wuji/tactile_sensitivity.py`
  (~17 s/frame/variant on CPU): zeroing tactile on contact frames moves the chunk by arm 0.027 / hand 0.022 deg (A)
  vs 0.044 / 0.055 deg (C); resampling the flow noise moves it ~0.15-0.25 deg for both. Removing the wrist cams
  roughly doubles tactile sensitivity but it stays well below the noise level; open-loop error is unchanged by
  zeroing tactile (C hand 0.83 -> 0.86 deg on contact frames). Largest effect: task2 ep8 t=173 (holding the bottle,
  release phase): C hand 0.39 deg (> its noise 0.33), A 0.06. C final val rmse vs A: +10-20% (head cam only).
- All four runs finished (D 2026-10-13 10:45). Offline eval C vs D (`offline_eval_headcam_9999/`), mean chunk error
  arm/hand deg, all 330 frames: A 1.12/0.99, B 1.15/0.99, C 1.28/1.07, D 1.31/1.08; contact frames C 1.26/1.04 vs
  D 1.28/1.06 (C wrist pos 14.1 mm vs D 13.8 mm). Tactile adds ~nothing offline with or without wrist cams; the
  wrist cams are worth ~10-15%. Head-only latency ~260 ms. Next step: real-robot A/B/C/D or force tactile use.
- Waiting on `queue_taxel.log` with `grep -q "<exp> exited"` matches stale lines from earlier attempts; count the
  matches before waiting and wait for the count to grow.
- `queue_after.sh` must be given the PID of the `torch.distributed.run` process (or the `train_wuji.sh` bash), not the
  `setsid nohup` wrapper PID, which exits immediately and would start the queued job at once. Don't `pkill -f` with a
  pattern that also matches your own shell command line.
- Deployment I/O: `scripts_exp_zarr/wuji/wuji_policy_io.py` (image crop/resize, 120-d state, tactile kwargs,
  mix-rep deltas -> absolute joint targets). offline_eval checks it bit-exact against training samples (state, tactile,
  targets), and the wrapper's normalization/denormalization matches the dataset to 1e-7. With relative pose reps and no
  history the state's wrist pose is always identity, so deployment needs no FK; actions are applied in joint space.
  `wrapper.infer()` latency on A800, bf16, 10 flow steps, no compile: ~295 ms (A and B) for a 1.6 s chunk.
- How the two offline metrics differ (both open-loop): the trainer's validation (every `val_interval`, all 6 runs)
  samples the val loader in fixed order (8 ranks x 10 batches x 16 = 1280 of the 1322 val frames, same frames for
  every run), one flow sample with unseeded noise, de-normalized RMSE per action group; `rmse_total` mixes m, 6-D
  rotation entries and rad, so use it only to rank runs. `offline_eval.py` goes through the deployment path, every
  4th val frame (330), seeded noise shared by both checkpoints, errors in mm/deg over real future frames only, plus a
  hold-current-pose baseline. offline_eval was run for A/B and C/D only, not E/F.
- `--ema_decay` is ignored: the PyTorch trainer logs "EMA is not supported for PyTorch training", so every
  checkpoint holds raw weights.
- Note: near episode ends the dataset pads the chunk by repeating the last action AND its mask, so count valid future
  frames from the episode end, not from `action_mask`.
- CPU pre-flight for config changes without touching GPUs: `scripts_exp_zarr/wuji/check_notactile_cpu.py`
  (needs `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`; ~80 s). 7 params get no grad in both variants (VLM layer-17
  o_proj/MLP/post-attn norm, VLM final norm, action-expert lm_head): expected, the action expert only reads VLM KV.

## Real-robot deployment (`deploy/`, modelled on OpenWAM-Official openwam/deploy + benchmarks/utils)
- GPU side: `bash deploy/serve.sh <ckpt_step_dir> [--port 8848 --device cuda:0]` -> `deploy/policy_server.py`
  (msgpack over WebSocket, contract in `deploy/protocol.py`). Camera set is read from the checkpoint's dataset
  config (A/B: head + both wrists, C/D: head only); tactile on/off from the checkpoint. Server does all
  preprocessing via `deploy/wuji_policy_io.py` and returns ABSOLUTE right arm (deg) / hand (rad) targets, 32 x 50 ms.
- Files the server reads from a step dir: `model.safetensors` (7.6 GB, full weights; the pretrained base ckpt is not
  needed), `model_config.json`, `train_config.json` (norm settings), `tactile_input_config_file.json`,
  `hpt_tokenizer/`, `normalization/` (~7.7 GB total). `optimizer.pt` (15 GB) and `metadata.pt` are training-only.
  Also needs `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` (4 MB, else fetched from gs://).
  On another machine pass `--cameras` (A/B/E: `head right_wrist left_wrist`, C/D/F: `head`): auto-detection reads
  `train_config.dataset_config_path` and globs the training zarrs, which only exist here.
- huggingface.co is DNS-poisoned here (fake IPs, TCP 443 times out); hf-mirror is download-only, so model uploads to
  HF must go from another machine or through a proxy. modelscope.cn is reachable.
- Robot side (thin, no torch): `deploy/client/ros_node.py --server ws://<gpu>:8848 --task task2|task3 [--execute]`.
  Dry run by default. Subscribes joint_states / cameras / tactile PointCloud2, publishes `/right_{arm,hand}/
  joint_commands` (JointState, arm deg / hand rad, names, BEST_EFFORT depth 1, CLOCK_REALTIME stamp, 120 Hz, hand
  velocity like teleop's hand_interface). Async chunk executor (horizon 14, lead 6, 0.1 s cross-fade) + safety filter
  (arm: URDF ∩ data envelope ±10 deg; hand: data envelope ±0.15 rad; speed <= data max; stop on cmd/state deviation).
- wuji-dc `roles/inference/README.md`: inference is NOT an allowed control source until the robot side has an
  owner/lease arbiter; `--execute` only with the robot owner's sign-off, teleop output stopped, operator at e-stop.
- Offline checks (no robot): `python -m pytest deploy/client/client_test.py` (executor, safety, spike filter ==
  converter rule) and `deploy/replay_bag.py --server ... --bag <val bag> --task ... --out ... [--sim-latency 0.3]`
  (full client stack on a recorded val bag through a live server). 2026-10-13, CPU server: A on task3
  rosbag_20261008_165210 chunk err arm 1.05 / hand 0.96 deg; D on task2 rosbag_20260929_182121 1.39 / 1.16 deg;
  no stops, max command step 0.86 deg / 0.018 rad per 8.3 ms tick. Val episodes (global index, task2 = 0-34,
  task3 = 35-101, bags in sorted order): ep8 = task2 bag 9 = rosbag_20260929_182121; ep44/65/76/101 = task3 bags
  10/31/42/67 (1-based), e.g. ep44 = rosbag_20261008_165210, ep101 = rosbag_20261008_171913.
- The robot has **Wuji hand 2**; `tianji_full/arm_*.urdf` carries the **gen-1** hand (4/20 recorded joints exceed its
  limits, 0/20 exceed `wuji-description/hand2_beta`). Deployment does not use the URDF. It only affects v1
  `*_wrist_pose` (flange -> docking -> palm_link offsets); gen-2 `r_wrist` uses the same axis convention
  (Z distal, Y thumb side, X palmar). No URDF with Tianji + gen-2 mount was found; flange -> hand2 base transform needed.

## Per-taxel tactile variant (2026-10-13)
- `data/_tools/convert_wuji_taxel.py [--drop-wrist-cams --out ...]` derives it from the v1 zarr + raw bags (~2 min,
  v1 converter untouched): copies every non-tactile array byte for byte, replaces `*_fingertips` with
  `right_tactile_*_taxelthumb (T,1,120)` area 0 + `right_tactile_*_taxelfingers (T,4,102)` areas 1-4, sensor
  `WujiHand2Taxel`, type state. It recomputes the grid from bag stamps and asserts the re-derived agg forces equal v1
  bit-exact. Outputs: `data/wuji-tactile-zarr-taxel/` (3 cams) and `data/wuji-tactile-zarr-taxel-headcam/`
  (`diff -rq` vs v1 / v1-headcam: only tactile differs). Dataset configs `dataset_wuji_taxel{,_headcam}.json`,
  `REPO_ID=wuji_tianji_v1_taxel{,_headcam}`; A's `train_val_split.json` copied in first, so state/action norm stats
  equal A's to 1e-15. CPU pre-flight OK (`check_taxel_cpu.log`): `WujiHand2Taxel_state_{120,102}` trained from
  scratch; 14 params without grad = the usual 7 VLM ones + the same 7 in the tactile expert's last layer (expected,
  the action expert only reads tactile KV).
- Runs E `wuji_tianji_ft_v1_taxel` (3 cams, vs A/B) and F `wuji_tianji_ft_v1_taxel_headcam` (head cam, vs C/D),
  same recipe as A-D (batch 128, 10k steps, ckpt every 2000, seed 42). Queued 2026-10-13 14:17 with
  `setsid nohup bash scripts_exp_zarr/wuji/queue_taxel_runs.sh`, which runs E then F, each after
  `wait_gpus_free.sh` sees all 8 GPUs < 1000 MiB for 5 checks 60 s apart (GPUs 0-3 were held by another
  container's job, PID not visible here, so PID-based `queue_after.sh` can't be used). F starts only if E's 9999 ckpt
  exists. Status in `ftp1_cache/queue_taxel.log`, training logs in `ftp1_cache/train_<exp>.log`.
  E ran 2026-10-13 15:01-19:51 (first try at 14:50 died at startup, see Known issues: relative DATASET_CONFIG),
  F 19:52-22:23; both exit 0. Kept 6000/8000/9999 like A-D (2000/4000 deleted 2026-10-15); all six runs = 382 GB.
  The volume is shared and ~100% full (135-215 GB free, fluctuating with other users; `df` lags deletes by ~1 min).
- Val rmse_total at 9999: A 0.0290 / B 0.0295 / E 0.0292; C 0.0329 / D 0.0331 / F 0.0328. Taxels change nothing
  offline (every metric within 0.001; F was ~0.001 below C/D at 4-6k steps, then converged).
- Eval/deploy scripts (`offline_eval.py`, `tactile_sensitivity.py`, `deploy/wuji_policy_io.py`) handle both formats
  (`wio.checkpoint_tactile_format` -> "agg" / "taxel").
- Tactile sensitivity A/E/C/F, step 9999, same 80 val frames (40 contact), `--per-episode 16 --device cuda`
  (~1 s/frame/variant on one A800, ~8 min total; `ftp1_cache/tactile_sensitivity_AECF{,.log}`). Contact frames,
  chunk change in deg when tactile is zeroed (arm / hand) vs resampling the flow noise:
  A 0.028 / 0.023 vs 0.18 / 0.15; E 0.054 / 0.069 vs 0.18 / 0.15; C 0.048 / 0.057 vs 0.25 / 0.21;
  **F 0.210 / 0.421 vs 0.25 / 0.21**. Only F (taxels + head cam only) clearly uses tactile: hand change > noise on
  48% of contact frames and in 4 of 5 val episodes (strongest ep8, task2 bottle: 0.97 vs 0.21), and zeroing raises
  its hand error 0.96 -> 1.18 deg (worse on 85% of contact frames); A/C/E stay <= +0.03 deg. Taxels ~2.5x the
  sensitivity of agg in both camera setups, and removing the wrist cams makes the policy lean on them. No-contact
  frames: ~0 for all (zeroing barely changes the input there). Offline rmse doesn't show this (F ~= C) -> F is the
  candidate to test on the robot against C/D, ideally with contact-critical moments (grasp/release).

### Analysis
- Layout: thumb 40 taxels, index..pinky 34 each with identical `x,y,z` (fingertip frame, curved pad ~15-19 mm, not a
  regular 2-D grid); layout constant across frames and bags (15 bags checked).
- Taxel `fx,fy,fz` are fixed-scale dimensionless (full scale 1.0), NOT per-frame normalized: sum|taxel| ~= 0.5-0.7 x
  |agg| (N), corr 0.98. So taxels carry magnitude plus distribution; agg_f linearly explains only R^2 0.3-0.8 of the
  map on contact frames (contact centroid moves ~1.5-2 mm std). Contact (|agg| > 0.3 N): thumb 46%, index 38%,
  middle 29%, ring 10%, pinky 6% of frames.
- Encoding options in FTP-1 (one token per fingertip in every case): `state` with the taxels flattened (chosen),
  as two groups because the shape must be uniform within a group: thumb `(T,1,120)` area 0 + fingers `(T,4,102)`
  areas 1-4 -> new tokenizers `WujiHand2Taxel_state_120/_102` (random init, no size limit). The z-score per dim is
  fine (std 2e-4..4e-2, no all-zero dims, <0.03% of elements clipped at +/-10). `matrix` needs a hand-made 2-D grid,
  and `MatrixCNNEncoder` crashes on anisotropic shapes like (2,64), because it pools both dims whenever either is >4.
  `image` (224x224x3 ViT) is overkill.
- Apply the spike mask to the taxels as well (spikes are blocks of byte-identical taxels), then interpolate like agg.
- Format verification 2026-10-17 (both formats, all 4 zarr variants): OXT-QC `precheck_metadata.py <variant>/WujiTianji
  -o /tmp/... --no-metadata` passes with 0 errors / 0 warnings (it only checks the four-piece set, channel count, type,
  timestamps). An independent read-only check (re-parse MCAP, rebuild the grid with np.interp, no converter code)
  matched both zarrs to <1e-7 on val bag `rosbag_20261008_165210` (no spikes); on `rosbag_20261008_171024` (35 index
  spikes) only 7 index frames differ, the same frames in both formats, all with lower |agg| (spike removed). Area ids
  match `definition_tactile_torque_function_area.png` (0 thumb tip, 1-4 index..pinky tips); finger order confirmed by
  corr(sum taxel fz, agg fz) being diagonal-dominant; headcam variants hold byte-identical tactile arrays.
- For an OXT submission: one physical sensor currently has two names (`WujiHand2`, `WujiHand2Taxel`); search
  open-x-tactile.github.io for an existing Wuji name first. Also needed: `sensor_<name>.png`, `hand_joints_idx.png`,
  `tactile_func_area_idx.png`, metadata.json. Tactile is point-sampled 100 -> 20 Hz (no window averaging), so contact
  transients shorter than 50 ms can be missed; deployment does the same (latest filtered sample).

## Known issues
- `DATASET_CONFIG` must be the same string (use the absolute path) for norm stats and training: the trainer compares
  `norm_params_snapshot.json` literally, so a relative path at norm time vs absolute at train time fails at startup
  with "Norm params mismatch ... dataset_config_path". Fix: rerun norm stats, or set the snapshot's path to the
  absolute form (the stats don't depend on the string).
- `uv` is not on PATH and `.venv` editable install points to a stale `/DATA/YuanZhen/...` path: call `.venv/bin/python`
  with `PYTHONPATH=src:packages/openpi-client/src` (done in `wuji_env.sh`). `.venv/bin/hf` shebang is also stale.
- `scripts/zarr_compute_norm_stats.py` did not call `config.finalize_config()`, so `--dataset_config_path` was ignored
  and it fell back to LeRobot (HF 404). Fixed by finalizing like the training script.
- Expected warnings when loading the pretrained ckpt: missing `embed_tokens.weight` (tied to `lm_head`, verified loaded),
  unexpected `shared_image_chunk_encoder.*` (no image tactile here), missing tokenizer `WujiHand2_state_3` (new sensor,
  trained from scratch).
- The ROS JointState name order is re-indexed per message by name in the converter; do not assume positional order.
- No head-camera hand-eye extrinsics exist for dc-dexumi-003 (`info.yaml` `calibration_export: null`, wrist cam
  calibration `exists: false`), so FK cannot be projected onto the images. `Base_R` is NOT a world-aligned frame
  (pick -> place, mostly horizontal toward the midline in the video, is ~-0.2 m along Base_R z). Harmless for FTP-1
  because state and action pose reps are relative (base frame cancels); it matters only for camera-frame poses/overlays.
- OXT-QC `precheck_4dvis.py` is of limited use here: without `camera_ego_pose` it puts the camera at the origin with
  identity rotation and there are no hand keypoints. Use `data/_tools/render_fk_check.py <zarr> <ep> <out.png>`
  (head + wrist images next to the FK skeleton and stored wrist axes) instead; checked OK on task3 ep10.
- visualize_tianji (docs/架构说明.md) treats the robot DH (`config/ccs_m6.MvKDCfg`) as authoritative (URDF arm chain
  matches it to 0.01 mm) but says the URDF flange->palm jig (51.5 mm) is measured too long and scales it to 36.5 mm
  (`JIG_LEN_M`). v1 wrist poses use the URDF value, i.e. the wrist origin sits 15 mm further toward the fingertips;
  effect on 32-step relative actions: median 2.7 mm / p95 8.4 mm vs median chunk translation 71 mm. Keep deployment
  FK identical to training (reuse `tianji_fk.py`); consider switching to 36.5 mm in a v2 conversion.
