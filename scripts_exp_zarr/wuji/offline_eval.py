"""Offline open-loop evaluation of Wuji FTP-1 checkpoints on the held-out validation episodes.

For every sampled validation frame the raw observation is fed through FTP1InferenceWrapper (the deployment entry
point) and the predicted 32-step action chunk is compared with the recorded one. The script also checks that
wuji_policy_io (robot-side state/tactile/target conversion) reproduces the training samples exactly.

Each checkpoint gets the tactile format it was trained with (fingertip forces or taxels, see wuji_policy_io) and, if
it was trained on another data variant (cameras / tactile format), its own copy of the same validation frames. So B
can be a no-tactile baseline or a checkpoint with the other tactile format (e.g. taxel vs force).

    source scripts_exp_zarr/wuji/wuji_env.sh
    "$PYTHON" scripts_exp_zarr/wuji/offline_eval.py --ckpt-a <tactile ckpt> --ckpt-b <other ckpt> --out <dir>
"""
import argparse
import collections
import dataclasses
import json
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "deploy"))
from openpi.dataset_zarr import MultiZarrDataset  # noqa: E402
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper  # noqa: E402
from openpi.pose_utils import pose10d_to_mat  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import wuji_policy_io as wio  # noqa: E402

HAND_COLS = wio.HAND_OFFSET + wio.HAND_JOINTS_IDX
PLOT_HAND = {"thumb cmc_flex": 0, "thumb mcp": 2, "index mcp_flex": 4, "index pip": 6}
TACTILE_LABEL = {"agg": "tactile", "taxel": "taxel tactile", None: "no tactile"}


def build_val_dataset(ckpt_dir: pathlib.Path):
    """Rebuild the exact training data config from the checkpoint's train_config.json and return the val split."""
    train_cfg = json.loads((ckpt_dir / "train_config.json").read_text())
    base = _config.get_config("ftp1")
    names = {f.name for f in dataclasses.fields(base) if f.init}
    kwargs = {k: v for k, v in train_cfg.items()
              if k in names and k != "name" and isinstance(v, (str, int, float, bool, list))}
    config = dataclasses.replace(base, **kwargs)
    config.finalize_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    val = MultiZarrDataset(data_config, config.model.action_horizon, split="train").get_val_dataset()
    val.set_skip_normalization(True)
    return val.domain_dataset_list[0]


def dataset_id(ckpt_dir: pathlib.Path):
    train_cfg = json.loads((ckpt_dir / "train_config.json").read_text())
    return train_cfg["repo_id"], train_cfg["dataset_config_path"]


def raw_tactile(fmt, data, frame):
    """Robot-side tactile reading at one zarr frame in format fmt (inverse of the converters' layout)."""
    if fmt == "agg":
        return np.asarray(data["right_tactile_data_fingertips"][frame])
    taxels = {}
    for key, (fingers, _) in wio.TAXEL_GROUPS.items():
        flat = np.asarray(data[key.replace("right_tactile_", "right_tactile_data_", 1)][frame])  # (n_fingers, 3P)
        taxels |= {finger: flat[i].reshape(wio.TAXEL_COUNTS[finger], 3) for i, finger in enumerate(fingers)}
    return taxels


def fingertip_force(vds, zarr_idx, start, end):
    """(T, 5) fingertip |aggregate force| (N) of an episode; taxel zarrs read it from the v1 zarr they came from, so
    contact frames are the same for every variant."""
    data = vds.datas[zarr_idx]
    if "right_tactile_data_fingertips" not in data:
        root = pathlib.Path(vds.dataset_path)
        src = pathlib.Path(json.loads((root.parent / "conversion_report.json").read_text())["src"])
        data = zarr.open_group(str(src / root.name / vds.zarr_file_list[zarr_idx] / "data"), mode="r")
    return np.linalg.norm(np.asarray(data["right_tactile_data_fingertips"][start:end]), axis=-1)


def tactile_io_err(kwargs, sample):
    """Max |robot-side tactile - training sample| over all tactile inputs (areas, sensors and keys must match)."""
    assert set(kwargs["tactiles"]) == set(sample["tactile"]), (sorted(kwargs["tactiles"]), sorted(sample["tactile"]))
    err = 0.0
    for key, tac in kwargs["tactiles"].items():
        assert list(sample["tactile_function_area"][key]) == kwargs["tactile_function_areas"][key]
        assert sample["tactile_sensor"][key] == kwargs["tactile_sensors"][key]
        err = max(err, float(np.abs(tac - sample["tactile"][key]).max()))
    return err


def predict(wrapper, images, state, prompt, tac_kwargs, noise):
    obs = wrapper.build_observation(images=images, state=state, prompt=prompt, **tac_kwargs)
    actions = wrapper.model.sample_actions(device=wrapper.device, observation=obs, noise=noise.to(wrapper.device),
                                           num_steps=wrapper.num_inference_steps)
    return wrapper.denormalize_action(actions.float().cpu().numpy()[0])


def chunk_errors(pred, gt, n):
    """Mean error over future steps 1..n-1 and at the last valid step, per action group."""
    s = slice(1, n)
    rp = pose10d_to_mat(pred[s, :9].astype(np.float64))[:, :3, :3]
    rg = pose10d_to_mat(gt[s, :9].astype(np.float64))[:, :3, :3]
    cos = (np.einsum("tij,tij->t", rp, rg) - 1) / 2
    err = {
        "wrist_pos_mm": np.linalg.norm(pred[s, :3] - gt[s, :3], axis=-1) * 1000,
        "wrist_rot_deg": np.degrees(np.arccos(np.clip(cos, -1, 1))),
        "arm_deg": np.degrees(np.abs(pred[s, wio.ARM_SLICE] - gt[s, wio.ARM_SLICE])).mean(-1),
        "hand_deg": np.degrees(np.abs(pred[s][:, HAND_COLS] - gt[s][:, HAND_COLS])).mean(-1),
    }
    out = {f"{k}_mean": float(v.mean()) for k, v in err.items()}
    out |= {f"{k}_end": float(v[-1]) for k, v in err.items()}
    return out


def joint_diff_deg(a, b, n):
    s = slice(1, n)
    return {"arm_deg": float(np.degrees(np.abs(a[s, wio.ARM_SLICE] - b[s, wio.ARM_SLICE])).mean()),
            "hand_deg": float(np.degrees(np.abs(a[s][:, HAND_COLS] - b[s][:, HAND_COLS])).mean())}


def mean_rows(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-a", type=pathlib.Path, required=True, help="checkpoint trained with tactile")
    parser.add_argument("--ckpt-b", type=pathlib.Path, required=True,
                        help="checkpoint to compare with: without tactile, or with the other tactile format")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--stride", type=int, default=4, help="evaluate every N-th validation frame")
    parser.add_argument("--plot-stride", type=int, default=12, help="draw a predicted chunk every N frames")
    parser.add_argument("--timing-calls", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    vds = build_val_dataset(args.ckpt_a)
    episodes = collections.defaultdict(dict)
    for p in range(len(vds)):
        g = vds.indices[p]
        key = (int(vds.zarr_idxs[g]), int(vds.episodes_idxs[g]), int(vds.start_frames[g]), int(vds.end_frames[g]))
        episodes[key][int(vds.data_idxs[g])] = p
    print(f"validation: {len(vds)} frames in {len(episodes)} episodes")

    wa = FTP1InferenceWrapper(str(args.ckpt_a), domain_name=wio.DOMAIN, device="cuda:0")
    wb = FTP1InferenceWrapper(str(args.ckpt_b), domain_name=wio.DOMAIN, device="cuda:1")
    fmt_a, fmt_b = wio.checkpoint_tactile_format(wa), wio.checkpoint_tactile_format(wb)
    assert fmt_a is not None, f"{args.ckpt_a} has no tactile input"
    print(f"tactile format: A {fmt_a}, B {fmt_b}")
    vds_b, pos_b = vds, None  # B trained on another data variant: same frames, from B's own dataset
    if dataset_id(args.ckpt_b) != dataset_id(args.ckpt_a):
        vds_b = build_val_dataset(args.ckpt_b)
        assert vds_b.zarr_file_list == vds.zarr_file_list, (vds_b.zarr_file_list, vds.zarr_file_list)
        pos_b = {}
        for p in range(len(vds_b)):
            g = vds_b.indices[p]
            pos_b[(int(vds_b.zarr_idxs[g]), int(vds_b.data_idxs[g]))] = p

    rows, io_check = [], collections.defaultdict(float)
    timing_input = None
    for (zarr_idx, ep_idx, start, end), frame_to_pos in sorted(episodes.items()):
        data = vds.datas[zarr_idx]
        task = "task2" if "task2" in vds.zarr_file_list[zarr_idx] else "task3"
        arm = np.asarray(data["right_arm_joints"][start:end], dtype=np.float64)
        hand = np.asarray(data["right_hand_joints"][start:end], dtype=np.float64)
        force = fingertip_force(vds, zarr_idx, start, end)  # (T, 5)
        plot_chunks = []
        for f in sorted(frame_to_pos)[::args.stride]:
            t = f - start
            s = vds[frame_to_pos[f]]
            # near the episode end the chunk is padded by repeating the last action (mask included), so count
            # the real future frames instead of trusting the mask
            n = min(32, end - f)
            if n < 2:  # no future step to compare against
                continue
            images = {k: np.asarray(v, dtype=np.float32) for k, v in s["image"].items()}
            sb = s if pos_b is None else vds_b[pos_b[(zarr_idx, f)]]
            images_b = images if pos_b is None else {k: np.asarray(v, dtype=np.float32) for k, v in sb["image"].items()}

            # deployment-side conversion must reproduce the training sample
            state = wio.build_state(arm[t], hand[t])
            tac_kwargs = wio.tactile_kwargs(fmt_a, raw_tactile(fmt_a, data, f))
            tac_b = wio.tactile_kwargs(fmt_b, raw_tactile(fmt_b, vds_b.datas[zarr_idx], f)) if fmt_b else {}
            gt = np.asarray(s["actions"], dtype=np.float64)
            arm_t, hand_t = wio.actions_to_targets(gt, arm[t], hand[t])
            io_check["state_max_abs_err"] = max(io_check["state_max_abs_err"], float(np.abs(state - s["state"]).max()))
            io_check["tactile_max_abs_err"] = max(io_check["tactile_max_abs_err"], tactile_io_err(tac_kwargs, s))
            if fmt_b is not None:
                io_check["tactile_b_max_abs_err"] = max(io_check["tactile_b_max_abs_err"], tactile_io_err(tac_b, sb))
            io_check["target_arm_max_abs_err_deg"] = max(io_check["target_arm_max_abs_err_deg"], float(
                np.degrees(np.abs(arm_t[:n] - arm[t:t + n]).max())))
            io_check["target_hand_max_abs_err_deg"] = max(io_check["target_hand_max_abs_err_deg"], float(
                np.degrees(np.abs(hand_t[:n] - hand[t:t + n]).max())))

            gen = torch.Generator().manual_seed(f * 1000 + ep_idx)
            noise, noise2 = (torch.randn((1, 32, wio.STATE_DIM), generator=gen) for _ in range(2))
            with torch.no_grad():
                pa = predict(wa, images, state, s["prompt"], tac_kwargs, noise)
                pa2 = predict(wa, images, state, s["prompt"], tac_kwargs, noise2)
                pa0 = predict(wa, images, state, s["prompt"], wio.zero_tactile(tac_kwargs), noise)
                pb = predict(wb, images_b, state, s["prompt"], tac_b, noise)
                if fmt_b is not None:
                    pb2 = predict(wb, images_b, state, s["prompt"], tac_b, noise2)
                    pb0 = predict(wb, images_b, state, s["prompt"], wio.zero_tactile(tac_b), noise)
            hold = np.repeat(gt[:1], len(gt), axis=0)
            contact = bool(force[t].max() > 1.0)
            row = {"task": task, "episode": ep_idx, "frame": t, "contact": contact}
            for name, pred in (("A", pa), ("B", pb), ("hold", hold)):
                row |= {f"{name}/{k}": v for k, v in chunk_errors(pred, gt, n).items()}
            row |= {f"A_vs_A0/{k}": v for k, v in joint_diff_deg(pa, pa0, n).items()}
            row |= {f"A_vs_A2/{k}": v for k, v in joint_diff_deg(pa, pa2, n).items()}
            if fmt_b is not None:
                row |= {f"B_vs_B0/{k}": v for k, v in joint_diff_deg(pb, pb0, n).items()}
                row |= {f"B_vs_B2/{k}": v for k, v in joint_diff_deg(pb, pb2, n).items()}
            rows.append(row)
            if t % args.plot_stride < args.stride:
                plot_chunks.append((t, n, *wio.actions_to_targets(pa, arm[t], hand[t]),
                                    *wio.actions_to_targets(pb, arm[t], hand[t])))
            timing_input = timing_input or (images, images_b, state, s["prompt"], tac_kwargs, tac_b)
        plot_episode(args.out / f"val_{task}_ep{ep_idx}.png", task, ep_idx, arm, hand, force, plot_chunks,
                     (f"A ({TACTILE_LABEL[fmt_a]}) chunk", f"B ({TACTILE_LABEL[fmt_b]}) chunk"))
        print(f"episode {task}/{ep_idx}: {len(frame_to_pos)} frames, {sum(r['episode'] == ep_idx for r in rows)} evaluated")

    # latency of the real deployment call (wrapper.infer, bf16, 10 flow steps), after warm-up
    images, images_b, state, prompt, tac_kwargs, tac_b = timing_input
    latency = {}
    for name, w, ims, kw in (("A", wa, images, tac_kwargs), ("B", wb, images_b, tac_b)):
        for _ in range(3):
            w.infer(images=ims, state=state, prompt=prompt, **kw)
        ts = []
        for _ in range(args.timing_calls):
            torch.cuda.synchronize(w.device)
            t0 = time.perf_counter()
            w.infer(images=ims, state=state, prompt=prompt, **kw)
            torch.cuda.synchronize(w.device)
            ts.append((time.perf_counter() - t0) * 1000)
        latency[name] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}

    metric_rows = [{k: v for k, v in r.items() if "/" in k} for r in rows]
    summary = {"n_samples": len(rows), "tactile_format": {"A": fmt_a, "B": fmt_b}, "io_check": dict(io_check),
               "latency": latency, "all": mean_rows(metric_rows)}
    for task in ("task2", "task3"):
        sel = [m for m, r in zip(metric_rows, rows) if r["task"] == task]
        if sel:
            summary[task] = mean_rows(sel) | {"n": len(sel)}
    for flag in (True, False):
        sel = [m for m, r in zip(metric_rows, rows) if r["contact"] == flag]
        if sel:
            summary["contact" if flag else "no_contact"] = mean_rows(sel) | {"n": len(sel)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out / "rows.json").write_text(json.dumps(rows))
    print(json.dumps(summary, indent=1))


def plot_episode(path, task, ep_idx, arm, hand, force, chunks, labels=("A (tactile) chunk", "B (no tactile) chunk")):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(arm)) / wio.FPS
    series = [(f"arm joint {j + 1} (deg)", np.degrees(arm[:, j]), lambda c, j=j: (np.degrees(c[2][:, j]),
                                                                                    np.degrees(c[4][:, j])))
              for j in range(7)]
    series += [(f"hand {name} (deg)", np.degrees(hand[:, k]), lambda c, k=k: (np.degrees(c[3][:, k]),
                                                                              np.degrees(c[5][:, k])))
               for name, k in PLOT_HAND.items()]
    for ax, (title, gt, pick) in zip(axes, series):
        ax.plot(x, gt, color="k", lw=1.5, label="recorded")
        for c in chunks:
            t, n = c[0], c[1]
            xa = (t + np.arange(n)) / wio.FPS
            a, b = pick(c)
            ax.plot(xa, a[:n], color="tab:blue", lw=0.9, alpha=0.8)
            ax.plot(xa, b[:n], color="tab:red", lw=0.9, alpha=0.8)
        ax.set_title(title, fontsize=9)
    ax = axes[11]
    for k, name in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
        ax.plot(x, force[:, k], label=name, lw=1)
    ax.set_title("fingertip |F| (N)", fontsize=9)
    ax.legend(fontsize=7)
    axes[0].plot([], [], color="tab:blue", label=labels[0])
    axes[0].plot([], [], color="tab:red", label=labels[1])
    axes[0].legend(fontsize=7)
    for ax in axes[-3:]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{task} val episode {ep_idx}: recorded vs predicted 32-step chunks (open loop)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
