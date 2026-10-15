"""How much does a checkpoint's predicted action chunk depend on tactile input?

For a fixed set of validation frames (stratified: contact / no contact, every val episode) each checkpoint predicts
the chunk with the real tactile input, with tactile zeroed (same flow noise), and with the real tactile input but a
different noise sample. Zeroing tactile is only meaningful if it moves the output clearly more than resampling noise.
Small enough to run on CPU while the GPUs are busy. Works for both tactile formats (fingertip forces / taxels); contact
frames are picked from the fingertip force in either case, so every checkpoint sees the same frames.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PYTHON" scripts_exp_zarr/wuji/tactile_sensitivity.py \
        --ckpt <ckpt> [--ckpt <ckpt> ...] --out <dir> --per-episode 4 --device cpu
"""
import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "deploy"))
from offline_eval import build_val_dataset  # noqa: E402
from offline_eval import chunk_errors  # noqa: E402
from offline_eval import fingertip_force  # noqa: E402
from offline_eval import joint_diff_deg  # noqa: E402
from offline_eval import predict  # noqa: E402
from offline_eval import raw_tactile  # noqa: E402
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper  # noqa: E402
import wuji_policy_io as wio  # noqa: E402


def frame_index(vds):
    """(zarr_idx, absolute frame) -> dataset position, plus per-episode frame lists."""
    pos, episodes = {}, collections.defaultdict(list)
    for p in range(len(vds)):
        g = vds.indices[p]
        z, f = int(vds.zarr_idxs[g]), int(vds.data_idxs[g])
        pos[(z, f)] = p
        episodes[(z, int(vds.episodes_idxs[g]), int(vds.start_frames[g]), int(vds.end_frames[g]))].append(f)
    return pos, episodes


def select_frames(vds, episodes, per_episode):
    """Half contact frames (max fingertip force > 1 N), half no-contact, evenly spread in time, >= 8 future frames."""
    chosen = []
    for (z, ep, start, end), frames in sorted(episodes.items()):
        force = fingertip_force(vds, z, start, end).max(-1)
        frames = [f for f in sorted(frames) if end - f >= 8]
        contact = [f for f in frames if force[f - start] > 1.0]
        free = [f for f in frames if force[f - start] <= 1.0]
        n_c = min(len(contact), (per_episode + 1) // 2)
        for group, n, flag in ((contact, n_c, True), (free, per_episode - n_c, False)):
            if n > 0 and group:
                for i in np.linspace(0, len(group) - 1, n + 2)[1:-1].round().astype(int):
                    chosen.append((z, ep, start, end, group[i], flag))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--per-episode", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    datasets = {ck: build_val_dataset(ck) for ck in args.ckpt}
    ref = datasets[args.ckpt[0]]
    _, episodes = frame_index(ref)
    chosen = select_frames(ref, episodes, args.per_episode)
    print(f"{len(chosen)} frames ({sum(c[-1] for c in chosen)} contact) from {len(episodes)} val episodes", flush=True)

    results = {}
    for ck in args.ckpt:
        vds = datasets[ck]
        pos, _ = frame_index(vds)
        w = FTP1InferenceWrapper(str(ck), domain_name=wio.DOMAIN, device=args.device)
        fmt = wio.checkpoint_tactile_format(w)
        assert fmt is not None, f"{ck} has no tactile input"
        rows, t0 = [], time.time()
        for z, ep, start, end, f, contact in chosen:
            s = vds[pos[(z, f)]]
            data = vds.datas[z]
            arm = np.asarray(data["right_arm_joints"][f], dtype=np.float64)
            hand = np.asarray(data["right_hand_joints"][f], dtype=np.float64)
            images = {k: np.asarray(v, dtype=np.float32) for k, v in s["image"].items()}
            state = wio.build_state(arm, hand)
            tac = wio.tactile_kwargs(fmt, raw_tactile(fmt, data, f))
            zero = wio.zero_tactile(tac)
            gen = torch.Generator().manual_seed(f * 1000 + ep)
            noise, noise2 = (torch.randn((1, 32, wio.STATE_DIM), generator=gen) for _ in range(2))
            with torch.no_grad():
                p = predict(w, images, state, s["prompt"], tac, noise)
                p0 = predict(w, images, state, s["prompt"], zero, noise)
                p2 = predict(w, images, state, s["prompt"], tac, noise2)
            n = min(32, end - f)
            gt = np.asarray(s["actions"], dtype=np.float64)
            row = {"episode": ep, "frame": f - start, "contact": contact, "images": sorted(images)}
            row |= {f"zero_tactile/{k}": v for k, v in joint_diff_deg(p, p0, n).items()}
            row |= {f"other_noise/{k}": v for k, v in joint_diff_deg(p, p2, n).items()}
            row |= {f"err/{k}": v for k, v in chunk_errors(p, gt, n).items()}
            row |= {f"err_zero_tactile/{k}": v for k, v in chunk_errors(p0, gt, n).items()}
            rows.append(row)
            print(f"[{ck.parent.name}] {len(rows)}/{len(chosen)} ep{ep} t={f - start} contact={contact} "
                  f"zero-tactile arm {row['zero_tactile/arm_deg']:.3f} hand {row['zero_tactile/hand_deg']:.3f} | "
                  f"other-noise arm {row['other_noise/arm_deg']:.3f} hand {row['other_noise/hand_deg']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        summary = {}
        for name, sel in (("all", rows), ("contact", [r for r in rows if r["contact"]]),
                          ("no_contact", [r for r in rows if not r["contact"]])):
            if sel:
                keys = [k for k in sel[0] if "/" in k]
                summary[name] = {k: float(np.mean([r[k] for r in sel])) for k in keys} | {"n": len(sel)}
        results[str(ck)] = {"tactile_format": fmt, "summary": summary, "rows": rows}
        del w
    (args.out / "tactile_sensitivity.json").write_text(json.dumps(results, indent=1))
    for ck, r in results.items():
        print(f"\n== {ck} ({r['tactile_format']})")
        for name, s in r["summary"].items():
            print(f"{name:10s} n={s['n']:2d}  zero-tactile: arm {s['zero_tactile/arm_deg']:.3f} hand "
                  f"{s['zero_tactile/hand_deg']:.3f} deg | other-noise: arm {s['other_noise/arm_deg']:.3f} hand "
                  f"{s['other_noise/hand_deg']:.3f} deg | err arm {s['err/arm_deg_mean']:.2f}->"
                  f"{s['err_zero_tactile/arm_deg_mean']:.2f} hand {s['err/hand_deg_mean']:.2f}->"
                  f"{s['err_zero_tactile/hand_deg_mean']:.2f} deg (real->zero tactile)")


if __name__ == "__main__":
    main()
