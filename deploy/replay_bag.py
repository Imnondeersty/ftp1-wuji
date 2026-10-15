"""Offline end-to-end check of the deployment stack on a recorded bag, against a running policy server.

Feeds a raw MCAP (cameras as recorded JPEG, joint_states, fingertip PointCloud2) through the same PolicyRunner the ROS
node uses, on a simulated clock driven by the bag stamps. Each request is sent to the live server; its real latency
decides when the reply is delivered in simulated time, so async scheduling, chunk switching, safety clamps and the
tactile spike filter all run as they would on the robot. The robot itself is the recording (open loop), so this
checks plumbing and timing, not task success. Use a validation episode (see the skill doc for the mapping).

    python deploy/replay_bag.py --server ws://127.0.0.1:8848 --bag <rosbag_dir> --task task3 --out <dir>
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "client"))
sys.path.insert(0, "/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs")  # mcap readers (not in the project venv)

from executor import ChunkExecutor  # noqa: E402
from executor import SafetyFilter  # noqa: E402
import protocol as P  # noqa: E402
from robot_obs import reorder_joints  # noqa: E402
from runner import PolicyRunner  # noqa: E402
from ws_client import PolicyClient  # noqa: E402


def load_bag(bag_dir: pathlib.Path, cameras, use_tactile):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    topics = {P.ARM_STATE_TOPIC: ("arm", None), P.HAND_STATE_TOPIC: ("hand", None)}
    topics |= {P.CAMERA_TOPICS[c]: ("image", c) for c in cameras}
    if use_tactile:
        topics |= {t: ("tactile", f) for f, t in P.TACTILE_TOPICS.items()}
    events = []
    with open(next(bag_dir.glob("*_0.mcap")), "rb") as f:
        for _, channel, _, msg in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=list(topics)):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            events.append((stamp, *topics[channel.topic], msg))
    events.sort(key=lambda e: e[0])
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="ws://127.0.0.1:8848")
    parser.add_argument("--bag", type=pathlib.Path, required=True)
    parser.add_argument("--task", required=True, choices=["task2", "task3"])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=["sync", "async"], default="async")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--lead", type=int, default=6)
    parser.add_argument("--control-hz", type=float, default=120.0)
    parser.add_argument("--sim-latency", type=float, default=None,
                        help="deliver replies this many seconds after the request in simulated time instead of the "
                             "measured round trip (e.g. 0.3 when the server runs on CPU for a plumbing test)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    client = PolicyClient(args.server, timeout=600)
    meta = client.ping()
    runner = PolicyRunner(meta, args.task, ChunkExecutor(args.mode, args.horizon, args.lead),
                          SafetyFilter(), check_deviation=False)
    events = load_bag(args.bag, meta["cameras"], meta["use_tactile"])
    arm_rec = [(e[0], reorder_joints(e[3], P.ARM_JOINT_NAMES)) for e in events if e[1] == "arm"]
    hand_rec = [(e[0], reorder_joints(e[3], P.HAND_JOINT_NAMES)) for e in events if e[1] == "hand"]
    ta, qa = np.array([s for s, _ in arm_rec]), np.stack([q for _, q in arm_rec])
    th, qh = np.array([s for s, _ in hand_rec]), np.stack([q for _, q in hand_rec])

    def recorded(t):
        return np.concatenate([[np.interp(t, ta, qa[:, j]) for j in range(7)],
                               [np.interp(t, th, qh[:, j]) for j in range(20)]])

    tick = 1.0 / args.control_hz
    t, i, pending_reply = events[0][0], 0, None
    chunks, log = [], {"t": [], "cmd": [], "target": [], "recorded": []}
    wall0 = time.time()
    while t < events[-1][0]:
        while i < len(events) and events[i][0] <= t:
            stamp, kind, key, msg = events[i]
            if kind == "arm":
                runner.on_arm(msg, t)
            elif kind == "hand":
                runner.on_hand(msg, t)
            elif kind == "image":
                runner.on_image(key, msg.data, t)
            else:
                runner.on_tactile(key, msg, stamp, t)
            i += 1
        if pending_reply is not None and t >= pending_reply[0]:
            runner.on_reply(pending_reply[1], t)
            pending_reply = None
        if pending_reply is None:
            request = runner.make_request(t)
            if request is not None:
                w0 = time.perf_counter()
                reply = client.infer(request)
                latency = time.perf_counter() - w0
                pending_reply = (t + (latency if args.sim_latency is None else args.sim_latency), reply)
                chunks.append({"obs_time": t, "latency": latency, "infer_ms": reply["infer_ms"],
                               "arm_deg": reply["arm_deg"], "hand_rad": reply["hand_rad"]})
        out = runner.control(t)
        if out is not None:
            log["t"].append(t)
            log["cmd"].append(np.concatenate([out["arm_deg"], out["hand_rad"]]))
            log["target"].append(out["target"])
            log["recorded"].append(recorded(t))
        t += tick
    client.close()

    # chunk accuracy vs the recording: target k is for obs_time + k * dt
    arm_err, hand_err = [], []
    for c in chunks:
        ks = [k for k in range(1, 32) if c["obs_time"] + k * 0.05 <= events[-1][0]]
        if not ks:
            continue
        rec = np.stack([recorded(c["obs_time"] + k * 0.05) for k in ks])
        arm_err.append(np.abs(c["arm_deg"][ks] - rec[:, :7]).mean())
        hand_err.append(np.degrees(np.abs(c["hand_rad"][ks] - rec[:, 7:]).mean()))
    cmd, rec = np.asarray(log["cmd"]), np.asarray(log["recorded"])
    summary = runner.summary() | {
        "bag": str(args.bag), "mode": args.mode, "horizon": args.horizon, "lead": args.lead,
        "sim_latency": args.sim_latency,
        "chunks": len(chunks), "sim_seconds": float(events[-1][0] - events[0][0]),
        "wall_seconds": time.time() - wall0,
        "server_infer_ms_median": float(np.median([c["infer_ms"] for c in chunks])),
        "roundtrip_ms_median": float(np.median([c["latency"] for c in chunks]) * 1000),
        "chunk_err_vs_recording": {"arm_deg": float(np.mean(arm_err)), "hand_deg": float(np.mean(hand_err))},
        "command_err_vs_recording": {"arm_deg": float(np.abs(cmd[:, :7] - rec[:, :7]).mean()),
                                     "hand_deg": float(np.degrees(np.abs(cmd[:, 7:] - rec[:, 7:]).mean()))},
        "max_command_step": {"arm_deg": float(np.abs(np.diff(cmd[:, :7], axis=0)).max()),
                             "hand_rad": float(np.abs(np.diff(cmd[:, 7:], axis=0)).max())},
    }
    np.savez_compressed(args.out / "replay.npz", **{k: np.asarray(v) for k, v in log.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    plot(args.out / "replay.png", log, summary)


def plot(path, log, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(log["t"]) - log["t"][0]
    cmd, rec, tgt = (np.asarray(log[k]) for k in ("cmd", "recorded", "target"))
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    cols = [(f"arm joint {j + 1} (deg)", j, 1.0) for j in range(7)]
    cols += [(f"hand {n} (deg)", 7 + k, 180 / np.pi)
             for n, k in (("thumb cmc_flex", 0), ("thumb mcp", 2), ("index mcp_flex", 4), ("index pip", 6),
                          ("middle mcp_flex", 8))]
    for ax, (title, j, s) in zip(axes.ravel(), cols):
        ax.plot(t, rec[:, j] * s, "k", lw=1.5, label="recorded")
        ax.plot(t, tgt[:, j] * s, color="tab:orange", lw=0.8, label="executor target")
        ax.plot(t, cmd[:, j] * s, color="tab:blue", lw=1, label="published command")
        ax.set_title(title, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"replay {pathlib.Path(summary['bag']).name} ({summary['mode']}, horizon {summary['horizon']}): "
                 f"chunk err arm {summary['chunk_err_vs_recording']['arm_deg']:.2f} deg / "
                 f"hand {summary['chunk_err_vs_recording']['hand_deg']:.2f} deg, "
                 f"roundtrip {summary['roundtrip_ms_median']:.0f} ms")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


if __name__ == "__main__":
    main()
