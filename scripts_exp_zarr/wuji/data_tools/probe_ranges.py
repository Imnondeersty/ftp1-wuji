"""Per-joint ranges of hand joint_states and taxel row layout for one bag."""

import sys

sys.path.insert(0, "/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs")
sys.path.insert(0, "/wuji-vepfs/wuji-il/YuanZhen/data/_tools")

import collections

from inspect_bag import pc2_to_struct
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np


def main(path):
    joints = collections.defaultdict(list)
    names = {}
    taxel_xyz = {}
    agg = collections.defaultdict(list)
    topics = [
        "/right_hand/joint_states",
        "/left_hand/joint_states",
        "/right_arm/joint_states",
        "/right_hand/tactile/thumb",
        "/right_hand/tactile/index",
    ]
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            t = channel.topic
            if t.endswith("joint_states"):
                names[t] = list(msg.name)
                joints[t].append(np.asarray(msg.position))
            else:
                pts = pc2_to_struct(msg)
                taxel_xyz.setdefault(t, np.stack([pts["x"], pts["y"], pts["z"]], -1))
                agg[t].append([pts["agg_fx"][0], pts["agg_fy"][0], pts["agg_fz"][0]])
    for t, arr in joints.items():
        a = np.stack(arr)
        print(f"\n{t}  n={len(a)}")
        for n, lo, hi in zip(names[t], a.min(0), a.max(0)):
            print(f"  {n:24s} [{lo:8.3f}, {hi:8.3f}]  range={hi - lo:.3f}")
    for t, xyz in taxel_xyz.items():
        a = np.asarray(agg[t])
        print(f"\n{t}: {len(xyz)} taxels; agg |F| max={np.linalg.norm(a, axis=1).max():.2f} N")
        for axis, name in enumerate("xyz"):
            vals, cnt = np.unique(np.round(xyz[:, axis], 4), return_counts=True)
            print(f"  unique {name}: {len(vals)} -> counts {cnt.tolist()}")


if __name__ == "__main__":
    main(sys.argv[1])
