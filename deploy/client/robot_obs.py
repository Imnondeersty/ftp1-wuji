"""Robot-side observation assembly (numpy only; shared by the ROS runner and the offline bag replay).

Matches data/_tools/convert_wuji_to_zarr.py: joint_states re-indexed by name, fingertip aggregate force from the
tactile PointCloud2 with the dataset-README spike rule applied per finger at the raw 100 Hz rate, cameras forwarded
as the compressed JPEG bytes (the server decodes / crops / resizes exactly like the converter).
"""

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import protocol as P  # noqa: E402

_PC2_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def parse_tactile(msg):
    """sensor_msgs/PointCloud2 (fields x,y,z,fx,fy,fz,agg_fx,agg_fy,agg_fz,...) -> (taxels (P,3), agg (3,))."""
    dt = np.dtype({"names": [f.name for f in msg.fields], "formats": [_PC2_DTYPES[f.datatype] for f in msg.fields],
                   "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
    pts = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    taxels = np.stack([pts["fx"], pts["fy"], pts["fz"]], axis=-1).astype(np.float32)
    agg = np.array([pts["agg_fx"][0], pts["agg_fy"][0], pts["agg_fz"][0]], dtype=np.float32)
    return taxels, agg


def reorder_joints(msg, expected_names):
    """JointState -> positions ordered like expected_names (by name when names are present)."""
    pos = np.asarray(msg.position, dtype=np.float64)
    names = list(msg.name)
    if names and len(names) == len(pos):
        index = {n: i for i, n in enumerate(names)}
        missing = [n for n in expected_names if n not in index]
        if missing:
            raise ValueError(f"joint_states lacks {missing}")
        return pos[[index[n] for n in expected_names]]
    if len(pos) != len(expected_names):
        raise ValueError(f"joint_states has {len(pos)} positions, expected {len(expected_names)}")
    return pos


class SpikeFilter:
    """Online version of the dataset README rule used by the converter (one instance per finger).

    A sample is a false spike if |agg| > 1 N, >= 6 taxels are lit, >= 80% of the lit taxels share one byte-identical
    vector, and |agg| >= median(|agg| over the previous 1 s) + 0.5 N. The spike and the next 5 samples are dropped;
    while dropping, the last accepted value is held. The taxels of the same samples are filtered alike (`taxels`).
    """

    def __init__(self):
        self._history = collections.deque()  # (stamp, |agg|)
        self._drop = 0
        self.value = np.zeros(3, np.float32)
        self.taxels = None  # (P, 3), last accepted per-taxel reading
        self.spikes = 0

    def update(self, stamp: float, taxels: np.ndarray, agg: np.ndarray) -> np.ndarray:
        if self.taxels is None:
            self.taxels = np.zeros(taxels.shape, np.float32)
        mag = float(np.linalg.norm(agg))
        while self._history and self._history[0][0] < stamp - 1.0:
            self._history.popleft()
        prev = [m for t, m in self._history if t < stamp]
        self._history.append((stamp, mag))
        if mag > 1.0 and prev:
            lit = np.any(taxels != 0, axis=1)
            n_lit = int(lit.sum())
            if n_lit >= 6:
                vecs = np.ascontiguousarray(taxels[lit]).view(np.dtype((np.void, 12))).ravel()
                if np.unique(vecs, return_counts=True)[1].max() >= 0.8 * n_lit and mag >= np.median(prev) + 0.5:
                    self._drop = 6
                    self.spikes += 1
        if self._drop > 0:
            self._drop -= 1
        else:
            self.value = agg.astype(np.float32)
            self.taxels = taxels.astype(np.float32)
        return self.value


class ObsBuffer:
    """Latest value of every input stream, plus when it arrived (caller's monotonic clock)."""

    def __init__(self, cameras, taxels: bool = False):
        self.cameras = list(cameras)
        self.send_taxels = taxels  # add per-taxel readings to the snapshot (checkpoints with tactile_format "taxel")
        self.arm_deg = None
        self.hand_rad = None
        self.images = {}
        self.filters = {f: SpikeFilter() for f in P.FINGERS}
        self.seen = {}

    def on_arm(self, msg, now: float):
        self.arm_deg = reorder_joints(msg, P.ARM_JOINT_NAMES)
        self.seen["arm"] = now

    def on_hand(self, msg, now: float):
        self.hand_rad = reorder_joints(msg, P.HAND_JOINT_NAMES)
        self.seen["hand"] = now

    def on_image(self, camera: str, jpeg: bytes, now: float):
        self.images[camera] = bytes(jpeg)
        self.seen[camera] = now

    def on_tactile(self, finger: str, msg, stamp: float, now: float):
        taxels, agg = parse_tactile(msg)
        self.filters[finger].update(stamp, taxels, agg)
        self.seen[f"tactile_{finger}"] = now

    def required(self, use_tactile: bool):
        keys = ["arm", "hand", *self.cameras]
        return keys + ([f"tactile_{f}" for f in P.FINGERS] if use_tactile else [])

    def ages(self, now: float, use_tactile: bool) -> dict:
        return {k: (now - self.seen[k]) if k in self.seen else float("inf") for k in self.required(use_tactile)}

    def snapshot(self, task: str, now: float) -> dict:
        obs = {"task": task, "images": {c: self.images.get(c) for c in self.cameras},
               "arm_deg": self.arm_deg.copy(), "hand_rad": self.hand_rad.copy(),
               "tactile_agg": np.stack([self.filters[f].value for f in P.FINGERS]), "obs_time": now}
        if self.send_taxels:
            obs["tactile_taxels"] = {f: self.filters[f].taxels.copy() for f in P.FINGERS}
        return obs
