"""ROS-free control core of the robot client: observation buffer + chunk executor + safety filter.

The ROS node (ros_node.py) and the offline bag replay (deploy/replay_bag.py) both drive this class, so everything
except the ROS plumbing is exercised offline. All times are the caller's monotonic seconds.
"""

import collections
import threading

import numpy as np

from executor import ChunkExecutor
from executor import HandVelocity
from executor import SafetyFilter
from robot_obs import ObsBuffer


class PolicyRunner:
    def __init__(self, metadata: dict, task: str, executor: ChunkExecutor, safety: SafetyFilter,
                 stale_sec: float = 0.3, max_errors: int = 3, check_deviation: bool = True):
        if task not in metadata["tasks"]:
            raise ValueError(f"task {task!r} not served by this checkpoint: {sorted(metadata['tasks'])}")
        self.meta = metadata
        self.task = task
        self.obs = ObsBuffer(metadata["cameras"], taxels=metadata.get("tactile_format") == "taxel")
        self.use_tactile = bool(metadata["use_tactile"])
        self.executor = executor
        self.safety = safety
        self.hand_velocity = HandVelocity()
        self.stale_sec = stale_sec
        self.max_errors = max_errors
        self.check_deviation = check_deviation
        self.lock = threading.Lock()
        self.stopped_reason = None
        self.errors = 0
        self.prev_arm = None
        self.stats = collections.Counter()
        self.latencies = []

    # ------------------------------------------------------------------ inputs (ROS callbacks / bag replay)
    def on_arm(self, msg, now):
        with self.lock:
            self.obs.on_arm(msg, now)

    def on_hand(self, msg, now):
        with self.lock:
            self.obs.on_hand(msg, now)

    def on_image(self, camera, jpeg, now):
        with self.lock:
            self.obs.on_image(camera, jpeg, now)

    def on_tactile(self, finger, msg, stamp, now):
        with self.lock:
            self.obs.on_tactile(finger, msg, stamp, now)

    # ------------------------------------------------------------------ inference side
    def stale_streams(self, now):
        return {k: round(a, 3) for k, a in self.obs.ages(now, self.use_tactile).items() if a > self.stale_sec}

    def make_request(self, now):
        """Observation to send now, or None. Marks the executor as waiting for a reply."""
        with self.lock:
            if self.stopped_reason or not self.executor.needs_request(now):
                return None
            stale = self.stale_streams(now)
            if stale:
                self.stats["skipped_stale_obs"] += 1
                return None
            self.executor.pending = True
            return self.obs.snapshot(self.task, now)

    def on_reply(self, reply, now):
        with self.lock:
            self.executor.add_chunk(reply["arm_deg"], reply["hand_rad"], reply["dt"], reply["obs_time"], now)
            self.errors = 0
            self.latencies.append(now - reply["obs_time"])
            self.stats["chunks"] += 1

    def on_error(self, exc, now):
        with self.lock:
            self.executor.pending = False
            self.errors += 1
            self.stats["inference_errors"] += 1
            if self.errors >= self.max_errors:
                self.stop(f"{self.errors} consecutive inference errors, last: {exc}")

    # ------------------------------------------------------------------ control side
    def stop(self, reason):
        if self.stopped_reason is None:
            self.stopped_reason = reason

    def control(self, now):
        """Commands for this control tick: dict(arm_deg, arm_vel, hand_rad, hand_vel) or None (publish nothing)."""
        with self.lock:
            if self.stopped_reason:
                return None
            target = self.executor.target(now)
            if target is None:
                return None
            measured = np.concatenate([self.obs.arm_deg, self.obs.hand_rad])
            if self.safety.last is None:
                self.safety.start(measured, now)
            if self.executor.mode == "async" and self.executor.exhausted(now):
                self.stats["exhausted_ticks"] += 1
            cmd = self.safety.apply(target, now)
            bad = self.safety.deviation_exceeded(measured)
            if bad is not None:
                self.stats["deviation_ticks"] += 1
                if self.check_deviation:
                    self.stop(f"command/state deviation too large at joints {bad.tolist()}: "
                              f"cmd {np.round(cmd[bad], 3).tolist()} vs measured {np.round(measured[bad], 3).tolist()}")
                    return None
            arm, hand = cmd[:7], cmd[7:]
            arm_vel = np.zeros(7) if self.prev_arm is None or now <= self.prev_arm[1] else \
                (arm - self.prev_arm[0]) / (now - self.prev_arm[1])
            self.prev_arm = (arm.copy(), now)
            self.stats["ticks"] += 1
            return {"arm_deg": arm, "arm_vel": arm_vel, "hand_rad": hand, "hand_vel": self.hand_velocity(hand, now),
                    "target": target}

    def summary(self):
        lat = np.asarray(self.latencies) * 1000 if self.latencies else np.zeros(1)
        return {"stats": dict(self.stats), "safety_clamps": dict(self.safety.clamped), "stopped": self.stopped_reason,
                "latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                               "max": float(lat.max())},
                "tactile_spikes": {f: flt.spikes for f, flt in self.obs.filters.items()}}
