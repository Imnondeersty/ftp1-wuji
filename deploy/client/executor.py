"""Turns 20 Hz action chunks into 100-120 Hz joint commands (numpy only).

Chunks are indexed by time, not by tick: target k of a chunk is meant for t0 + k * dt and is linearly interpolated
in between. Like OpenWAM's executors there are two modes:

- sync:  request -> wait (robot holds the last command) -> execute `horizon` steps -> request ... ;
         t0 = arrival time, because the robot did not move while the server was thinking.
- async: the next request is sent `lead` steps before the current chunk's horizon runs out, so inference overlaps
         execution; t0 = observation time (the server's targets are absolute, so the steps that elapsed during
         inference are simply skipped), and the switch to a new chunk is cross-faded over `blend` seconds.

SafetyFilter clamps every published command to joint limits (URDF hard limits intersected with the training data
envelope plus a margin) and to per-joint speed limits, and reports when the measured state is far from the command.
"""

import collections

import numpy as np

# Envelope of the training data (right arm deg, right hand rad): union of the measured joints (v1 action targets) and
# the teleop commands (*_cmd action targets), see the deploy section of the skill doc. The commands go past the
# measured joints on contact (that is the grip), e.g. thumb mcp up to 0.724 rad vs 0.53 measured.
ARM_DATA_MIN_DEG = np.array([-133.2, -87.9, 46.4, -145.0, -156.8, -59.3, -5.2])
ARM_DATA_MAX_DEG = np.array([-48.3, -42.9, 119.7, -70.0, -82.0, 37.7, 41.5])
HAND_DATA_MIN_RAD = np.array([-0.071, -0.689, -0.004, -0.049, -0.2, -0.321, -0.01, 0.0, -0.126, -0.298, -0.01,
                              0.041, -0.079, -0.354, -0.012, 0.001, -0.126, -0.364, -0.009, 0.022])
HAND_DATA_MAX_RAD = np.array([1.004, 0.456, 0.724, 0.277, 0.362, 0.453, 1.17, 0.897, 0.838, 0.342, 0.581, 0.995,
                              0.819, 0.195, 0.741, 0.843, 1.075, 0.364, 0.52, 0.495])
# Arm hard limits (tianji_full/arm_right.urdf; all recorded arm data lies inside them). The hand part of that URDF
# does NOT match the real Wuji hand 2 (e.g. thumb joint2 lower limit -0.139 rad, recorded down to -0.69 rad), so hand
# commands are bounded by the data envelope only; the hand driver additionally clamps to the device soft limits.
ARM_URDF_MIN_DEG = np.array([-178.0, -120.0, -178.0, -145.0, -178.0, -60.0, -90.0])
ARM_URDF_MAX_DEG = np.array([178.0, 120.0, 178.0, 60.0, 178.0, 60.0, 90.0])
# Max joint speed seen in the training data at 20 Hz, measured joints and commands (arm deg/s, hand rad/s)
ARM_DATA_VMAX_DEG = np.array([90.9, 44.6, 74.1, 65.7, 139.8, 112.9, 54.0])
HAND_DATA_VMAX_RAD = np.array([1.53, 2.23, 1.45, 1.72, 1.18, 1.29, 3.19, 2.59, 1.5, 0.89, 1.14, 3.16, 1.69, 1.02,
                               2.09, 2.16, 2.22, 1.69, 1.83, 1.19])


class Chunk:
    def __init__(self, arm_deg, hand_rad, dt: float, t0: float):
        self.targets = np.concatenate([np.asarray(arm_deg, np.float64), np.asarray(hand_rad, np.float64)], axis=1)
        self.dt = float(dt)
        self.t0 = float(t0)

    def step(self, t: float) -> float:
        return (t - self.t0) / self.dt

    def at(self, t: float) -> np.ndarray:
        k = float(np.clip(self.step(t), 0.0, len(self.targets) - 1))
        i = min(int(k), len(self.targets) - 2)
        w = k - i
        return (1 - w) * self.targets[i] + w * self.targets[i + 1]


class ChunkExecutor:
    def __init__(self, mode: str = "async", horizon: int = 14, lead: int = 6, blend: float = 0.1):
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be sync or async, got {mode!r}")
        if mode == "async" and not 0 <= lead < horizon:
            raise ValueError("async needs 0 <= lead < horizon")
        self.mode, self.horizon, self.lead, self.blend = mode, int(horizon), int(lead), float(blend)
        self.reset()

    def reset(self):
        self.current = None
        self.previous = None
        self.switch_time = None
        self.pending = False

    def add_chunk(self, arm_deg, hand_rad, dt: float, obs_time: float, now: float):
        t0 = now if self.mode == "sync" else obs_time
        self.previous, self.current = self.current, Chunk(arm_deg, hand_rad, dt, t0)
        self.switch_time = now
        self.pending = False

    def needs_request(self, now: float) -> bool:
        if self.pending:
            return False
        if self.current is None:
            return True
        trigger = self.horizon if self.mode == "sync" else self.horizon - self.lead
        return self.current.step(now) >= trigger

    def exhausted(self, now: float) -> bool:
        """Past the planned horizon by more than the chunk can cover: the caller should hold and investigate."""
        return self.current is not None and self.current.step(now) > len(self.current.targets) - 1

    def target(self, now: float):
        """Joint target (7 arm deg + 20 hand rad) for time `now`, or None before the first chunk."""
        if self.current is None:
            return None
        if self.mode == "sync":
            # never run past the horizon: hold its last target while the next chunk is computed
            t = min(now, self.current.t0 + self.horizon * self.current.dt)
            return self.current.at(t)
        out = self.current.at(now)
        if self.previous is not None and self.blend > 0 and now - self.switch_time < self.blend:
            w = (now - self.switch_time) / self.blend
            out = (1 - w) * self.previous.at(now) + w * out
        return out


class SafetyFilter:
    """Clamp commands to joint limits and speed limits; flag large command/state deviations."""

    def __init__(self, data_margin_arm_deg: float = 10.0, data_margin_hand_rad: float = 0.15, vel_scale: float = 1.0,
                 max_dev_arm_deg: float = 20.0, max_dev_hand_rad: float = 0.6):
        self.lo = np.concatenate([np.maximum(ARM_URDF_MIN_DEG, ARM_DATA_MIN_DEG - data_margin_arm_deg),
                                  HAND_DATA_MIN_RAD - data_margin_hand_rad])
        self.hi = np.concatenate([np.minimum(ARM_URDF_MAX_DEG, ARM_DATA_MAX_DEG + data_margin_arm_deg),
                                  HAND_DATA_MAX_RAD + data_margin_hand_rad])
        self.vmax = np.concatenate([ARM_DATA_VMAX_DEG, HAND_DATA_VMAX_RAD]) * float(vel_scale)
        self.max_dev = np.concatenate([np.full(7, max_dev_arm_deg), np.full(20, max_dev_hand_rad)])
        self.last = None
        self.last_time = None
        self.clamped = collections.Counter()

    def start(self, measured: np.ndarray, now: float):
        """Begin from the measured state so the first commands ramp instead of jumping."""
        self.last = np.clip(np.asarray(measured, np.float64), self.lo, self.hi)
        self.last_time = now

    def apply(self, target: np.ndarray, now: float) -> np.ndarray:
        dt = max(now - self.last_time, 0.0)
        cmd = np.clip(target, self.lo, self.hi)
        if np.any(cmd != target):
            self.clamped["position"] += 1
        step = np.clip(cmd - self.last, -self.vmax * dt, self.vmax * dt)
        if np.any(step != cmd - self.last):
            self.clamped["velocity"] += 1
        self.last = self.last + step
        self.last_time = now
        return self.last.copy()

    def deviation_exceeded(self, measured: np.ndarray):
        """Indices where |command - measured| is beyond the limit (None if all fine)."""
        bad = np.flatnonzero(np.abs(self.last - measured) > self.max_dev)
        return bad if bad.size else None


class HandVelocity:
    """Velocity field for /right_hand/joint_commands, computed like teleop's hand_interface._velocity_for.

    The Wuji hand driver feeds it into the MIT kd term: Δq/Δt of the published positions, zero when Δt is outside
    [2 ms, 50 ms] or any joint jumps more than 0.35 rad, averaged over the last 3 ticks.
    """

    def __init__(self):
        self._prev = None
        self._window = collections.deque(maxlen=3)

    def __call__(self, positions: np.ndarray, now: float) -> np.ndarray:
        prev, self._prev = self._prev, (positions.copy(), now)
        if prev is None:
            return np.zeros(20)
        dt = now - prev[1]
        delta = positions - prev[0]
        if not 0.002 <= dt <= 0.05 or np.max(np.abs(delta)) > 0.35:
            self._window.clear()
            return np.zeros(20)
        self._window.append(delta / dt)
        return np.mean(self._window, axis=0)
