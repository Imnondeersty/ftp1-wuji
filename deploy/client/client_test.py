"""Unit tests for the robot-side client (no ROS, no GPU): python -m pytest deploy/client/client_test.py"""

import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/wuji-vepfs/wuji-il/YuanZhen/data/_tools")

from executor import Chunk  # noqa: E402
from executor import ChunkExecutor  # noqa: E402
from executor import HandVelocity  # noqa: E402
from executor import SafetyFilter  # noqa: E402
import protocol as P  # noqa: E402
from robot_obs import ObsBuffer  # noqa: E402
from robot_obs import SpikeFilter  # noqa: E402
from robot_obs import reorder_joints  # noqa: E402


def _chunk_values(start, step=1.0):
    k = np.arange(32, dtype=float)[:, None]
    return start + step * k * np.ones((1, 7)), (start + step * k * np.ones((1, 20))) / 100


def test_chunk_interpolates_in_time_and_holds_the_end():
    arm, hand = _chunk_values(0.0)
    c = Chunk(arm, hand, dt=0.05, t0=10.0)
    assert c.at(10.0)[0] == pytest.approx(0.0)
    assert c.at(10.075)[0] == pytest.approx(1.5)
    assert c.at(9.0)[0] == pytest.approx(0.0)
    assert c.at(20.0)[0] == pytest.approx(31.0)


def test_sync_holds_at_horizon_and_references_arrival_time():
    ex = ChunkExecutor("sync", horizon=10)
    assert ex.needs_request(0.0)
    arm, hand = _chunk_values(0.0)
    ex.add_chunk(arm, hand, 0.05, obs_time=0.0, now=0.3)  # 300 ms inference: robot was holding still
    assert ex.target(0.3)[0] == pytest.approx(0.0)
    assert not ex.needs_request(0.6)
    assert ex.needs_request(0.8)
    assert ex.target(5.0)[0] == pytest.approx(10.0)  # waits at the horizon, does not run on


def test_async_requests_early_skips_elapsed_steps_and_blends():
    ex = ChunkExecutor("async", horizon=10, lead=7, blend=0.1)
    a1, h1 = _chunk_values(0.0)
    ex.add_chunk(a1, h1, 0.05, obs_time=0.0, now=0.3)
    assert ex.target(0.3)[0] == pytest.approx(6.0)  # obs time reference: 6 steps already elapsed
    assert not ex.needs_request(0.1) and ex.needs_request(0.16)
    a2, h2 = _chunk_values(100.0)
    ex.add_chunk(a2, h2, 0.05, obs_time=0.15, now=0.45)
    assert ex.target(0.45)[0] == pytest.approx(9.0)  # still the old chunk at the switch instant
    assert ex.target(0.55)[0] == pytest.approx(108.0)  # fully the new chunk after the blend


def test_safety_limits_position_and_speed():
    s = SafetyFilter(vel_scale=1.0)
    start = np.concatenate([np.full(7, -80.0), np.full(20, 0.1)])
    start[2], start[3] = 80.0, -100.0
    s.start(start, now=0.0)
    far = start.copy()
    far[0] += 50.0
    out = s.apply(far, now=0.01)
    assert out[0] == pytest.approx(start[0] + 81.8 * 0.01)
    beyond = start.copy()
    beyond[3] = 50.0  # inside the URDF limit (60) but far outside the data envelope (max -70 + 10)
    for k in range(1, 400):
        out = s.apply(beyond, now=0.01 * k)
    assert out[3] == pytest.approx(-60.0)
    assert s.clamped["position"] > 0 and s.clamped["velocity"] > 0


def test_hand_velocity_matches_teleop_gates():
    hv = HandVelocity()
    q = np.zeros(20)
    assert not hv(q, 0.0).any()
    v = hv(q + 0.01, 1 / 120)
    assert v[0] == pytest.approx(1.2)
    assert not hv(q + 0.5, 2 / 120).any()  # 0.49 rad jump -> zero
    assert not hv(q + 0.5, 1.0).any()  # dt > 50 ms -> zero


def test_reorder_joints_by_name():
    msg = types.SimpleNamespace(name=list(reversed(P.ARM_JOINT_NAMES)), position=list(range(7)))
    assert reorder_joints(msg, P.ARM_JOINT_NAMES).tolist() == list(reversed(range(7)))
    unnamed = types.SimpleNamespace(name=[], position=list(range(7)))
    assert reorder_joints(unnamed, P.ARM_JOINT_NAMES).tolist() == list(range(7))


def test_spike_filter_agrees_with_converter_rule():
    from convert_wuji_to_zarr import spike_mask

    rng = np.random.default_rng(0)
    n, taxels_n = 400, 34
    stamps = np.arange(n) * 0.01
    taxels = np.zeros((n, taxels_n, 3), np.float32)
    agg = np.zeros((n, 3), np.float32)
    agg[100:300, 2] = 0.3 + 0.05 * rng.standard_normal(200)  # light real contact, varied taxels
    taxels[100:300, :8] = rng.uniform(0.01, 0.05, (200, 8, 3))
    for s in (150, 220, 360):  # false spikes: big, uniform taxel pattern
        agg[s] = [0, 0, 3.0]
        taxels[s, :10] = [0.0, 0.0, 0.2]
    drop, n_spikes = spike_mask(stamps, taxels, agg)
    f = SpikeFilter()
    held = np.stack([f.update(stamps[i], taxels[i], agg[i]) for i in range(n)])
    assert n_spikes == f.spikes == 3
    assert np.all(held[drop, 2] < 1.0)  # spikes never reach the policy
    assert np.allclose(held[~drop], agg[~drop])  # everything else passes through unchanged


def test_spike_filter_holds_taxels_of_the_same_samples():
    from convert_wuji_to_zarr import spike_mask

    rng = np.random.default_rng(1)
    n = 300
    stamps = np.arange(n) * 0.01
    taxels = rng.uniform(0.0, 0.05, (n, 34, 3)).astype(np.float32)
    agg = np.zeros((n, 3), np.float32)
    agg[:, 2] = 0.3
    for s in (120, 200):
        agg[s] = [0, 0, 3.0]
        taxels[s, :10] = [0.0, 0.0, 0.2]
        taxels[s, 10:] = 0.0
    drop, _ = spike_mask(stamps, taxels, agg)
    f = SpikeFilter()
    held = []
    for i in range(n):
        f.update(stamps[i], taxels[i], agg[i])
        held.append(f.taxels.copy())
    held = np.stack(held)
    last_ok = np.maximum.accumulate(np.where(~drop, np.arange(n), 0))
    assert drop.sum() == 12
    assert np.array_equal(held, taxels[last_ok])  # accepted samples pass, dropped ones hold the last accepted map


def test_snapshot_sends_taxels_only_for_taxel_checkpoints():
    pc2 = types.SimpleNamespace(fields=[types.SimpleNamespace(name=n, datatype=7, offset=4 * i) for i, n in
                                        enumerate(("x", "y", "z", "fx", "fy", "fz", "agg_fx", "agg_fy", "agg_fz"))],
                                point_step=36, width=0, height=1, data=b"")
    for taxels in (False, True):
        obs = ObsBuffer(["head"], taxels=taxels)
        obs.arm_deg, obs.hand_rad = np.zeros(7), np.zeros(20)
        for k, finger in enumerate(P.FINGERS):
            n_taxels = 40 if finger == "thumb" else 34
            pts = np.zeros((n_taxels, 9), np.float32)
            pts[:, 3:6] = k + 1
            pts[0, 6:9] = [0, 0, k]
            pc2.width, pc2.data = n_taxels, pts.tobytes()
            obs.on_tactile(finger, pc2, stamp=0.0, now=0.0)
        snap = obs.snapshot("task3", now=0.0)
        assert snap["tactile_agg"][:, 2].tolist() == [0, 1, 2, 3, 4]
        assert ("tactile_taxels" in snap) == taxels
        if taxels:
            assert {f: v.shape for f, v in snap["tactile_taxels"].items()} == {
                "thumb": (40, 3), "index": (34, 3), "middle": (34, 3), "ring": (34, 3), "pinky": (34, 3)}
            assert all(np.all(snap["tactile_taxels"][f] == k + 1) for k, f in enumerate(P.FINGERS))
