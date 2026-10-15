"""Robot-side input/output conversion for FTP-1 policies trained on the Wuji/Tianji v1 data.

Mirrors data/_tools/convert_wuji_to_zarr.py (conversion) and src/openpi/dataset_zarr.py (sample assembly) so that
deployment feeds the model exactly what it saw in training. Verified against the training data by
scripts_exp_zarr/wuji/offline_eval.py. Right arm/hand only (v1 data has no left arm/hand).

Units on the robot side: arm joints in DEGREES (as in /right_arm/joint_states), hand joints in RADIANS,
fingertip aggregate force agg_f{x,y,z} in N in the order thumb, index, middle, ring, pinky.

Two tactile formats, fixed per checkpoint (see checkpoint_tactile_format):
    "agg"    v1 data: per-fingertip aggregate force (5, 3) in N                        -> build_tactile
    "taxel"  data/_tools/convert_wuji_taxel.py: per-taxel fx, fy, fz (dimensionless,   -> build_tactile_taxel
             full scale 1.0) of every fingertip, {finger: (P, 3)} in PointCloud2 point order
"""

import cv2
import numpy as np

from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_JOINT_DIM
from openpi.pose_utils import mat_to_pose9d

DOMAIN = "WujiTianji"
INSTRUCTIONS = {
    "task2": "place the bottle into the tape roll.",
    "task3": "place the small tape roll on the large tape roll.",
}
IMG_SIZE = 224
# image key in the model -> robot topic (head camera = camera_ego_rgb)
CAMERA_TOPICS = {
    "camera_ego_rgb_0": "/cam_head/color/image_raw/compressed",
    "right_wrist_camera_rgb_0": "/cam_right_wrist/color/image_raw/compressed",
    "left_wrist_camera_rgb_0": "/cam_left_wrist/color/image_raw/compressed",
}
# Wuji hand 2, per finger joint1..4 (thumb: cmc_flex, cmc_abd, mcp, ip / others: mcp_flex, mcp_abd, pip, dip)
HAND_JOINTS_IDX = np.array([1, 26, 2, 3, 7, 6, 8, 9, 12, 11, 13, 14, 17, 16, 18, 19, 22, 21, 23, 24])
TACTILE_KEY = "right_tactile_fingertips"
TACTILE_AREAS = [0, 1, 2, 3, 4]
TACTILE_SENSOR = "WujiHand2"
# Per-taxel format: thumb pad 40 taxels, the other fingers share one 34-taxel layout -> two tactile inputs, one token
# per fingertip, each fingertip flattened taxel-major (t0 fx, fy, fz, t1 fx, ...) like the converter.
TAXEL_SENSOR = "WujiHand2Taxel"
TAXEL_COUNTS = {"thumb": 40, "index": 34, "middle": 34, "ring": 34, "pinky": 34}
TAXEL_GROUPS = {  # tactile key -> (fingers, function areas)
    "right_tactile_taxelthumb": (("thumb",), [0]),
    "right_tactile_taxelfingers": (("index", "middle", "ring", "pinky"), [1, 2, 3, 4]),
}
FPS = 20  # one action step = 50 ms; action[0] is the current state, action[k] the target at t + k / FPS

STATE_DIM = 120
ARM_SLICE = slice(9, 9 + 7)  # right arm joints in the 120-d state/action
HAND_OFFSET = 9 + FTP1_SINGLE_ARM_JOINT_DIM  # right hand 32-slot block starts here
_IDENTITY_POSE9D = mat_to_pose9d(np.eye(4))


def preprocess_image(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG (any size) -> centre square crop -> 224x224 RGB uint8, identical to the converter."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return preprocess_bgr(img)


def preprocess_bgr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_state(arm_rad: np.ndarray, hand_rad: np.ndarray) -> np.ndarray:
    """Current right arm (7, rad) + hand (20, rad) -> raw (1, 120) state.

    With proprioception_pose_rep='relative' and no history the wrist pose is expressed relative to itself, i.e. it is
    always the identity, so no FK is needed here. Arm and hand joints are absolute (proprioception_joint_rep='abs').
    """
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    state[0, :9] = _IDENTITY_POSE9D
    state[0, ARM_SLICE] = arm_rad
    state[0, HAND_OFFSET + HAND_JOINTS_IDX] = hand_rad
    return state


def build_tactile(agg_force: np.ndarray):
    """(5, 3) fingertip aggregate forces (N) -> (tactiles, function_areas, sensors) kwargs for wrapper.infer()."""
    tac = np.asarray(agg_force, dtype=np.float32).reshape(1, 5, 3)
    return {TACTILE_KEY: tac}, {TACTILE_KEY: list(TACTILE_AREAS)}, {TACTILE_KEY: TACTILE_SENSOR}


def build_tactile_taxel(taxels: dict):
    """{finger: (P, 3) taxel fx, fy, fz} -> (tactiles, function_areas, sensors) kwargs for wrapper.infer()."""
    tactiles = {}
    for key, (fingers, _) in TAXEL_GROUPS.items():
        flat = [np.asarray(taxels[f], dtype=np.float32).reshape(TAXEL_COUNTS[f] * 3) for f in fingers]
        tactiles[key] = np.stack(flat)[None]  # (1, n_fingers, 3P)
    return tactiles, {k: list(a) for k, (_, a) in TAXEL_GROUPS.items()}, {k: TAXEL_SENSOR for k in TAXEL_GROUPS}


def tactile_format(tactile_keys) -> str | None:
    """Tactile input keys of a checkpoint -> "agg", "taxel" or None (no tactile input)."""
    keys = set(tactile_keys)
    if not keys:
        return None
    if keys == {TACTILE_KEY}:
        return "agg"
    if keys == set(TAXEL_GROUPS):
        return "taxel"
    raise ValueError(f"unknown Wuji tactile inputs {sorted(keys)}")


def checkpoint_tactile_format(wrapper) -> str | None:
    """Tactile format an FTP1InferenceWrapper's checkpoint was trained with (None if it has no tactile input)."""
    if not wrapper.model_config.use_tactile_input:
        return None
    return tactile_format(wrapper.tactile_key_type_map)


def tactile_kwargs(fmt: str | None, raw) -> dict:
    """Raw tactile in format fmt ((5, 3) forces or {finger: (P, 3)} taxels) -> keyword arguments for wrapper.infer()."""
    if fmt is None:
        return {}
    build = {"agg": build_tactile, "taxel": build_tactile_taxel}[fmt]
    return dict(zip(("tactiles", "tactile_function_areas", "tactile_sensors"), build(raw), strict=True))


def zero_tactile(kwargs: dict) -> dict:
    """Same tactile kwargs with every reading set to zero (no contact)."""
    return dict(kwargs, tactiles={k: np.zeros_like(v) for k, v in kwargs["tactiles"].items()})


def actions_to_targets(actions: np.ndarray, arm_rad_now: np.ndarray, hand_rad_now: np.ndarray):
    """Raw (32, 120) model actions -> absolute right arm (32, 7, rad) and hand (32, 20, rad) joint targets.

    action_joint_rep='mix': arm and hand joints are deltas w.r.t. the joints at inference time (the gripper slot 28,
    the only absolute hand slot, is unused by the Wuji hand).
    """
    arm = arm_rad_now[None] + actions[:, ARM_SLICE]
    hand = hand_rad_now[None] + actions[:, HAND_OFFSET + HAND_JOINTS_IDX]
    return arm, hand
