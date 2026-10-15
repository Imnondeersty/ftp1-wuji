"""Wire contract between the FTP-1 policy server (GPU) and the Wuji/Tianji robot client.

Modelled on OpenWAM's deploy split (openwam/deploy/server.py + benchmarks/utils/transport.py): the robot client
is thin and sends raw sensor data; every model-specific step (image crop/resize, state layout, normalization,
relative -> absolute actions) happens on the server, driven by the checkpoint. Messages are msgpack with numpy
support (openpi_client.msgpack_numpy) over one persistent WebSocket.

Client -> server
    {"type": "obs",
     "task": "task2" | "task3",           # or "prompt": "<exact training instruction>"
     "images": {"head": <jpeg bytes>, "right_wrist": <jpeg bytes> | None, "left_wrist": <jpeg bytes> | None},
     "arm_deg": float[7],                 # /right_arm/joint_states, DEGREES, order right_joint_1..7
     "hand_rad": float[20],               # /right_hand/joint_states, RADIANS, order right_finger{1..5}_joint{1..4}
     "tactile_agg": float[5, 3],          # fingertip aggregate force agg_f{x,y,z} in N, thumb..pinky
     "tactile_taxels": {finger: float[P, 3]},  # only if metadata tactile_format == "taxel": per-taxel fx,fy,fz
                                          # (dimensionless) in PointCloud2 point order, thumb P=40, others P=34
     "obs_time": float}                   # client clock, echoed back untouched
    {"type": "reset"} | {"type": "ping"}

Server -> client
    {"type": "action",
     "arm_deg": float[H, 7], "hand_rad": float[H, 20],   # ABSOLUTE joint targets, H = 32
     "dt": 0.05,                                         # target k is meant for obs time + k * dt (k = 0 is "now")
     "obs_time": <echo>, "infer_ms": float, "step": int}
    {"type": "pong", **metadata}        # see PolicyMetadata keys below
    {"type": "reset_ack"}
    {"type": "error", "code": str, "message": str}

This module must stay importable without torch or ROS.
"""

OBS = "obs"
RESET = "reset"
PING = "ping"
ACTION = "action"
PONG = "pong"
RESET_ACK = "reset_ack"
ERROR = "error"

ERR_UNKNOWN_TYPE = "unknown_message_type"
ERR_OBS_VALIDATION = "obs_validation_error"
ERR_INTERNAL = "internal_error"

CAMERAS = ("head", "right_wrist", "left_wrist")
CAMERA_TOPICS = {
    "head": "/cam_head/color/image_raw/compressed",
    "right_wrist": "/cam_right_wrist/color/image_raw/compressed",
    "left_wrist": "/cam_left_wrist/color/image_raw/compressed",
}

ARM_JOINT_NAMES = [f"right_joint_{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [f"right_finger{k}_joint{j}" for k in range(1, 6) for j in range(1, 5)]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

ARM_STATE_TOPIC = "/right_arm/joint_states"
HAND_STATE_TOPIC = "/right_hand/joint_states"
ARM_COMMAND_TOPIC = "/right_arm/joint_commands"
HAND_COMMAND_TOPIC = "/right_hand/joint_commands"
TACTILE_TOPICS = {finger: f"/right_hand/tactile/{finger}" for finger in FINGERS}

# Keys of the metadata dict returned in PONG. tactile_format: "agg" (tactile_agg), "taxel" (tactile_taxels) or None.
METADATA_KEYS = ("checkpoint", "cameras", "use_tactile", "tactile_format", "action_horizon", "dt", "tasks",
                 "arm_joint_names", "hand_joint_names")
