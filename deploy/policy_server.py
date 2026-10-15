"""WebSocket policy server for Wuji/Tianji FTP-1 checkpoints (runs on the GPU machine).

The robot client sends raw JPEGs, joint angles and fingertip forces or taxels (see protocol.py); this server
reproduces the training-time preprocessing exactly (wuji_policy_io, verified bit-exact against the training data by
scripts_exp_zarr/wuji/offline_eval.py), runs FTP1InferenceWrapper and returns an absolute 32-step joint target chunk.
The tactile format ("agg" forces or "taxel") follows the checkpoint and is reported in the PONG metadata.

    bash deploy/serve.sh <checkpoint_dir> [--port 8848] [--device cuda:0]
"""

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

_DEPLOY_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOY_DIR.parent
for _p in (_DEPLOY_DIR, _REPO_ROOT / "src", _REPO_ROOT / "packages" / "openpi-client" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openpi_client import msgpack_numpy  # noqa: E402
import protocol as P  # noqa: E402
import wuji_policy_io as wio  # noqa: E402

logger = logging.getLogger("wuji_policy_server")

# training image key (zarr) -> protocol camera name -> model image key
_ZARR_TO_CAMERA = {"camera_ego_rgb": "head", "right_wrist_camera_rgb": "right_wrist",
                   "left_wrist_camera_rgb": "left_wrist"}
_CAMERA_TO_MODEL_KEY = {"head": "camera_ego_rgb_0", "right_wrist": "right_wrist_camera_rgb_0",
                        "left_wrist": "left_wrist_camera_rgb_0"}


class ObsValidationError(ValueError):
    pass


def cameras_from_checkpoint(ckpt_dir: pathlib.Path) -> list[str]:
    """Camera set the checkpoint was trained with, read from its dataset config -> zarr image arrays."""
    train_cfg = json.loads((ckpt_dir / "train_config.json").read_text())
    dataset_cfg = json.loads(pathlib.Path(train_cfg["dataset_config_path"]).read_text())
    found = set()
    for entry in dataset_cfg["datasets"]:
        if not entry.get("enabled", True):
            continue
        for zarr_dir in sorted(pathlib.Path(entry["path"]).glob("*.zarr")):
            found |= {_ZARR_TO_CAMERA[p.name] for p in (zarr_dir / "data").iterdir() if p.name in _ZARR_TO_CAMERA}
    if not found:
        raise RuntimeError(f"could not determine the cameras of {ckpt_dir}; pass --cameras")
    return [c for c in P.CAMERAS if c in found]


class WujiPolicy:
    """Raw robot observation -> absolute joint target chunk."""

    def __init__(self, ckpt_dir: pathlib.Path, device: str, cameras: list[str] | None, num_steps: int):
        from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

        self.ckpt_dir = ckpt_dir
        self.cameras = cameras or cameras_from_checkpoint(ckpt_dir)
        self.wrapper = FTP1InferenceWrapper(str(ckpt_dir), domain_name=wio.DOMAIN, device=device,
                                            num_inference_steps=num_steps)
        self.use_tactile = bool(self.wrapper.model_config.use_tactile_input)
        self.tactile_format = wio.checkpoint_tactile_format(self.wrapper)
        self.horizon = int(self.wrapper.get_action_horizon())
        self._lock = threading.Lock()
        self.steps = 0

    def metadata(self) -> dict:
        return {"checkpoint": str(self.ckpt_dir), "cameras": list(self.cameras), "use_tactile": self.use_tactile,
                "tactile_format": self.tactile_format, "action_horizon": self.horizon, "dt": 1.0 / wio.FPS,
                "tasks": dict(wio.INSTRUCTIONS),
                "arm_joint_names": P.ARM_JOINT_NAMES, "hand_joint_names": P.HAND_JOINT_NAMES}

    def _parse(self, obs: dict):
        if "prompt" in obs:
            prompt = str(obs["prompt"])
            if prompt not in wio.INSTRUCTIONS.values():
                raise ObsValidationError(f"prompt {prompt!r} is not a training instruction: {wio.INSTRUCTIONS}")
        elif obs.get("task") in wio.INSTRUCTIONS:
            prompt = wio.INSTRUCTIONS[obs["task"]]
        else:
            raise ObsValidationError(f"need 'task' in {sorted(wio.INSTRUCTIONS)} or an exact 'prompt'")

        arm = np.asarray(obs.get("arm_deg"), dtype=np.float64).reshape(-1)
        hand = np.asarray(obs.get("hand_rad"), dtype=np.float64).reshape(-1)
        if arm.shape != (7,) or hand.shape != (20,) or not (np.isfinite(arm).all() and np.isfinite(hand).all()):
            raise ObsValidationError(f"arm_deg must be 7 and hand_rad 20 finite values, got {arm.shape} {hand.shape}")

        images = {}
        for cam in self.cameras:
            data = (obs.get("images") or {}).get(cam)
            if data is None:
                raise ObsValidationError(f"checkpoint needs camera {cam!r}, missing in obs")
            bgr = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ObsValidationError(f"camera {cam!r}: not a decodable JPEG/PNG")
            images[_CAMERA_TO_MODEL_KEY[cam]] = wio.preprocess_bgr(bgr)

        tac_kwargs = {}
        if self.tactile_format == "agg":
            tac = np.asarray(obs.get("tactile_agg"), dtype=np.float32)
            if tac.shape != (5, 3) or not np.isfinite(tac).all():
                raise ObsValidationError(f"tactile_agg must be finite (5, 3), got {tac.shape}")
            tac_kwargs = dict(zip(("tactiles", "tactile_function_areas", "tactile_sensors"), wio.build_tactile(tac)))
        elif self.tactile_format == "taxel":
            sent = obs.get("tactile_taxels") or {}
            taxels = {}
            for finger, n in wio.TAXEL_COUNTS.items():
                tac = np.asarray(sent[finger], dtype=np.float32) if sent.get(finger) is not None else None
                if tac is None or tac.shape != (n, 3) or not np.isfinite(tac).all():
                    raise ObsValidationError(f"tactile_taxels[{finger!r}] must be finite ({n}, 3), got "
                                             f"{None if tac is None else tac.shape}")
                taxels[finger] = tac
            tac_kwargs = wio.tactile_kwargs("taxel", taxels)
        return prompt, np.deg2rad(arm), hand, images, tac_kwargs

    def infer(self, obs: dict) -> dict:
        prompt, arm_rad, hand_rad, images, tac_kwargs = self._parse(obs)
        state = wio.build_state(arm_rad, hand_rad)
        with self._lock:
            t0 = time.perf_counter()
            actions = self.wrapper.infer(images=images, state=state, prompt=prompt, **tac_kwargs)
            infer_ms = (time.perf_counter() - t0) * 1000
            self.steps += 1
        arm_t, hand_t = wio.actions_to_targets(np.asarray(actions, dtype=np.float64), arm_rad, hand_rad)
        return {"type": P.ACTION, "arm_deg": np.degrees(arm_t), "hand_rad": hand_t, "dt": 1.0 / wio.FPS,
                "obs_time": obs.get("obs_time"), "infer_ms": infer_ms, "step": self.steps}


def serve(policy: WujiPolicy, host: str, port: int) -> None:
    import websockets.asyncio.server as ws_server
    import websockets.exceptions

    async def handler(ws):
        logger.info("client connected: %s", ws.remote_address)
        packer = msgpack_numpy.Packer()
        try:
            async for raw in ws:
                try:
                    msg = msgpack_numpy.unpackb(raw)
                    kind = msg.get("type", P.OBS)
                    if kind == P.OBS:
                        # inference blocks for ~0.3 s; keep the event loop free for pings
                        reply = await asyncio.to_thread(policy.infer, msg)
                    elif kind == P.PING:
                        reply = {"type": P.PONG, **policy.metadata()}
                    elif kind == P.RESET:
                        reply = {"type": P.RESET_ACK}
                    else:
                        reply = {"type": P.ERROR, "code": P.ERR_UNKNOWN_TYPE, "message": f"unknown type {kind!r}"}
                except ObsValidationError as e:
                    logger.info("obs rejected: %s", e)
                    reply = {"type": P.ERROR, "code": P.ERR_OBS_VALIDATION, "message": str(e)}
                except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                    logger.exception("inference failed")
                    reply = {"type": P.ERROR, "code": P.ERR_INTERNAL, "message": f"{type(e).__name__}: {e}"}
                await ws.send(packer.pack(reply))
        except websockets.exceptions.ConnectionClosed:
            pass
        logger.info("client disconnected: %s", ws.remote_address)

    async def main():
        async with ws_server.serve(handler, host, port, compression=None, max_size=None, ping_interval=None):
            logger.info("serving %s on ws://%s:%d (cameras=%s, tactile=%s)", policy.ckpt_dir, host, port,
                        policy.cameras, policy.tactile_format)
            await asyncio.Future()

    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=pathlib.Path, help="FTP-1 step directory, e.g. .../wuji_tianji_ft_v1/9999")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cameras", nargs="+", choices=P.CAMERAS, help="override the camera set read from the ckpt")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3, help="dummy inferences before serving")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    policy = WujiPolicy(args.checkpoint.resolve(), args.device, args.cameras, args.denoise_steps)
    if args.warmup:
        jpeg = cv2.imencode(".jpg", np.zeros((720, 1280, 3), np.uint8))[1].tobytes()
        dummy = {"task": "task3", "images": {c: jpeg for c in policy.cameras}, "arm_deg": np.zeros(7),
                 "hand_rad": np.zeros(20), "tactile_agg": np.zeros((5, 3)),
                 "tactile_taxels": {f: np.zeros((n, 3)) for f, n in wio.TAXEL_COUNTS.items()}}
        for _ in range(args.warmup):
            out = policy.infer(dummy)
        logger.info("warm-up done, last inference %.0f ms", out["infer_ms"])
        policy.steps = 0
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
