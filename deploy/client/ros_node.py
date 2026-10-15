"""ROS2 robot client: Wuji/Tianji sensors -> FTP-1 policy server -> right arm/hand joint_commands.

    python deploy/client/ros_node.py --server ws://<gpu-host>:8848 --task task3              # dry run (default)
    python deploy/client/ros_node.py --server ws://<gpu-host>:8848 --task task3 --execute    # actually command

!!! wuji-dc/roles/inference/README.md: until the robot side has an owner/lease arbiter, teleop is the only allowed
!!! control source and inference must not publish. Use --execute only with that rule cleared by the robot owner,
!!! with teleop's command output stopped, and with an operator at the e-stop.

Command contract (wuji-dc roles/teleop/docs/topics-and-qos.md): sensor_msgs/JointState on /right_arm/joint_commands
(DEGREES, 7) and /right_hand/joint_commands (RADIANS, 20, with names), BEST_EFFORT depth 1, non-zero header.stamp
from CLOCK_REALTIME, published continuously (arm watchdog 0.25 s: on silence the arm holds its measured position;
the hand freezes its last target). The left arm/hand are never commanded. Ctrl-C or --max-duration stops publishing.
Needs rclpy + sensor_msgs (ROS2 Humble) and numpy/msgpack/websockets; no torch.
"""

import argparse
import json
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from executor import ChunkExecutor  # noqa: E402
from executor import SafetyFilter  # noqa: E402
import protocol as P  # noqa: E402
from runner import PolicyRunner  # noqa: E402
from ws_client import PolicyClient  # noqa: E402


def realtime_stamp():
    """header.stamp from CLOCK_REALTIME, like wuji_clock.clock.now_msg()."""
    from builtin_interfaces.msg import Time

    sec, nanosec = divmod(time.clock_gettime_ns(time.CLOCK_REALTIME), 1_000_000_000)
    return Time(sec=int(sec), nanosec=int(nanosec))


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="ws://host:port of deploy/policy_server.py")
    parser.add_argument("--task", required=True, choices=["task2", "task3"])
    parser.add_argument("--execute", action="store_true", help="publish joint_commands (default: dry run)")
    parser.add_argument("--mode", choices=["sync", "async"], default="async")
    parser.add_argument("--horizon", type=int, default=14, help="chunk step (50 ms each) at which the next chunk should take over")
    parser.add_argument("--lead", type=int, default=6, help="async: request this many steps before the horizon (~ latency / 50 ms)")
    parser.add_argument("--blend", type=float, default=0.1, help="async: cross-fade between chunks (s)")
    parser.add_argument("--control-hz", type=float, default=120.0, help="arm/hand command rate (contract: 120)")
    parser.add_argument("--vel-scale", type=float, default=1.0, help="speed cap = data max speed x this")
    parser.add_argument("--max-duration", type=float, default=40.0, help="stop publishing after this many seconds")
    parser.add_argument("--stale-sec", type=float, default=0.3)
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("deploy_logs"))
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import JointState
    from sensor_msgs.msg import PointCloud2

    client = PolicyClient(args.server)
    meta = client.ping()
    shown = ("checkpoint", "cameras", "use_tactile", "tactile_format")
    print(f"[runner] server: {json.dumps({k: meta.get(k) for k in shown})}")
    runner = PolicyRunner(meta, args.task, ChunkExecutor(args.mode, args.horizon, args.lead, args.blend),
                          SafetyFilter(vel_scale=args.vel_scale), stale_sec=args.stale_sec,
                          check_deviation=args.execute)

    def qos(depth):
        return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=depth)

    rclpy.init()
    node = Node("ftp1_wuji_runner")
    mono = time.monotonic
    node.create_subscription(JointState, P.ARM_STATE_TOPIC, lambda m: runner.on_arm(m, mono()), qos(1))
    node.create_subscription(JointState, P.HAND_STATE_TOPIC, lambda m: runner.on_hand(m, mono()), qos(5))
    for cam in meta["cameras"]:
        node.create_subscription(CompressedImage, P.CAMERA_TOPICS[cam],
                                 lambda m, c=cam: runner.on_image(c, m.data, mono()), qos(1))
    if meta["use_tactile"]:
        for finger, topic in P.TACTILE_TOPICS.items():
            node.create_subscription(PointCloud2, topic,
                                     lambda m, f=finger: runner.on_tactile(f, m, stamp_sec(m), mono()), qos(50))
    arm_pub = node.create_publisher(JointState, P.ARM_COMMAND_TOPIC, qos(1))
    hand_pub = node.create_publisher(JointState, P.HAND_COMMAND_TOPIC, qos(1))

    args.log_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    log = {"t": [], "cmd": [], "target": [], "measured": []}
    t_start = None

    def control_tick():
        nonlocal t_start
        now = mono()
        out = runner.control(now)
        if out is None:
            return
        if t_start is None:
            t_start = now
            print(f"[runner] first command ({'EXECUTE' if args.execute else 'dry run'})")
        if now - t_start > args.max_duration:
            runner.stop(f"max duration {args.max_duration}s reached")
            return
        log["t"].append(now)
        log["cmd"].append(np.concatenate([out["arm_deg"], out["hand_rad"]]))
        log["target"].append(out["target"])
        log["measured"].append(np.concatenate([runner.obs.arm_deg, runner.obs.hand_rad]))
        if not args.execute:
            return
        stamp = realtime_stamp()
        arm = JointState()
        arm.header.stamp = stamp
        arm.name = P.ARM_JOINT_NAMES
        arm.position = out["arm_deg"].tolist()
        arm.velocity = out["arm_vel"].tolist()
        arm_pub.publish(arm)
        hand = JointState()
        hand.header.stamp = stamp
        hand.name = P.HAND_JOINT_NAMES
        hand.position = out["hand_rad"].tolist()
        hand.velocity = out["hand_vel"].tolist()
        hand_pub.publish(hand)

    def inference_loop():
        while rclpy.ok() and runner.stopped_reason is None:
            request = runner.make_request(mono())
            if request is None:
                time.sleep(0.005)
                continue
            try:
                runner.on_reply(client.infer(request), mono())
            except Exception as exc:  # noqa: BLE001 - counted; runner stops after repeated failures
                print(f"[runner] inference error: {exc}")
                runner.on_error(exc, mono())

    node.create_timer(1.0 / args.control_hz, control_tick)
    worker = threading.Thread(target=inference_loop, daemon=True)
    print("[runner] waiting for sensor streams ...")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not worker.is_alive() and runner.stopped_reason is None and not runner.stale_streams(mono()):
                worker.start()
                print("[runner] all streams live, inference started")
            if runner.stopped_reason:
                print(f"[runner] stopped: {runner.stopped_reason}")
                break
    except KeyboardInterrupt:
        runner.stop("keyboard interrupt")
    finally:
        summary = runner.summary()
        print(json.dumps(summary, indent=1))
        np.savez_compressed(args.log_dir / f"run_{tag}.npz", **{k: np.asarray(v) for k, v in log.items()})
        (args.log_dir / f"run_{tag}.json").write_text(json.dumps({"args": vars(args) | {"log_dir": str(args.log_dir)},
                                                                   "server": meta, "summary": summary}, indent=1))
        client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
