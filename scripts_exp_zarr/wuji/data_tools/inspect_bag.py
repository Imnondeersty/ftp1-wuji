"""Print the structure of one rosbag2 MCAP: per-topic message layout and a sample decode."""

import sys

sys.path.insert(0, "/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs")

import collections

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np

PC2_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pc2_to_struct(msg):
    dt = np.dtype(
        {
            "names": [f.name for f in msg.fields],
            "formats": [PC2_DTYPES[f.datatype] for f in msg.fields],
            "offsets": [f.offset for f in msg.fields],
            "itemsize": msg.point_step,
        }
    )
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def main(path):
    seen = collections.OrderedDict()
    counts = collections.Counter()
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages():
            counts[channel.topic] += 1
            if channel.topic in seen:
                continue
            seen[channel.topic] = (schema.name, ros_msg, message.log_time)
    for topic, (stype, msg, log_time) in seen.items():
        print(f"\n=== {topic}  [{stype}]  n={counts[topic]}")
        hdr = getattr(msg, "header", None)
        if hdr is not None:
            print(f"  frame_id={hdr.frame_id!r} stamp={hdr.stamp.sec}.{hdr.stamp.nanosec:09d} log_time={log_time}")
        if stype.endswith("PointCloud2"):
            print(f"  width={msg.width} height={msg.height} point_step={msg.point_step}")
            print("  fields:", [(fl.name, fl.offset, PC2_DTYPES[fl.datatype], fl.count) for fl in msg.fields])
            pts = pc2_to_struct(msg)
            for name in pts.dtype.names:
                col = pts[name]
                print(f"    {name:16s} min={col.min():.4g} max={col.max():.4g} first={col[:4]}")
        elif stype.endswith("JointState"):
            print(f"  names({len(msg.name)})={list(msg.name)}")
            print(f"  position={np.round(msg.position, 4).tolist()}")
            print(f"  velocity_len={len(msg.velocity)} effort_len={len(msg.effort)}")
        elif stype.endswith("PoseArray"):
            print(f"  num_poses={len(msg.poses)}")
            p = msg.poses[0]
            print(f"  pose0 pos=({p.position.x:.4f},{p.position.y:.4f},{p.position.z:.4f})")
        elif stype.endswith("CompressedImage"):
            print(f"  format={msg.format!r} bytes={len(msg.data)}")
        else:
            print("  ", str(msg)[:400])


if __name__ == "__main__":
    main(sys.argv[1])
