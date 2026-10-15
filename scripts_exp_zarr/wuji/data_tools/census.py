"""Per-bag census: motion ranges of both arms/hands, tactile contact, firmware, duration."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = '/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/dc-dexumi-003'
FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']


def agg_of(msg):
    off = {f.name: f.offset for f in msg.fields}
    b = bytes(msg.data)
    return [np.frombuffer(b, '<f4', 1, off[k])[0] for k in ('agg_fx', 'agg_fy', 'agg_fz')]


def one(bag_dir):
    bag_dir = Path(bag_dir)
    mcap = next(bag_dir.glob('*.mcap'))
    info = yaml.safe_load((bag_dir / 'info.yaml').read_text())
    hands = info.get('robot_effective', {}).get('values', {})
    fw = {s: (hands.get(f'hand_{s}_hand') or {}).get('firmware_version') for s in ('left', 'right')}
    joints = {t: [] for t in ('/left_arm/joint_states', '/right_arm/joint_states', '/left_hand/joint_states', '/right_hand/joint_states')}
    tac = {f'/{s}_hand/tactile/{f}': [] for s in ('left', 'right') for f in FINGERS}
    head = []
    topics = list(joints) + list(tac) + ['/cam_head/color/image_raw/compressed']
    with open(mcap, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, msg, dec in reader.iter_decoded_messages(topics=topics):
            t = ch.topic
            if t in joints:
                joints[t].append(dec.position)
            elif t in tac:
                tac[t].append(agg_of(dec))
            else:
                head.append(dec.header.stamp.sec + dec.header.stamp.nanosec * 1e-9)
    out = {'bag': bag_dir.name, 'group': str(bag_dir.parent.relative_to(ROOT)), 'success': info.get('success'),
           'fw_left': fw['left'], 'fw_right': fw['right'], 'head_frames': len(head),
           'duration_s': round(head[-1] - head[0], 2) if len(head) > 1 else 0.0}
    for t, v in joints.items():
        a = np.asarray(v, dtype=np.float64)
        if 'hand' in t:
            a = np.rad2deg(a)
        key = t.split('/')[1]
        out[f'{key}_max_range_deg'] = round(float((a.max(0) - a.min(0)).max()), 2) if len(a) else None
    for t, v in tac.items():
        a = np.linalg.norm(np.asarray(v, dtype=np.float64), axis=1) if v else np.zeros(1)
        side, finger = t.split('/')[1].split('_')[0], t.split('/')[-1]
        out[f'tac_{side}_{finger}_max_N'] = round(float(a.max()), 2)
        out[f'tac_{side}_{finger}_frac_gt1N'] = round(float((a > 1.0).mean()), 3)
    return out


if __name__ == '__main__':
    bags = sorted(glob.glob(f'{ROOT}/*/ros2_bags/*/rosbag_*'))
    with mp.Pool(16) as pool:
        rows = pool.map(one, bags)
    json.dump(rows, open(sys.argv[1], 'w'), indent=1)
    print(f'{len(rows)} bags -> {sys.argv[1]}')
