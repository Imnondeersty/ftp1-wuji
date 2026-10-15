"""
Convert Wuji / Tianji teleop recordings (ROS2 rosbag2 MCAP) into FTP-1 zarr.

The raw dataset is only read. Output: one FTP-1 domain directory with one zarr per task
(written with FTP-1's own ReplayBuffer), plus a conversion report next to it.

    python -I convert_wuji_to_zarr.py                       # both tasks, right arm/hand only
    python -I convert_wuji_to_zarr.py --tasks task3 --max-bags 2 --out /tmp/wuji_test
    python -I convert_wuji_to_zarr.py --keep-left           # also export the left arm/hand

Conventions (see tianji_fk.py and the FTP-1 data_processing/README.md):
    timeline   uniform FPS grid over the span where every used stream has data (header.stamp clock);
               joints / tactile are linearly interpolated, images take the nearest frame
    arm        {side}_arm_joints in rad (bag: deg); {side}_wrist_pose = FK palm link -> FTP-1 wrist frame,
               expressed in that arm's base frame (no camera_ego_pose; FTP-1 then uses identity)
    hand       {side}_hand_joints = 20 joint_states in rad, {side}_hand_joints_idx = FAAS slots
    tactile    per-fingertip aggregate force agg_f{x,y,z} (N) as type 'state', areas 0-4
               (the loader adds +24 for the left hand); README spike rule applied at the raw 100 Hz
"""

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import interp1d

sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_tools')
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/ftp1-policy/data_processing')
from common.replay_buffer import ReplayBuffer  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402
from tianji_fk import tianji_fk_wrist  # noqa: E402

log = logging.getLogger('convert_wuji')

DATA_ROOT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/dc-dexumi-003')
DEFAULT_OUT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr')
DOMAIN = 'WujiTianji'
TASKS = {
    'task2': (DATA_ROOT / '20260929/ros2_bags/tactile_task2', 'place the bottle into the tape roll'),
    'task3': (DATA_ROOT / '20261008/ros2_bags/tactile_task3', 'place the small tape roll on the large tape roll'),
}

FPS = 20
IMG_SIZE = 224
CAMERAS = {
    'camera_ego_rgb': '/cam_head/color/image_raw/compressed',
    'right_wrist_camera_rgb': '/cam_right_wrist/color/image_raw/compressed',
    'left_wrist_camera_rgb': '/cam_left_wrist/color/image_raw/compressed',
}
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
# Wuji hand 2, per finger joint1..4: thumb = cmc_flex, cmc_abd, mcp, ip; others = mcp_flex, mcp_abd, pip, dip
HAND_JOINTS_IDX = [1, 26, 2, 3, 7, 6, 8, 9, 12, 11, 13, 14, 17, 16, 18, 19, 22, 21, 23, 24]
TACTILE_GROUP = 'fingertips'
TACTILE_AREAS = [0, 1, 2, 3, 4]  # thumb..pinky tips, same ids for both hands
TACTILE_SENSOR = 'WujiHand2'
TACTILE_FIRMWARE_OK = 'v2.6.0'  # v2.5.1 tactile is unusable (dataset README)

_PC2_DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def expected_joint_names(topic):
    side = topic.split('/')[1].split('_')[0]
    if '_arm/' in topic:
        return [f'{side}_joint_{i}' for i in range(1, 8)]
    return [f'{side}_finger{k}_joint{j}' for k in range(1, 6) for j in range(1, 5)]


def parse_tactile(msg):
    """PointCloud2 -> (per-taxel fx,fy,fz (P,3) float32, aggregate force (3,) float32)."""
    dt = np.dtype({'names': [f.name for f in msg.fields], 'formats': [_PC2_DTYPES[f.datatype] for f in msg.fields],
                   'offsets': [f.offset for f in msg.fields], 'itemsize': msg.point_step})
    pts = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    taxels = np.stack([pts['fx'], pts['fy'], pts['fz']], axis=-1).astype(np.float32)
    agg = np.array([pts['agg_fx'][0], pts['agg_fy'][0], pts['agg_fz'][0]], dtype=np.float32)
    return taxels, agg


def spike_mask(stamps, taxels, agg):
    """Dataset README rule: a sample is a false spike if |agg| > 1 N, >= 6 lit taxels, >= 80% of the lit
    taxels share one byte-identical vector and |agg| >= median(|agg| over the previous 1 s) + 0.5 N.
    Returns (samples to drop: spike + next 5, number of spikes)."""
    mag = np.linalg.norm(agg, axis=1)
    lit = np.any(taxels != 0, axis=2)
    n_lit = lit.sum(axis=1)
    flag = np.zeros(len(mag), dtype=bool)
    for i in np.flatnonzero((mag > 1.0) & (n_lit >= 6)):
        vecs = np.ascontiguousarray(taxels[i][lit[i]]).view(np.dtype((np.void, 12))).ravel()
        if np.unique(vecs, return_counts=True)[1].max() < 0.8 * n_lit[i]:
            continue
        prev = mag[(stamps >= stamps[i] - 1.0) & (stamps < stamps[i])]
        if len(prev) and mag[i] >= np.median(prev) + 0.5:
            flag[i] = True
    drop = flag.copy()
    for k in range(1, 6):
        drop[k:] |= flag[:-k]
    return drop, int(flag.sum())


def square_resize(jpeg_bytes):
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nearest_index(stamps, query):
    j = np.clip(np.searchsorted(stamps, query), 1, len(stamps) - 1)
    return np.where(query - stamps[j - 1] <= stamps[j] - query, j - 1, j)


def read_bag(job):
    """Worker: one bag directory -> (episode dict or None, qa dict)."""
    bag_dir, instruction, keep_left = job
    bag_dir = Path(bag_dir)
    qa = {'bag': bag_dir.name}
    info = yaml.safe_load((bag_dir / 'info.yaml').read_text())
    values = (info.get('robot_effective') or {}).get('values') or {}
    sides = ['right', 'left'] if keep_left else ['right']
    tactile_sides = [s for s in sides if (values.get(f'hand_{s}_hand') or {}).get('firmware_version') == TACTILE_FIRMWARE_OK]
    qa['tactile_sides'] = tactile_sides
    if not info.get('success', False):
        qa['skipped'] = 'info.yaml success is not true'
        return None, qa

    topics = list(CAMERAS.values())
    topics += [f'/{s}_{part}/joint_states' for s in sides for part in ('arm', 'hand')]
    topics += [f'/{s}_hand/tactile/{f}' for s in tactile_sides for f in FINGERS]
    stamps = {t: [] for t in topics}
    values_ = {t: [] for t in topics}
    order = {}
    with open(next(bag_dir.glob('*_0.mcap')), 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            t = channel.topic
            stamps[t].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if t in CAMERAS.values():
                values_[t].append(bytes(msg.data))
            elif t.endswith('joint_states'):
                names = list(msg.name)
                if order.get(t, (None,))[0] != names:
                    order[t] = (names, [names.index(n) for n in expected_joint_names(t)])
                values_[t].append(np.asarray(msg.position, dtype=np.float64)[order[t][1]])
            else:
                values_[t].append(parse_tactile(msg))

    for t in topics:
        if len(stamps[t]) < 2:
            qa['skipped'] = f'missing stream {t}'
            return None, qa
        s = np.asarray(stamps[t])
        keep = np.concatenate([[True], np.diff(s) > 0]) if np.all(np.diff(s) >= 0) else None
        if keep is None:  # out-of-order stamps: sort, then drop duplicates
            perm = np.argsort(s, kind='stable')
            s = s[perm]
            values_[t] = [values_[t][i] for i in perm]
            keep = np.concatenate([[True], np.diff(s) > 0])
        stamps[t] = s[keep]
        values_[t] = [v for v, k in zip(values_[t], keep) if k]

    t0 = max(s[0] for s in stamps.values())
    t1 = min(s[-1] for s in stamps.values())
    grid = np.arange(t0, t1, 1.0 / FPS)
    T = len(grid)
    if T < 2 * FPS:
        qa['skipped'] = f'too short ({T} frames)'
        return None, qa
    qa['frames'] = T
    qa['max_gap_s'] = {t: round(float(np.diff(stamps[t]).max()), 3) for t in topics}

    ep = {'timestamps': (grid - grid[0]).astype(np.float32)}
    qa['image_dt_max_ms'] = {}
    for key, topic in CAMERAS.items():
        idx = nearest_index(stamps[topic], grid)
        qa['image_dt_max_ms'][key] = round(float(np.abs(stamps[topic][idx] - grid).max() * 1e3), 1)
        ep[key] = np.stack([square_resize(values_[topic][i]) for i in idx], dtype=np.uint8)

    for side in sides:
        arm = interp1d(stamps[f'/{side}_arm/joint_states'], np.stack(values_[f'/{side}_arm/joint_states']), axis=0)(grid)
        hand = interp1d(stamps[f'/{side}_hand/joint_states'], np.stack(values_[f'/{side}_hand/joint_states']), axis=0)(grid)
        ep[f'{side}_wrist_pose'] = tianji_fk_wrist(side, arm).astype(np.float32)
        ep[f'{side}_arm_joints'] = np.deg2rad(arm).astype(np.float32)
        ep[f'{side}_hand_joints'] = hand.astype(np.float32)
        ep[f'{side}_hand_joints_idx'] = np.tile(np.asarray(HAND_JOINTS_IDX, dtype=np.int64), (T, 1))

    qa['spikes'] = {}
    for side in tactile_sides:
        forces = np.zeros((T, len(FINGERS), 3), dtype=np.float32)
        for k, finger in enumerate(FINGERS):
            topic = f'/{side}_hand/tactile/{finger}'
            taxels = np.stack([v[0] for v in values_[topic]])
            agg = np.stack([v[1] for v in values_[topic]])
            drop, n_spikes = spike_mask(stamps[topic], taxels, agg)
            qa['spikes'][f'{side}_{finger}'] = n_spikes
            ok = ~drop
            forces[:, k] = interp1d(stamps[topic][ok], agg[ok], axis=0, bounds_error=False,
                                    fill_value=(agg[ok][0], agg[ok][-1]))(grid)
        ep[f'{side}_tactile_data_{TACTILE_GROUP}'] = forces
        ep[f'{side}_tactile_area_{TACTILE_GROUP}'] = np.tile(np.asarray(TACTILE_AREAS, dtype=np.int64), (T, 1))
        ep[f'{side}_tactile_sensor_{TACTILE_GROUP}'] = np.full(T, TACTILE_SENSOR, dtype='<U16')
        ep[f'{side}_tactile_type_{TACTILE_GROUP}'] = np.full(T, 'state', dtype='<U8')

    ep['sub_task_instruction'] = np.full(T, instruction, dtype='<U100')
    return ep, qa


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='output root (domain dir + report go here)')
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    parser.add_argument('--keep-left', action='store_true', help='also export left arm/hand (+ tactile if firmware ok)')
    parser.add_argument('--max-bags', type=int, default=None, help='first N bags per task (for testing)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    domain_dir = args.out / DOMAIN
    domain_dir.mkdir(parents=True, exist_ok=True)
    report = {'fps': FPS, 'img_size': IMG_SIZE, 'keep_left': args.keep_left, 'hand_joints_idx': HAND_JOINTS_IDX,
              'tasks': {}}
    for task in args.tasks:
        bag_root, instruction = TASKS[task]
        bags = sorted(p for p in bag_root.iterdir() if p.is_dir() and any(p.glob('*_0.mcap')))[:args.max_bags]
        zarr_path = domain_dir / f"wuji_{task}+{instruction.replace(' ', '_')}+.zarr"
        if zarr_path.exists():
            if not args.overwrite:
                raise SystemExit(f'{zarr_path} exists; pass --overwrite to replace it')
            shutil.rmtree(zarr_path)
        log.info('%s: %d bags -> %s', task, len(bags), zarr_path)
        buffer = ReplayBuffer.create_from_path(str(zarr_path), mode='a')
        qas = []
        with mp.Pool(min(args.workers, len(bags))) as pool:
            for ep, qa in pool.imap(read_bag, [(str(b), instruction, args.keep_left) for b in bags]):
                qas.append(qa)
                if ep is None:
                    log.warning('  skip %s: %s', qa['bag'], qa.get('skipped'))
                    continue
                buffer.add_episode(ep, compressors='disk')
                log.info('  %s: %d frames, spikes=%s', qa['bag'], qa['frames'], sum(qa['spikes'].values()))
        report['tasks'][task] = {'zarr': str(zarr_path), 'instruction': instruction, 'episodes': int(buffer.n_episodes),
                                 'frames': int(buffer.n_steps), 'bags': qas}
        log.info('%s done: %d episodes, %d frames', task, buffer.n_episodes, buffer.n_steps)
    report_path = args.out / 'conversion_report.json'
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    log.info('report -> %s', report_path)


if __name__ == '__main__':
    main()
