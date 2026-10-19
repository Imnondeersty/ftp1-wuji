"""
Add the teleop joint commands to an existing Wuji / Tianji FTP-1 zarr variant (convert_wuji_to_zarr.py output or one
derived from it), so that the loader can use them as action targets.

The source zarr and the raw bags are only read. Every source array is copied byte for byte and three arrays are added:

    {side}_arm_joints_cmd    (T, 7)   /{side}_arm/joint_commands in rad (bag: deg), same joint order as *_arm_joints
    {side}_hand_joints_cmd   (T, 20)  /{side}_hand/joint_commands in rad, same joint order as *_hand_joints
    {side}_wrist_pose_cmd    (T, 6)   FK of the arm commands, same convention as *_wrist_pose

joint_states is where the joints are, joint_commands is where teleop told them to go. They agree in free motion, but a
finger blocked by the object stops while its command keeps closing (thumb mcp ~14 deg on contact frames): for a
position-controlled hand that gap is the grip force. dataset_zarr.py takes the actions from `<key>_cmd` when it exists
and keeps the state input (and the base of the relative actions) on the measured joints.

The grid is recomputed from the bag stamps exactly like convert_wuji_to_zarr.read_bag and checked by re-deriving the
source's arm and hand joints bit-exact. Commands are linearly interpolated onto it; in about half the bags they start
up to ~7 ms after the grid, where the first command is held.

    python -I convert_wuji_cmd.py --src .../wuji-tactile-zarr --out .../wuji-tactile-zarr-cmd
    python -I convert_wuji_cmd.py --src .../wuji-tactile-zarr-taxel-headcam --out .../wuji-tactile-zarr-taxel-headcam-cmd
"""

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.interpolate import interp1d

sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_tools')
import convert_wuji_to_zarr as v1  # noqa: E402
from convert_wuji_to_zarr import make_reader, DecoderFactory  # noqa: E402
from common.replay_buffer import ReplayBuffer, get_optimal_chunks  # noqa: E402
from tianji_fk import tianji_fk_wrist  # noqa: E402

log = logging.getLogger('convert_wuji_cmd')

V1_REPORT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr/conversion_report.json')
PARTS = ('arm', 'hand')


def read_commands(job):
    """Worker: one bag -> ({side: {part: (cmd, state)}} on the v1 grid, arm in deg, hand in rad; qa)."""
    bag_dir, sides = job
    bag_dir = Path(bag_dir)
    qa = {'bag': bag_dir.name}
    info = yaml.safe_load((bag_dir / 'info.yaml').read_text())
    values = (info.get('robot_effective') or {}).get('values') or {}
    tactile_sides = [s for s in sides if (values.get(f'hand_{s}_hand') or {}).get('firmware_version') == v1.TACTILE_FIRMWARE_OK]
    if not info.get('success', False):
        return None, dict(qa, skipped='info.yaml success is not true')

    # Same topics as v1.read_bag so that the overlap window (and hence the grid) is identical.
    grid_topics = list(v1.CAMERAS.values())
    grid_topics += [f'/{s}_{part}/joint_states' for s in sides for part in PARTS]
    grid_topics += [f'/{s}_hand/tactile/{f}' for s in tactile_sides for f in v1.FINGERS]
    cmd_topics = [f'/{s}_{part}/joint_commands' for s in sides for part in PARTS]
    joint_topics = [t for t in grid_topics + cmd_topics if t.endswith(('joint_states', 'joint_commands'))]
    stamps = {t: [] for t in grid_topics + cmd_topics}
    values_ = {t: [] for t in joint_topics}
    order = {}
    with open(next(bag_dir.glob('*_0.mcap')), 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=grid_topics + cmd_topics):
            t = channel.topic
            stamps[t].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if t in values_:
                names = list(msg.name)
                if order.get(t, (None,))[0] != names:
                    order[t] = (names, [names.index(n) for n in v1.expected_joint_names(t)])
                values_[t].append(np.asarray(msg.position, dtype=np.float64)[order[t][1]])

    for t in stamps:  # identical stamp cleanup to v1.read_bag
        if len(stamps[t]) < 2:
            return None, dict(qa, skipped=f'missing stream {t}')
        s = np.asarray(stamps[t])
        perm = None
        keep = np.concatenate([[True], np.diff(s) > 0]) if np.all(np.diff(s) >= 0) else None
        if keep is None:
            perm = np.argsort(s, kind='stable')
            s = s[perm]
            keep = np.concatenate([[True], np.diff(s) > 0])
        stamps[t] = s[keep]
        if t in values_:
            vals = values_[t] if perm is None else [values_[t][i] for i in perm]
            values_[t] = [v for v, k in zip(vals, keep) if k]

    t0 = max(stamps[t][0] for t in grid_topics)
    t1 = min(stamps[t][-1] for t in grid_topics)
    grid = np.arange(t0, t1, 1.0 / v1.FPS)
    T = len(grid)
    if T < 2 * v1.FPS:
        return None, dict(qa, skipped=f'too short ({T} frames)')
    qa['frames'] = T

    out = {}
    for side in sides:
        out[side] = {}
        for part in PARTS:
            st, ct = f'/{side}_{part}/joint_states', f'/{side}_{part}/joint_commands'
            state = interp1d(stamps[st], np.stack(values_[st]), axis=0)(grid)  # as in v1.read_bag
            cv = np.stack(values_[ct])
            cmd = interp1d(stamps[ct], cv, axis=0, bounds_error=False, fill_value=(cv[0], cv[-1]))(grid)
            out[side][part] = (cmd, state)
            qa[f'{side}_{part}'] = {
                'cmd_start_gap_ms': round(max(0.0, float(stamps[ct][0] - grid[0])) * 1e3, 1),
                'cmd_end_gap_ms': round(max(0.0, float(grid[-1] - stamps[ct][-1])) * 1e3, 1),
                'cmd_max_gap_ms': round(float(np.diff(stamps[ct]).max()) * 1e3, 1),
                'max_abs_cmd_minus_state': np.round(np.abs(cmd - state).max(axis=0), 4).tolist(),
            }
    return out, qa


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, required=True, help='source variant root (contains the domain dir)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bags-report', type=Path, default=V1_REPORT,
                        help='v1 conversion_report.json: bag order of the episodes (derived variants keep it)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    v1_report = json.loads(args.bags_report.read_text())
    sides = ['right', 'left'] if v1_report['keep_left'] else ['right']
    report = {'src': str(args.src), 'bags_report': str(args.bags_report),
              'added': [f'{s}_{k}_cmd' for s in sides for k in ('arm_joints', 'hand_joints', 'wrist_pose')], 'tasks': {}}
    for task, task_report in v1_report['tasks'].items():
        name = Path(task_report['zarr']).name
        src_zarr = args.src / v1.DOMAIN / name
        dst_zarr = args.out / v1.DOMAIN / name
        if dst_zarr.exists():
            if not args.overwrite:
                raise SystemExit(f'{dst_zarr} exists; pass --overwrite to replace it')
            shutil.rmtree(dst_zarr)
        src_data = zarr.open_group(str(src_zarr / 'data'), mode='r')
        if any(k.endswith('_cmd') for k in src_data.array_keys()):
            raise SystemExit(f'{src_zarr} already has *_cmd arrays')
        shutil.copytree(src_zarr, dst_zarr)  # directory store, zarr v2 layout: copy everything byte for byte
        episode_ends = zarr.open_group(str(src_zarr / 'meta'), mode='r')['episode_ends'][:]
        bags = [b for b in task_report['bags'] if 'skipped' not in b]
        assert len(bags) == len(episode_ends), (task, len(bags), len(episode_ends))
        bag_root = v1.TASKS[task][0]
        log.info('%s: %d bags, %s -> %s', task, len(bags), src_zarr, dst_zarr.parent)

        per_key, bag_qa = {}, []
        start = 0
        with mp.Pool(min(args.workers, len(bags))) as pool:
            jobs = [(str(bag_root / b['bag']), sides) for b in bags]
            for i, (joints, qa) in enumerate(pool.imap(read_commands, jobs)):
                end = int(episode_ends[i])
                if joints is None or qa['frames'] != end - start:
                    raise SystemExit(f"{qa['bag']}: {qa.get('skipped')} frames={qa.get('frames')} vs source {end - start}")
                for side, parts in joints.items():
                    (arm_cmd, arm_state), (hand_cmd, hand_state) = parts['arm'], parts['hand']
                    for key, derived in ((f'{side}_arm_joints', np.deg2rad(arm_state).astype(np.float32)),
                                         (f'{side}_hand_joints', hand_state.astype(np.float32))):
                        src = src_data[key][start:end]
                        if not np.array_equal(derived, src):
                            raise SystemExit(f"{qa['bag']}: re-derived {key} differs from the source "
                                             f'(max {np.abs(derived - src).max():.3g}); grid mismatch')
                    per_key.setdefault(f'{side}_arm_joints_cmd', []).append(np.deg2rad(arm_cmd).astype(np.float32))
                    per_key.setdefault(f'{side}_hand_joints_cmd', []).append(hand_cmd.astype(np.float32))
                    per_key.setdefault(f'{side}_wrist_pose_cmd', []).append(tianji_fk_wrist(side, arm_cmd).astype(np.float32))
                bag_qa.append(qa)
                start = end
                log.info('  %s: %d frames ok', qa['bag'], qa['frames'])

        dst_data = zarr.open_group(str(dst_zarr / 'data'), mode='a')
        cpr = ReplayBuffer.resolve_compressor('disk')
        for key, parts in per_key.items():
            value = np.concatenate(parts)
            # Chunked like ReplayBuffer.add_episode would for the first episode.
            chunks = get_optimal_chunks((len(parts[0]),) + value.shape[1:], value.dtype)
            arr = dst_data.create_array(name=key, shape=value.shape, dtype=value.dtype, fill_value=0, chunks=chunks,
                                        compressor=cpr)
            arr[:] = value
            log.info('  %s %s', key, value.shape)
        report['tasks'][task] = {'zarr': str(dst_zarr), 'episodes': len(episode_ends), 'frames': int(episode_ends[-1]),
                                 'bags': bag_qa}
    (args.out / 'conversion_report.json').write_text(json.dumps(report, indent=1))
    log.info('report -> %s', args.out / 'conversion_report.json')


if __name__ == '__main__':
    main()
