"""
Derive a per-taxel tactile variant of the v1 Wuji / Tianji FTP-1 zarr (convert_wuji_to_zarr.py output).

The v1 zarr and the raw bags are only read. Every array except the tactile ones is copied byte for byte, so
images, joints, poses and the 20 Hz timeline are identical to v1; only the tactile input changes:

    v1       {side}_tactile_*_fingertips     (T, 5, 3)    aggregate force agg_f{x,y,z} in N, areas 0-4
    taxel    {side}_tactile_*_taxelthumb     (T, 1, 120)  thumb taxels fx,fy,fz (40 x 3, dimensionless,
                                                          full scale 1.0, taxel-major), area 0
             {side}_tactile_*_taxelfingers   (T, 4, 102)  index..pinky taxels (34 x 3 each, same layout), areas 1-4

Taxel order is the PointCloud2 point order (constant across frames and bags, checked). The README spike rule is
applied at 100 Hz exactly as for v1 (same samples dropped), then taxels are linearly interpolated onto the v1 grid.
The grid is recomputed from the bag stamps and checked by re-deriving v1's aggregate forces bit-exact.

    python -I convert_wuji_taxel.py                                   # -> wuji-tactile-zarr-taxel
    python -I convert_wuji_taxel.py --drop-wrist-cams --out .../wuji-tactile-zarr-taxel-headcam
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

log = logging.getLogger('convert_wuji_taxel')

SRC = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr')
DEFAULT_OUT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr-taxel')
SENSOR = 'WujiHand2Taxel'
GROUPS = {'taxelthumb': (['thumb'], [0]), 'taxelfingers': (['index', 'middle', 'ring', 'pinky'], [1, 2, 3, 4])}
WRIST_CAMS = ('right_wrist_camera_rgb', 'left_wrist_camera_rgb')


def read_taxels(job):
    """Worker: one bag -> ({side: {group: (T, n_areas, 3P)}}, {side: v1 aggregate forces (T, 5, 3)}, xyz, qa)."""
    bag_dir, sides = job
    bag_dir = Path(bag_dir)
    qa = {'bag': bag_dir.name}
    info = yaml.safe_load((bag_dir / 'info.yaml').read_text())
    values = (info.get('robot_effective') or {}).get('values') or {}
    tactile_sides = [s for s in sides if (values.get(f'hand_{s}_hand') or {}).get('firmware_version') == v1.TACTILE_FIRMWARE_OK]
    if not info.get('success', False):
        return None, None, None, dict(qa, skipped='info.yaml success is not true')

    # Same topics as v1.read_bag so that the overlap window (and hence the grid) is identical.
    topics = list(v1.CAMERAS.values())
    topics += [f'/{s}_{part}/joint_states' for s in sides for part in ('arm', 'hand')]
    tac_topics = [f'/{s}_hand/tactile/{f}' for s in tactile_sides for f in v1.FINGERS]
    topics += tac_topics
    stamps = {t: [] for t in topics}
    values_ = {t: [] for t in tac_topics}
    xyz = {}
    with open(next(bag_dir.glob('*_0.mcap')), 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, _, msg in reader.iter_decoded_messages(topics=topics):
            t = channel.topic
            stamps[t].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if t in values_:
                values_[t].append(v1.parse_tactile(msg))
                if t not in xyz:
                    dt = np.dtype({'names': [x.name for x in msg.fields],
                                   'formats': [v1._PC2_DTYPES[x.datatype] for x in msg.fields],
                                   'offsets': [x.offset for x in msg.fields], 'itemsize': msg.point_step})
                    pts = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
                    xyz[t] = np.stack([pts['x'], pts['y'], pts['z']], axis=-1)

    for t in topics:  # identical stamp cleanup to v1.read_bag
        if len(stamps[t]) < 2:
            return None, None, None, dict(qa, skipped=f'missing stream {t}')
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

    t0 = max(s[0] for s in stamps.values())
    t1 = min(s[-1] for s in stamps.values())
    grid = np.arange(t0, t1, 1.0 / v1.FPS)
    T = len(grid)
    if T < 2 * v1.FPS:
        return None, None, None, dict(qa, skipped=f'too short ({T} frames)')
    qa['frames'] = T

    out, agg_out = {}, {}
    for side in tactile_sides:
        taxel_grid, forces = {}, np.zeros((T, len(v1.FINGERS), 3), dtype=np.float32)
        for k, finger in enumerate(v1.FINGERS):
            topic = f'/{side}_hand/tactile/{finger}'
            taxels = np.stack([v[0] for v in values_[topic]])
            agg = np.stack([v[1] for v in values_[topic]])
            drop, _ = v1.spike_mask(stamps[topic], taxels, agg)
            ok = ~drop
            forces[:, k] = interp1d(stamps[topic][ok], agg[ok], axis=0, bounds_error=False,
                                    fill_value=(agg[ok][0], agg[ok][-1]))(grid)
            tx = taxels[ok].reshape(ok.sum(), -1)
            taxel_grid[finger] = interp1d(stamps[topic][ok], tx, axis=0, bounds_error=False,
                                          fill_value=(tx[0], tx[-1]))(grid).astype(np.float32)
        out[side] = {g: np.stack([taxel_grid[f] for f in fingers], axis=1) for g, (fingers, _) in GROUPS.items()}
        agg_out[side] = forces
    return out, agg_out, {t.split('/')[-1]: p for t, p in xyz.items() if t.startswith('/right')}, qa


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=SRC, help='v1 output root (domain dir + conversion_report.json)')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--drop-wrist-cams', action='store_true', help='head camera only (like wuji-tactile-zarr-headcam)')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    v1_report = json.loads((args.src / 'conversion_report.json').read_text())
    sides = ['right', 'left'] if v1_report['keep_left'] else ['right']
    report = {'src': str(args.src), 'sensor': SENSOR, 'groups': {g: {'fingers': f, 'areas': a} for g, (f, a) in GROUPS.items()},
              'drop_wrist_cams': args.drop_wrist_cams, 'tasks': {}}
    xyz_ref = None
    for task, task_report in v1_report['tasks'].items():
        src_zarr = Path(task_report['zarr'])
        dst_zarr = args.out / v1.DOMAIN / src_zarr.name
        if dst_zarr.exists():
            if not args.overwrite:
                raise SystemExit(f'{dst_zarr} exists; pass --overwrite to replace it')
            shutil.rmtree(dst_zarr)
        src_data = zarr.open_group(str(src_zarr / 'data'), mode='r')
        tactile_keys = [k for k in src_data.array_keys() if k.startswith(tuple(f'{s}_tactile_' for s in sides))]
        drop = set(tactile_keys) | (set(WRIST_CAMS) if args.drop_wrist_cams else set())
        # Copy everything else byte for byte (directory store, zarr v2 layout).
        shutil.copytree(src_zarr, dst_zarr, ignore=lambda d, names: [n for n in names if Path(d) == src_zarr / 'data'
                                                                     and n in drop])
        episode_ends = zarr.open_group(str(src_zarr / 'meta'), mode='r')['episode_ends'][:]
        bags = [b for b in task_report['bags'] if 'skipped' not in b]
        assert len(bags) == len(episode_ends), (task, len(bags), len(episode_ends))
        bag_root = v1.TASKS[task][0]
        log.info('%s: %d bags, %s -> %s', task, len(bags), src_zarr.name, dst_zarr.parent)

        per_key = {}
        start = 0
        with mp.Pool(min(args.workers, len(bags))) as pool:
            jobs = [(str(bag_root / b['bag']), sides) for b in bags]
            for i, (taxels, agg, xyz, qa) in enumerate(pool.imap(read_taxels, jobs)):
                end = int(episode_ends[i])
                if taxels is None or qa['frames'] != end - start:
                    raise SystemExit(f"{qa['bag']}: {qa.get('skipped')} frames={qa.get('frames')} vs v1 {end - start}")
                for side, groups in taxels.items():
                    v1_forces = src_data[f'{side}_tactile_data_fingertips'][start:end]
                    if not np.array_equal(agg[side], v1_forces):
                        raise SystemExit(f"{qa['bag']}: re-derived {side} forces differ from v1 "
                                         f'(max {np.abs(agg[side] - v1_forces).max():.3g}); grid mismatch')
                    for g, data in groups.items():
                        per_key.setdefault((side, g), []).append(data)
                if xyz_ref is None:
                    xyz_ref = xyz
                elif any(not np.array_equal(xyz[f], xyz_ref[f]) for f in xyz):
                    raise SystemExit(f"{qa['bag']}: taxel layout differs from the first bag")
                start = end
                log.info('  %s: %d frames ok', qa['bag'], qa['frames'])

        dst_data = zarr.open_group(str(dst_zarr / 'data'), mode='a')
        cpr = ReplayBuffer.resolve_compressor('disk')
        n = int(episode_ends[-1])
        for (side, g), parts in per_key.items():
            areas = GROUPS[g][1]
            arrays = {
                f'{side}_tactile_data_{g}': np.concatenate(parts),
                f'{side}_tactile_area_{g}': np.tile(np.asarray(areas, dtype=np.int64), (n, 1)),
                f'{side}_tactile_sensor_{g}': np.full(n, SENSOR, dtype='<U16'),
                f'{side}_tactile_type_{g}': np.full(n, 'state', dtype='<U8'),
            }
            for key, value in arrays.items():
                # Chunked like ReplayBuffer.add_episode would for the first episode.
                chunks = get_optimal_chunks((len(parts[0]),) + value.shape[1:], value.dtype)
                arr = dst_data.create_array(name=key, shape=value.shape, dtype=value.dtype, fill_value=0, chunks=chunks,
                                            compressor=cpr)
                arr[:] = value
            log.info('  %s_tactile_data_%s %s', side, g, arrays[f'{side}_tactile_data_{g}'].shape)
        report['tasks'][task] = {'zarr': str(dst_zarr), 'episodes': len(episode_ends), 'frames': n,
                                 'removed': sorted(drop)}
    report['taxel_xyz_m'] = {f: p.round(6).tolist() for f, p in xyz_ref.items()}
    (args.out / 'conversion_report.json').write_text(json.dumps(report, indent=1))
    log.info('report -> %s', args.out / 'conversion_report.json')


if __name__ == '__main__':
    main()
