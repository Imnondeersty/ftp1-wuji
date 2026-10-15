"""
Test that converted Wuji zarr format is correct for FTP-1 training.

Usage:
    cd /wuji-vepfs/wuji-il/YuanZhen/ftp1-policy
    .venv/bin/python -I /wuji-vepfs/wuji-il/YuanZhen/data/_tools/test_load_wuji_zarr.py
"""

import sys
from pathlib import Path

sys.path.insert(0, 'data_processing')
from common.replay_buffer import ReplayBuffer

ZARR_ROOT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/_tools/scratch/convert_test/WujiTianji')

def main():
    print('=== Testing ReplayBuffer loading ===\n')

    # List all task zarr directories
    zarr_dirs = sorted(ZARR_ROOT.glob('*.zarr'))
    print(f'Found {len(zarr_dirs)} zarr datasets in {ZARR_ROOT}')

    if not zarr_dirs:
        print('✗ No zarr datasets found!')
        return

    # Load first zarr (task2)
    zarr_path = zarr_dirs[0]
    print(f'\nLoading zarr: {zarr_path.name}')

    try:
        buf = ReplayBuffer.create_from_path(str(zarr_path), mode='r')
        print(f'✓ Loaded zarr with {buf.n_steps} steps, {buf.n_episodes} episodes')

        # Check structure
        print('\n--- Zarr Structure ---')
        print(f'Keys: {list(buf.data.keys())}')

        # Check each key's shape and dtype
        for key in sorted(buf.data.keys()):
            arr = buf.data[key]
            print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')

        # Sample one episode
        print('\n--- Sample Episode 0 ---')
        step = buf.get_episode(0)
        for key in sorted(step.keys()):
            val = step[key]
            if hasattr(val, 'shape'):
                print(f'  {key}: shape={val.shape}, dtype={val.dtype}')
                if key == 'right_tactile_data_fingertips':
                    print(f'    Sample values (first 3 fingers): {val[0, :3, :]}')
            else:
                print(f'  {key}: {val}')

        print('\n✓ All checks passed!')

    except Exception as e:
        print(f'✗ Failed to load: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
