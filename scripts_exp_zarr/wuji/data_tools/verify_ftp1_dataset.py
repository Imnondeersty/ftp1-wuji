"""
Verify that converted Wuji zarr structure matches FTP-1 requirements.

Usage:
    cd /wuji-vepfs/wuji-il/YuanZhen/ftp1-policy
    PYTHONPATH=src:$PYTHONPATH .venv/bin/python /wuji-vepfs/wuji-il/YuanZhen/data/_tools/verify_ftp1_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'ftp1-policy' / 'src'))

from openpi.replay_buffer import ReplayBuffer

ZARR_ROOT = Path('/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile-zarr/WujiTianji')

def test_single_domain():
    print('=== Test 1: Task2 zarr structure ===')
    task2_zarr = ZARR_ROOT / 'wuji_task2+place_the_bottle_into_the_tape_roll+.zarr'
    print(f'Loading {task2_zarr.name}...')

    buf = ReplayBuffer.create_from_path(str(task2_zarr), mode='r')
    print(f'  ✓ Episodes: {buf.n_episodes}, Total steps: {buf.n_steps}')

    # Check required keys for FTP-1
    required_keys = [
        'camera_ego_rgb',
        'right_wrist_camera_rgb',
        'left_wrist_camera_rgb',
        'right_arm_joints',
        'right_hand_joints',
        'right_wrist_pose',
        'right_tactile_data_fingertips',
        'right_tactile_type_fingertips',
        'right_tactile_sensor_fingertips',
        'right_tactile_area_fingertips',
        'sub_task_instruction',
    ]

    missing = [k for k in required_keys if k not in buf.data]
    if missing:
        print(f'  ✗ Missing keys: {missing}')
        return False
    print(f'  ✓ All required keys present')

    # Check shapes
    print(f'\n  Key shapes:')
    for key in sorted(required_keys):
        arr = buf.data[key]
        print(f'    {key}: {arr.shape} {arr.dtype}')

    # Check tactile metadata
    ep0 = buf.get_episode(0)
    print(f'\n  Tactile metadata (episode 0):')
    print(f'    Type: {ep0["right_tactile_type_fingertips"][0]}')
    print(f'    Sensor: {ep0["right_tactile_sensor_fingertips"][0]}')
    print(f'    Areas: {ep0["right_tactile_area_fingertips"][0]}')

    # Check tactile values
    tactile = ep0['right_tactile_data_fingertips']
    print(f'    Data shape: {tactile.shape}')
    print(f'    Data range: [{tactile.min():.2f}, {tactile.max():.2f}]')

    # Check instruction
    instr = ep0['sub_task_instruction'][0]
    print(f'\n  Instruction: "{instr}"')
    print()
    return True


def test_multi_domain():
    print('=== Test 2: Task3 zarr structure ===')
    task3_zarr = ZARR_ROOT / 'wuji_task3+place_the_small_tape_roll_on_the_large_tape_roll+.zarr'
    print(f'Loading {task3_zarr.name}...')

    buf = ReplayBuffer.create_from_path(str(task3_zarr), mode='r')
    print(f'  ✓ Episodes: {buf.n_episodes}, Total steps: {buf.n_steps}')

    # Sample episode
    ep0 = buf.get_episode(0)
    instr = ep0['sub_task_instruction'][0]
    print(f'  Instruction: "{instr}"')

    # Check tactile
    tactile = ep0['right_tactile_data_fingertips']
    print(f'  Tactile shape: {tactile.shape}')
    print(f'  Tactile range: [{tactile.min():.2f}, {tactile.max():.2f}]')
    print()
    return True


def test_tactile_config():
    print('=== Test 3: Generate tactile config ===')
    task2_zarr = ZARR_ROOT / 'wuji_task2+place_the_bottle_into_the_tape_roll+.zarr'

    buf = ReplayBuffer.create_from_path(str(task2_zarr), mode='r')
    ep0 = buf.get_episode(0)

    # Extract tactile metadata from first frame
    tactile_type = str(ep0['right_tactile_type_fingertips'][0])
    sensor_type = str(ep0['right_tactile_sensor_fingertips'][0])
    areas = ep0['right_tactile_area_fingertips'][0]

    print(f'  Tactile type: {tactile_type}')
    print(f'  Sensor type: {sensor_type}')
    print(f'  Areas: {areas}')

    # Generate config (FTP-1 format)
    config = {}
    for area_idx in areas:
        key = f'right_tactile_data_fingertips_{int(area_idx)}'
        config[key] = {
            'data_type': tactile_type,
            'sensor_type': sensor_type,
        }

    print(f'  Generated {len(config)} area configs')
    print(f'  Sample keys: {list(config.keys())[:3]}')

    # Save config
    config_path = task2_zarr.parent / 'tactile_input_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Saved to: {config_path}')
    print()
    return True


def test_summary():
    print('=== Summary ===')
    report_path = ZARR_ROOT.parent / 'conversion_report.json'
    with open(report_path) as f:
        report = json.load(f)

    print(f'  Conversion settings:')
    print(f'    FPS: {report["fps"]}')
    print(f'    Image size: {report["img_size"]}')
    print(f'    Keep left: {report["keep_left"]}')

    print(f'\n  Tasks:')
    for task_name, task_info in report['tasks'].items():
        print(f'    {task_name}:')
        print(f'      Episodes: {task_info["episodes"]}')
        print(f'      Frames: {task_info["frames"]}')
        print(f'      Instruction: "{task_info["instruction"]}"')
    print()


if __name__ == '__main__':
    try:
        ok1 = test_single_domain()
        ok2 = test_multi_domain()
        ok3 = test_tactile_config()
        test_summary()

        if ok1 and ok2 and ok3:
            print('✅ All tests passed!')
        else:
            print('❌ Some tests failed')
            sys.exit(1)
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
