"""Check which Wuji hand joints actually move in a bag."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np

BAG = '/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/dc-dexumi-003/20261008/ros2_bags/tactile_task3/rosbag_20261008_165822/rosbag_20261008_165822_0.mcap'

def check_hand(topic, side):
    with open(BAG, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        msgs = [m.decoded_message for m in reader.iter_decoded_messages(topics=[topic])]
    
    if not msgs:
        print(f'{side}: no messages'); return
    
    first = msgs[0]
    print(f'\n{side.upper()} hand (N={len(msgs)}):')
    print(f'  Joint names: {first.name}')
    
    positions = []
    for m in msgs:
        if hasattr(m, 'position') and len(m.position) == 20:
            positions.append(np.array(m.position))
    
    if not positions:
        print('  No valid 20-joint messages'); return
    
    positions = np.rad2deg(np.array(positions))  # bag is in radians
    
    print(f'  Degrees range per joint:')
    ranges = []
    for i in range(20):
        mn, mx = positions[:, i].min(), positions[:, i].max()
        rng = mx - mn
        ranges.append(rng)
        finger = (i // 4) + 1
        joint_in_finger = (i % 4) + 1
        print(f'    finger{finger}_joint{joint_in_finger} (idx {i:2d}): [{mn:6.2f}, {mx:6.2f}]° range={rng:5.2f}°')
    
    print(f'\n  Joints with >5° motion (likely active): ', end='')
    active = [i for i, r in enumerate(ranges) if r > 5]
    for i in active:
        f, j = (i//4)+1, (i%4)+1
        print(f'finger{f}_joint{j}', end='  ')
    print()

check_hand('/right_hand/joint_states', 'right')
check_hand('/left_hand/joint_states', 'left')
