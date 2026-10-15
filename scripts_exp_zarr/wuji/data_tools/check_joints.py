"""Check joint angles from a bag against URDF limits."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import xml.etree.ElementTree as ET

def get_limits(urdf_path, joint_names):
    """Return (lower, upper) limits in radians for each joint."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get('name'): j for j in root.findall('joint')}
    lims = []
    for name in joint_names:
        j = joints.get(name)
        if j is None:
            lims.append((None, None))
        else:
            lim = j.find('limit')
            if lim is None:
                lims.append((None, None))
            else:
                lims.append((float(lim.get('lower')), float(lim.get('upper'))))
    return lims

def main(mcap_path, urdf_r, urdf_l):
    lims_r = get_limits(urdf_r, [f'Joint{i}_R' for i in range(1, 8)])
    lims_l = get_limits(urdf_l, [f'Joint{i}_L' for i in range(1, 8)])
    
    topics = ['/right_arm/joint_states', '/left_arm/joint_states']
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        msgs = list(reader.iter_decoded_messages(topics=topics))
    
    if not msgs:
        print('No joint_states in bag'); return
    
    for side, lims in [('right', lims_r), ('left', lims_l)]:
        side_msgs = [m for m in msgs if side in m.channel.topic]
        if not side_msgs:
            print(f'{side}: no messages'); continue
        
        # Collect all position arrays
        positions = []
        for m in side_msgs:
            msg = m.decoded_message
            if hasattr(msg, 'position') and len(msg.position) == 7:
                positions.append(np.array(msg.position))
        
        if not positions:
            print(f'{side}: no valid 7-joint messages'); continue
        
        positions = np.array(positions)  # (N, 7) in degrees
        positions_rad = np.deg2rad(positions)
        
        print(f'\n{side.upper()} (N={len(positions)}):')
        print(f'  Bag joint names: {side_msgs[0].decoded_message.name[:7]}')
        print(f'  Degrees range per joint:')
        for i in range(7):
            mn, mx = positions[:, i].min(), positions[:, i].max()
            print(f'    joint{i+1}: [{mn:7.2f}, {mx:7.2f}]°  ({positions_rad[:, i].min():.4f}, {positions_rad[:, i].max():.4f} rad)')
        
        print(f'  URDF limits (radians):')
        violations = []
        for i, (lo, hi) in enumerate(lims):
            if lo is None:
                print(f'    Joint{i+1}: no limit')
            else:
                print(f'    Joint{i+1}: [{lo:.4f}, {hi:.4f}]')
                below = (positions_rad[:, i] < lo - 0.01).sum()
                above = (positions_rad[:, i] > hi + 0.01).sum()
                if below or above:
                    violations.append(f'Joint{i+1}: {below} below, {above} above')
        
        if violations:
            print(f'  VIOLATIONS (>0.01 rad tolerance): {"; ".join(violations)}')
        else:
            print(f'  All within limits ✓')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
