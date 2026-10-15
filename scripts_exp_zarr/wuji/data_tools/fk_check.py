"""Compute FK for right palm link and check trajectory smoothness."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
import numpy as np
import kinpy as kp
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

URDF_R = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/arm_right.urdf'
URDF_L = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/arm_left.urdf'
BAG = '/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/dc-dexumi-003/20261008/ros2_bags/tactile_task3/rosbag_20261008_165822/rosbag_20261008_165822_0.mcap'

def bag_joints(path, topic):
    with open(path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _,ch,_,msg in reader.iter_decoded_messages(topics=[topic]):
            if hasattr(msg,'position') and len(msg.position)==7:
                yield msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9, np.deg2rad(np.array(msg.position))

def run_fk(urdf, topic, joint_names, palm_link, side):
    chain = kp.build_chain_from_urdf(open(urdf).read())
    data = list(bag_joints(BAG, topic))
    if not data: print(f'{side}: no data'); return
    data = sorted(data)
    times = np.array([d[0] for d in data])
    joints_rad = np.array([d[1] for d in data])

    poses = []
    for q in joints_rad:
        th = {n: float(v) for n, v in zip(joint_names, q)}
        tf = chain.forward_kinematics(th)
        if palm_link not in tf:
            print(f'{side}: {palm_link} not in FK output. Keys: {list(tf.keys())[:20]}'); break
        pos = tf[palm_link].pos
        rot = tf[palm_link].rot  # quaternion wxyz
        poses.append(np.concatenate([pos, rot]))

    poses = np.array(poses)
    pos = poses[:, :3]

    dt = np.diff(times)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = dp / dt  # m/s

    print(f'\n{side} FK on {palm_link} (N={len(poses)}):')
    print(f'  Position range: x=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]  '
          f'y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]  '
          f'z=[{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]  (m, base frame)')
    print(f'  Total path length: {dp.sum():.3f} m')
    print(f'  Speed: mean={speed.mean():.3f} m/s  max={speed.max():.3f} m/s  '
          f'99pct={np.percentile(speed,99):.3f} m/s')
    print(f'  Max single-step jump: {dp.max()*1000:.1f} mm at idx {dp.argmax()}')
    print(f'  Timestamps: dur={times[-1]-times[0]:.1f}s  max_gap={dt.max()*1000:.1f}ms')

    # Palm frame axes at a mid-trajectory pose
    mid = len(poses)//2
    q = joints_rad[mid]
    th = {n: float(v) for n, v in zip(joint_names, q)}
    tf = chain.forward_kinematics(th)
    from scipy.spatial.transform import Rotation
    quat_xyzw = np.array([poses[mid, 4], poses[mid, 5], poses[mid, 6], poses[mid, 3]])  # kinpy is wxyz
    R_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    print(f'\n  Palm rotation matrix at mid-trajectory (cols = local x,y,z in base frame):')
    for row, label in zip(R_mat, ['base_x', 'base_y', 'base_z']):
        print(f'    {label}: {row}')
    print(f'  -> palm local X in base: {R_mat[:,0].round(3)}')
    print(f'  -> palm local Y in base: {R_mat[:,1].round(3)}')
    print(f'  -> palm local Z in base: {R_mat[:,2].round(3)}')

    # Check zero-joint home position
    th_zero = {n: 0.0 for n in joint_names}
    tf0 = chain.forward_kinematics(th_zero)
    if palm_link in tf0:
        pos0 = tf0[palm_link].pos
        quat0 = tf0[palm_link].rot
        quat0_xyzw = [quat0[1], quat0[2], quat0[3], quat0[0]]
        R0 = Rotation.from_quat(quat0_xyzw).as_matrix()
        print(f'\n  Palm at zero joints: pos={pos0.round(3)}')
        print(f'  Palm rot at zero: local X={R0[:,0].round(3)} local Y={R0[:,1].round(3)} local Z={R0[:,2].round(3)}')

print('### RIGHT ARM ###')
run_fk(URDF_R, '/right_arm/joint_states',
       [f'Joint{i}_R' for i in range(1,8)], 'right_palm_link', 'RIGHT')
print('\n### LEFT ARM ###')
run_fk(URDF_L, '/left_arm/joint_states',
       [f'Joint{i}_L' for i in range(1,8)], 'left_palm_link', 'LEFT')
