"""Thumb joint1/joint2 motion directions in the right palm frame (X palmar, Y radial, Z distal)."""
import numpy as np
import kinpy as kp

U = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/arm_right.urdf'
chain = kp.build_chain_from_urdf(open(U).read())
q0 = {n: 0.0 for n in chain.get_joint_parameter_names()}
# a typical observed thumb posture (rad) from the task3 sample
q0.update({'right_finger1_joint1': 0.3, 'right_finger1_joint2': 0.0, 'right_finger1_joint3': 0.2})


def tip(q):
    tf = chain.forward_kinematics(q)
    p = np.linalg.inv(tf['right_palm_link'].matrix())
    return (p @ tf['right_finger1_tip_link'].matrix())[:3, 3]


t0 = tip(q0)
for j in ['right_finger1_joint1', 'right_finger1_joint2', 'right_finger1_joint3']:
    q = dict(q0); q[j] += 0.3
    d = tip(q) - t0
    print(f'{j}: +0.3 rad -> tip moves {d.round(4)} (X palmar, Y radial, Z distal); |out-of-palm-plane X| share={abs(d[0]) / np.linalg.norm(d):.2f}')
tf = chain.forward_kinematics(q0)
P = np.linalg.inv(tf['right_palm_link'].matrix())
for k in (1, 2):
    m = P @ tf[f'right_finger1_link{k}'].matrix()
    print(f'joint{k} axis (link{k} y) in palm frame: {m[:3, 1].round(3)}')
