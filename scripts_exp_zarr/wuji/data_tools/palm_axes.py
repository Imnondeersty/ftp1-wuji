"""Find anatomical axes of {side}_palm_link from URDF hand geometry (finger bases, flexion direction)."""
import sys
import numpy as np
import kinpy as kp

D = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full'


def link_in_palm(chain, q, side, link):
    tf = chain.forward_kinematics(q)
    p = tf[f'{side}_palm_link'].matrix()
    m = tf[link].matrix()
    return (np.linalg.inv(p) @ m)[:3, 3]


for side in ['right', 'left']:
    chain = kp.build_chain_from_urdf(open(f'{D}/arm_{side}.urdf').read())
    names = chain.get_joint_parameter_names()
    q0 = {n: 0.0 for n in names}
    print(f'\n=== {side} hand, positions in {side}_palm_link frame (m), all joints 0')
    for k, name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky'], start=1):
        base = link_in_palm(chain, q0, side, f'{side}_finger{k}_link1')
        tip = link_in_palm(chain, q0, side, f'{side}_finger{k}_tip_link')
        print(f'  finger{k} ({name:6s}) base={base.round(4)}  tip={tip.round(4)}')
    # flexion direction: bend middle finger MCP (joint1) and PIP (joint3)
    tip0 = link_in_palm(chain, q0, side, f'{side}_finger3_tip_link')
    q1 = dict(q0); q1[f'{side}_finger3_joint1'] = 1.0; q1[f'{side}_finger3_joint3'] = 1.0
    tip1 = link_in_palm(chain, q1, side, f'{side}_finger3_tip_link')
    d = tip1 - tip0
    print(f'  middle tip displacement when flexing j1,j3 +1 rad: {d.round(4)}  -> palmar dir ~ {np.round(d / np.linalg.norm(d), 2)}')
    # abduction direction: index joint2 +0.3
    tipa0 = link_in_palm(chain, q0, side, f'{side}_finger2_tip_link')
    qa = dict(q0); qa[f'{side}_finger2_joint2'] = 0.3
    tipa1 = link_in_palm(chain, qa, side, f'{side}_finger2_tip_link')
    print(f'  index tip displacement when joint2 +0.3 rad: {(tipa1 - tipa0).round(4)}')
