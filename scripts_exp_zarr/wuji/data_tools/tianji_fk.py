"""
Tianji arm FK -> FTP-1 wrist_pose.

Usage:
    from tianji_fk import tianji_fk_wrist
    wrist_pose = tianji_fk_wrist('right', joints_deg)  # (T,7) deg -> (T,6) xyz + rotvec, in Base_R / Base_L frame

Frames (derived from the URDF hand geometry, see __main__ self-test):
    {side}_palm_link:  X = palmar normal, Z = toward fingertips, Y = thumb side (right) / pinky side (left)
    FTP-1 wrist:       with hands flat (palm down, fingers forward) wrist == camera frame:
                       Z = toward fingertips, Y = palmar normal ("down"), X = "right"
                       (pinky side for the right hand, thumb side for the left hand)
    => R_wrist = R_palm @ Rz(-90deg) for both hands; the origin stays at the palm link (hand base).
"""
import numpy as np
import kinpy as kp
from scipy.spatial.transform import Rotation

URDF_DIR = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full'
R_PALM_TO_WRIST = Rotation.from_euler('z', -90, degrees=True).as_matrix()

_chains = {}


def _chain(side):
    if side not in _chains:
        with open(f'{URDF_DIR}/arm_{side}.urdf') as f:
            _chains[side] = kp.build_chain_from_urdf(f.read())
    return _chains[side]


def palm_mats(side, joints_deg):
    """(T,7) bag arm joints in degrees -> (T,4,4) palm link poses in the arm base frame."""
    chain = _chain(side)
    names = [f'Joint{i}_{side[0].upper()}' for i in range(1, 8)]
    q_all = np.deg2rad(np.atleast_2d(np.asarray(joints_deg, dtype=np.float64)))
    out = np.empty((len(q_all), 4, 4))
    for t, q in enumerate(q_all):
        out[t] = chain.forward_kinematics(dict(zip(names, q.tolist())))[f'{side}_palm_link'].matrix()
    return out


def tianji_fk_wrist(side, joints_deg):
    """(T,7) or (7,) degrees -> (T,6) or (6,) wrist pose [x, y, z, rx, ry, rz] (m, rotvec) in the arm base frame."""
    mats = palm_mats(side, joints_deg)
    rot = mats[:, :3, :3] @ R_PALM_TO_WRIST
    pose = np.concatenate([mats[:, :3, 3], Rotation.from_matrix(rot).as_rotvec()], axis=-1)
    return pose[0] if np.ndim(joints_deg) == 1 else pose


if __name__ == '__main__':
    # Self-test: in the palm frame, check the wrist axes against hand anatomy from the URDF.
    for side in ('right', 'left'):
        chain = _chain(side)
        q0 = {n: 0.0 for n in chain.get_joint_parameter_names()}

        def in_palm(q, link):
            tf = chain.forward_kinematics(q)
            return (np.linalg.inv(tf[f'{side}_palm_link'].matrix()) @ tf[link].matrix())[:3, 3]

        distal = in_palm(q0, f'{side}_finger3_tip_link')
        q_flex = dict(q0, **{f'{side}_finger3_joint1': 1.0, f'{side}_finger3_joint3': 1.0})
        palmar = in_palm(q_flex, f'{side}_finger3_tip_link') - distal
        to_pinky = in_palm(q0, f'{side}_finger5_link1') - in_palm(q0, f'{side}_finger2_link1')
        x_w, y_w, z_w = R_PALM_TO_WRIST.T
        unit = lambda v: v / np.linalg.norm(v)
        expected_x = to_pinky if side == 'right' else -to_pinky
        print(f'{side}: cos(wrist Z, toward fingertips)={unit(distal) @ z_w:+.2f}  '
              f'cos(wrist Y, palmar)={unit(palmar) @ y_w:+.2f}  '
              f'cos(wrist X, "right" = {"pinky" if side == "right" else "thumb"} side)={unit(expected_x) @ x_w:+.2f}')
    print('palm at zero joints (right):', tianji_fk_wrist('right', np.zeros(7)).round(3))
