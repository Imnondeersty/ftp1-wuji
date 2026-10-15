"""Side-by-side check of a converted episode: head / right-wrist images next to the URDF FK skeleton of the
right arm + hand and the stored right_wrist_pose axes (red X, green Y, blue Z), in the Base_R frame.

    .venv/bin/python -I render_fk_check.py <episode.zarr> <episode_idx> <out.png>
"""
import sys

sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/ftp1-policy/data_processing')
import xml.etree.ElementTree as ET

import kinpy as kp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common.replay_buffer import ReplayBuffer
from scipy.spatial.transform import Rotation

URDF = '/wuji-vepfs/wuji-il/YuanZhen/wuji-openwam-dev/scripts/visualize_tianji/urdf/tianji_full/arm_right.urdf'
ARM = [f'Joint{i}_R' for i in range(1, 8)]
HAND = [f'right_finger{k}_joint{j}' for k in range(1, 6) for j in range(1, 5)]


def main(zarr_path, ep_idx, out):
    ep = ReplayBuffer.create_from_path(zarr_path, mode='r').get_episode(int(ep_idx))
    text = open(URDF).read()
    chain = kp.build_chain_from_urdf(text)
    edges = [(j.find('parent').get('link'), j.find('child').get('link')) for j in ET.fromstring(text).findall('joint')]
    T = len(ep['timestamps'])
    force = np.linalg.norm(ep['right_tactile_data_fingertips'], axis=-1)  # (T,5)
    frames = sorted(set(np.linspace(0, T - 1, 5).astype(int).tolist() + [int(force[:, 0].argmax())]))

    fig = plt.figure(figsize=(16, 4.2 * len(frames)))
    for row, t in enumerate(frames):
        q = dict(zip(ARM, ep['right_arm_joints'][t].tolist())) | dict(zip(HAND, ep['right_hand_joints'][t].tolist()))
        tf = chain.forward_kinematics(q)
        pos = {k: v.pos for k, v in tf.items()}
        ax_img = fig.add_subplot(len(frames), 4, 4 * row + 1)
        ax_img.imshow(ep['camera_ego_rgb'][t]); ax_img.set_axis_off()
        ax_img.set_title(f't={t} ({t / 20:.1f}s) head  |F| thumb..pinky=' + ' '.join(f'{f:.1f}' for f in force[t]), fontsize=8)
        ax_w = fig.add_subplot(len(frames), 4, 4 * row + 2)
        ax_w.imshow(ep['right_wrist_camera_rgb'][t]); ax_w.set_axis_off(); ax_w.set_title('right wrist cam', fontsize=8)
        for col, (elev, azim, name) in enumerate([(25, -150, 'arm'), (15, -120, 'hand close-up')]):
            ax = fig.add_subplot(len(frames), 4, 4 * row + 3 + col, projection='3d')
            for a, b in edges:
                if a in pos and b in pos:
                    finger = 'finger1' in b
                    ax.plot(*np.stack([pos[a], pos[b]]).T, color='orange' if finger else 'k', lw=1.5 if finger else 1)
            p = ep['right_wrist_pose'][t]
            R = Rotation.from_rotvec(p[3:]).as_matrix()
            for k, c in enumerate('rgb'):
                ax.quiver(*p[:3], *(R[:, k] * 0.08), color=c, lw=2)
            centre = pos['right_palm_link'] if col == 1 else np.array([0.3, -0.2, 0.3])
            half = 0.12 if col == 1 else 0.5
            ax.set_xlim(centre[0] - half, centre[0] + half); ax.set_ylim(centre[1] - half, centre[1] + half)
            ax.set_zlim(centre[2] - half, centre[2] + half); ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim); ax.set_title(f'FK {name} (Base_R; orange = thumb)', fontsize=8)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    fig.tight_layout()
    fig.savefig(out, dpi=70)
    print('saved', out, 'frames', frames)


if __name__ == '__main__':
    main(*sys.argv[1:4])
