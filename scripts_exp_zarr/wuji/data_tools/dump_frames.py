"""Dump head/wrist frames at a few points of one bag, with the MotionTrans-style 640x480 centre crop drawn."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

bag, out = sys.argv[1], sys.argv[2]
topics = ['/cam_head/color/image_raw/compressed', '/cam_right_wrist/color/image_raw/compressed', '/cam_left_wrist/color/image_raw/compressed']
msgs = {t: [] for t in topics}
with open(bag, 'rb') as f:
    for _, ch, _, m in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(topics=topics):
        msgs[ch.topic].append(m)
for frac in (0.15, 0.5, 0.8):
    row = []
    for t in topics:
        m = msgs[t][int(frac * (len(msgs[t]) - 1))]
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if t.startswith('/cam_head'):
            h, w = img.shape[:2]
            print(f'{t} {w}x{h}')
            cv2.rectangle(img, ((w - 640) // 2, (h - 480) // 2), ((w + 640) // 2, (h + 480) // 2), (0, 0, 255), 4)
            cv2.rectangle(img, ((w - h) // 2, 0), ((w + h) // 2, h), (0, 255, 0), 4)
        else:
            print(f'{t} {img.shape[1]}x{img.shape[0]}')
        row.append(cv2.resize(img, (int(img.shape[1] * 360 / img.shape[0]), 360)))
    cv2.imwrite(f'{out}/frames_{int(frac * 100):02d}.jpg', np.concatenate(row, axis=1))
