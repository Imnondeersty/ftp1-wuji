"""Inspect one frame of tactile data to understand the fields."""
import sys
sys.path.insert(0, '/wuji-vepfs/wuji-il/YuanZhen/data/_pylibs')
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np

BAG = '/wuji-vepfs/wuji-il/YuanZhen/data/wuji-tactile/dc-dexumi-003/20261008/ros2_bags/tactile_task3/rosbag_20261008_165822/rosbag_20261008_165822_0.mcap'

with open(BAG, 'rb') as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        topic = f'/right_hand/tactile/{finger}'
        msgs = list(reader.iter_decoded_messages(topics=[topic]))
        if not msgs:
            continue
        
        # Take a mid-trajectory frame with likely contact
        msg = msgs[len(msgs)//2].decoded_message
        
        # PointCloud2 fields
        import struct
        fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
        
        print(f'\n{finger.upper()} (N={len(msgs)} frames, mid-frame has {msg.width} points):')
        print(f'  Fields: {list(fields.keys())}')
        
        # Parse a few points
        data = msg.data
        row_step = msg.point_step
        
        points = []
        for i in range(min(3, msg.width)):
            offset = i * row_step
            point = {}
            for name, (off, dtype) in fields.items():
                val_offset = offset + off
                if dtype == 7:  # float32
                    val = struct.unpack_from('f', data, val_offset)[0]
                elif dtype == 4:  # uint16
                    val = struct.unpack_from('H', data, val_offset)[0]
                else:
                    val = None
                point[name] = val
            points.append(point)
        
        print(f'  First 3 taxels:')
        for i, p in enumerate(points):
            print(f'    taxel {i}: x={p.get("x",0):.4f} y={p.get("y",0):.4f} z={p.get("z",0):.4f}  '
                  f'fx={p.get("fx",0):.3f} fy={p.get("fy",0):.3f} fz={p.get("fz",0):.3f}  '
                  f'agg_fx={p.get("agg_fx",0):.3f} agg_fy={p.get("agg_fy",0):.3f} agg_fz={p.get("agg_fz",0):.3f}')
        
        # Compute stats across all taxels in this frame
        all_fx, all_fy, all_fz = [], [], []
        agg_fx_val, agg_fy_val, agg_fz_val = None, None, None
        for i in range(msg.width):
            offset = i * row_step
            fx = struct.unpack_from('f', data, offset + fields['fx'][0])[0]
            fy = struct.unpack_from('f', data, offset + fields['fy'][0])[0]
            fz = struct.unpack_from('f', data, offset + fields['fz'][0])[0]
            all_fx.append(fx)
            all_fy.append(fy)
            all_fz.append(fz)
            if i == 0:
                agg_fx_val = struct.unpack_from('f', data, offset + fields['agg_fx'][0])[0]
                agg_fy_val = struct.unpack_from('f', data, offset + fields['agg_fy'][0])[0]
                agg_fz_val = struct.unpack_from('f', data, offset + fields['agg_fz'][0])[0]
        
        all_fx = np.array(all_fx)
        all_fy = np.array(all_fy)
        all_fz = np.array(all_fz)
        
        print(f'  Per-taxel force range: fx=[{all_fx.min():.3f}, {all_fx.max():.3f}]  '
              f'fy=[{all_fy.min():.3f}, {all_fy.max():.3f}]  fz=[{all_fz.min():.3f}, {all_fz.max():.3f}]')
        print(f'  Aggregate force (same for all taxels): agg_fx={agg_fx_val:.3f}  agg_fy={agg_fy_val:.3f}  agg_fz={agg_fz_val:.3f}')
        print(f'  Sum of per-taxel forces: sum_fx={all_fx.sum():.3f}  sum_fy={all_fy.sum():.3f}  sum_fz={all_fz.sum():.3f}')
        
        break  # Only show one finger in detail
