"""Smoke test: load pretrained ftp1_v0426_50kstep and run one dummy forward.

Uses domain=FreeTacMan (has normalization/FreeTacMan/ folder) with dummy
inputs that match the FreeTacMan tactile config: right_tactile_gripper,
image type, 2 function areas, 224x224x3.
"""
import time
import numpy as np

from openpi.policies import FTP1InferenceWrapper


def main():
    ckpt = "/DATA/YuanZhen/ftp1-policy/ckpts/ftp1_v0426_50kstep"

    t0 = time.perf_counter()
    wrapper = FTP1InferenceWrapper(
        checkpoint_dir=ckpt,
        domain_name="FreeTacMan",
        device="cuda",
        num_inference_steps=10,
    )
    # tactile_input_config_file has domain 'FreeTacMan_copy0' but not bare 'FreeTacMan';
    # patch the runtime type map so tactile normalization can resolve stats.
    wrapper.tactile_key_type_map = {"right_tactile_gripper": "image"}
    print(f"[load] took {time.perf_counter()-t0:.1f}s")

    state_dim = wrapper.get_state_dim()
    action_dim = wrapper.get_action_dim()
    action_horizon = wrapper.get_action_horizon()
    print(f"[shapes] state_dim={state_dim} action_dim={action_dim} action_horizon={action_horizon}")

    images = {"camera_ego_rgb_0": np.zeros((224, 224, 3), dtype=np.uint8)}
    state = np.zeros((1, state_dim), dtype=np.float32)
    tactiles = {"right_tactile_gripper": np.zeros((1, 2, 224, 224, 3), dtype=np.float32)}
    tactile_function_areas = {"right_tactile_gripper": [0, 1]}
    tactile_sensors = {"right_tactile_gripper": "FreeTacMan"}
    prompt = "insert the tube"

    print("[warmup] running first forward (compile + cudagraph)...")
    t0 = time.perf_counter()
    action = wrapper.infer(
        images=images,
        state=state,
        prompt=prompt,
        tactiles=tactiles,
        tactile_function_areas=tactile_function_areas,
        tactile_sensors=tactile_sensors,
    )
    print(f"[warmup] total {time.perf_counter()-t0:.1f}s  action.shape={action.shape} dtype={action.dtype}")
    print(f"[warmup] action stats: min={action.min():.4f} max={action.max():.4f} "
          f"mean={action.mean():.4f} std={action.std():.4f}")

    print("[bench] running 3 more forwards...")
    for i in range(3):
        t0 = time.perf_counter()
        action = wrapper.infer(
            images=images,
            state=state,
            prompt=prompt,
            tactiles=tactiles,
            tactile_function_areas=tactile_function_areas,
            tactile_sensors=tactile_sensors,
        )
        print(f"[bench {i}] {(time.perf_counter()-t0)*1000:.1f}ms  first-3-actions[0]={action[0, :3]}")


if __name__ == "__main__":
    main()
