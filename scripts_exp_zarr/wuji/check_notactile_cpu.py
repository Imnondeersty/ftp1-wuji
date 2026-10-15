"""CPU-only pre-flight for the no-tactile baseline: same config path as train_wuji.sh, one real batch,
pretrained weights loaded, one forward + backward. Does not touch the GPUs.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:packages/openpi-client/src \
        .venv/bin/python scripts_exp_zarr/wuji/check_notactile_cpu.py <same trainer args as train_wuji.sh>
"""
import dataclasses
import logging
import sys

import torch

sys.path.insert(0, '.')
import openpi.models_pytorch.ftp1_pytorch  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data  # noqa: E402
from openpi.training.data_loader import _unwrap_dataset  # noqa: E402
from scripts.zarr_train_ftp1_utils import load_ftp1_model  # noqa: E402
from scripts.zarr_train_ftp1_utils import move_to_device  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    assert not torch.cuda.is_available(), 'run with CUDA_VISIBLE_DEVICES= so the GPUs stay untouched'
    config = _config.cli()
    config.finalize_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, _ = _data.create_torch_data_loader(
        data_config, model_config=config.model, action_horizon=config.model.action_horizon,
        batch_size=config.batch_size, shuffle=True, num_workers=0, val_num_workers=0, seed=config.seed,
        framework='pytorch', domain_batch_split=True, load_zarr_norm_stats=True)
    base_dataset = _unwrap_dataset(loader._data_loader.torch_loader.dataset)

    model_cfg = dataclasses.replace(config.model, dtype='float32',
                                    tactile_input_config_file=base_dataset.default_tactile_input_config_path)
    print(f'use_tactile_input={model_cfg.use_tactile_input}')
    model = openpi.models_pytorch.ftp1_pytorch.FTP1Pytorch(model_cfg)
    print('hpt_tactile_encoder:', type(model.hpt_tactile_encoder).__name__,
          '| tactile expert branch:', model.use_tactile_expert_branch)
    load_ftp1_model(checkpoint_dir=config.pytorch_weight_path, tactile_input_config_file=model_cfg.tactile_input_config_file,
                    device='cpu', model=model, load_hpt_tokenizers_strict=False, allow_partial_load=config.allow_partial_load)

    observation, actions = next(iter(loader))
    observation = move_to_device(observation, 'cpu')
    model.train()
    output = model(observation, actions.to(torch.float32))
    loss = output['loss'] if isinstance(output, dict) else torch.as_tensor(output).mean()
    loss = loss.mean()
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    n_param = sum(1 for p in model.parameters() if p.requires_grad)
    print(f'loss={loss.item():.4f} finite={torch.isfinite(loss).item()} params_with_grad={n_grad}/{n_param}')
    print('no grad:', [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None])


if __name__ == '__main__':
    main()
