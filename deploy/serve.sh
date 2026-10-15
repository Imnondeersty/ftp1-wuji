#!/usr/bin/env bash
# Start the FTP-1 policy server for the Wuji/Tianji robot (GPU machine).
#   bash deploy/serve.sh <checkpoint_step_dir> [--port 8848] [--device cuda:0] [--cameras head ...]
# e.g.
#   bash deploy/serve.sh /wuji-vepfs/wuji-il/YuanZhen/data/ftp1_cache/checkpoints/ftp1/wuji_tianji_ft_v1/9999
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# same interpreter / path setup as scripts_exp_zarr/wuji/wuji_env.sh
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/wuji-vepfs/wuji-il/YuanZhen/cache/openpi}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"  # openpi imports jax; keep it off the GPU
exec "${REPO_ROOT}/.venv/bin/python" deploy/policy_server.py "$@"
