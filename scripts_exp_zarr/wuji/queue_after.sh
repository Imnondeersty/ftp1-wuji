#!/usr/bin/env bash
# Wait for a running training process to exit, then start another run, but only if the previous run
# wrote its final checkpoint (so a crash does not silently cascade into the next job).
#   queue_after.sh <pid> <required_ckpt_dir> <log> -- <command...>
set -uo pipefail
pid="$1"; required="$2"; log="$3"; shift 4

echo "[queue] $(date '+%F %T') waiting for pid ${pid}; will require ${required}" >> "${log}"
while kill -0 "${pid}" 2>/dev/null; do sleep 60; done
sleep 120  # let workers exit and GPU memory free up
if [ ! -f "${required}/model.safetensors" ]; then
  echo "[queue] $(date '+%F %T') ${required}/model.safetensors missing; previous run did not finish, not starting" >> "${log}"
  exit 1
fi
echo "[queue] $(date '+%F %T') starting: $*" >> "${log}"
exec "$@" >> "${log}" 2>&1
