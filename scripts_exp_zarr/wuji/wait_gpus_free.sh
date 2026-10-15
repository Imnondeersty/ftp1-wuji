#!/usr/bin/env bash
# Block until GPUs 0-7 have all been idle (< MAX_MIB used) for STABLE_CHECKS consecutive checks, POLL_S seconds
# apart. Used to queue a run behind other users' jobs whose PIDs are not visible from this container.
#   wait_gpus_free.sh <log>
set -uo pipefail
log="$1"
max_mib="${MAX_MIB:-1000}"
poll_s="${POLL_S:-60}"
stable_checks="${STABLE_CHECKS:-5}"

echo "[wait_gpus] $(date '+%F %T') waiting for 8 idle GPUs (< ${max_mib} MiB for ${stable_checks} checks)" >> "${log}"
ok=0
while [ "${ok}" -lt "${stable_checks}" ]; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v m="${max_mib}" '$1 >= m' | wc -l)
  n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | wc -l)
  if [ "${busy}" -eq 0 ] && [ "${n}" -eq 8 ]; then ok=$((ok + 1)); else ok=0; fi
  [ "${ok}" -lt "${stable_checks}" ] && sleep "${poll_s}"
done
echo "[wait_gpus] $(date '+%F %T') all 8 GPUs idle" >> "${log}"
