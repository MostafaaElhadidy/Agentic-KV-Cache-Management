#!/usr/bin/env bash
# Safe wrapper for every GPU/vLLM job (added after crash #1, see docs/WORK_LOG.md).
#
#   scripts/gpu_run.sh <name> <timeout_s> <command...>
#   e.g. scripts/gpu_run.sh tiny_vanilla 600 python -m cachescout.run --config ... --system vanilla
#
# - refuses to start if another vLLM/GPU python job or heavy python process is running,
#   or if free RAM < MIN_FREE_GIB / GPU used > MAX_GPU_USED_MIB;
# - runs the command with `timeout`, unbuffered output, PYTHONPATH=src, log in logs/<name>.log;
# - a monitor logs free RAM/swap and GPU memory every 2 s to logs/<name>.monitor.log and
#   calls `sync` after each line, so the state before a crash survives a VM reset.
set -u
NAME="$1"; TIMEOUT_S="$2"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/logs/$NAME.log"; MON="$REPO/logs/$NAME.monitor.log"
MIN_FREE_GIB="${MIN_FREE_GIB:-6}"; MAX_GPU_USED_MIB="${MAX_GPU_USED_MIB:-1000}"
mkdir -p "$REPO/logs"

# only processes whose executable is python (not shells whose command line mentions it)
others=$(ps -eo pid=,comm=,args= | awk '$2 ~ /^(python|VLLM)/' | grep -E "vllm|cachescout|check_prefix|smoke_vllm|tune_constants|calibrate|EngineCore" || true)
if [ -n "$others" ]; then echo "REFUSE: other heavy python jobs running:"; echo "$others"; exit 3; fi
free_gib=$(free -g | awk '/^Mem:/{print $7}')
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "preflight: available RAM ${free_gib} GiB, GPU used ${gpu_used} MiB"
if [ "$free_gib" -lt "$MIN_FREE_GIB" ]; then echo "REFUSE: available RAM < ${MIN_FREE_GIB} GiB"; exit 3; fi
if [ "$gpu_used" -gt "$MAX_GPU_USED_MIB" ]; then echo "REFUSE: GPU already uses ${gpu_used} MiB"; exit 3; fi

( while true; do
    echo "$(date +%T) $(free -m | awk '/^Mem:/{printf "ram_used=%dM avail=%dM", $3, $7} /^Swap:/{printf " swap_used=%dM", $3}') gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)M" >> "$MON"
    sync "$MON" 2>/dev/null || sync
    sleep 2
  done ) &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

cd "$REPO"
echo "[gpu_run] $(date +%T) start: $*" | tee -a "$LOG"
PYTHONUNBUFFERED=1 PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  timeout --signal=TERM --kill-after=30 "$TIMEOUT_S" "$@" >> "$LOG" 2>&1
rc=$?
echo "[gpu_run] $(date +%T) exit code $rc" | tee -a "$LOG"
sync
exit $rc
