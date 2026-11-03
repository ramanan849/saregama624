#!/usr/bin/env bash
# Run pipeline stages, in order, over a list of Saraga recording ids on one GPU.
#
#   scripts/run_stages.sh <mbids-file> <gpu> [stage ...]
#
# Default stages: sync stems f0 emissions (the ones that need no lyrics and no
# sections).  Every stage is resumable, so re-running the script after an
# interruption continues where it stopped.  Output goes to stdout; run it in a
# tmux window and tee it into annotations/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
list=$1; gpu=$2; shift 2
stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=(sync stems f0 emissions)
for st in "${stages[@]}"; do
  echo "=== $(date '+%F %T') stage $st on GPU $gpu"
  extra=()
  case $st in stems|f0|emissions) extra=(--device cuda) ;; esac
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m saregama.cli "$st" --mbids "$list" "${extra[@]}"
done
echo "=== $(date '+%F %T') all stages done"
