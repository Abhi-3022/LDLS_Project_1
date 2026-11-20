#!/usr/bin/env bash
# Submit the PARAM Rudra jobs with the settings in config.sh (run on the login node):
#   ./submit.sh reference           the shared reference run at 1-4 threads      -> results/ref_<jobid>/
#   ./submit.sh report              limits, hardware, bandwidth, timings, roofline -> results/rudra_<jobid>/
#   ./submit.sh reference report    both (the QOS lets 2 jobs run at once)
#   ./submit.sh --dry-run report    print the sbatch command, submit nothing
# The #SBATCH lines in each job ask for 4 CPUs, 8 GB, 4.5 min; this adds partition, account, QOS, reservation.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# shellcheck disable=SC1091
source ./config.sh
DRY=0; WHAT=()
for a in "$@"; do if [[ $a == --dry-run ]]; then DRY=1; else WHAT+=("$a"); fi; done
[[ ${#WHAT[@]} -gt 0 ]] || { sed -n 2,7p "$0"; exit 1; }
mkdir -p logs
if [[ ! -x $LLAMA_BUILD/bin/llama-bench ]]; then
  echo "llama-bench not built yet ($LLAMA_BUILD/bin): run ./00_setup_login.sh first" >&2
  [[ $DRY == 1 ]] || exit 1
fi
opts=(--time="$JOB_TIME" --partition="$PARTITION")
[[ -n $ACCOUNT ]] && opts+=(--account="$ACCOUNT")
[[ -n $QOS ]] && opts+=(--qos="$QOS")
[[ -n $RESERVATION ]] && opts+=(--reservation="$RESERVATION")
# shellcheck disable=SC2206
[[ -n $EXTRA_SBATCH ]] && opts+=($EXTRA_SBATCH)
submit() {  # submit <job file>
  if [[ $DRY == 1 ]]; then echo "sbatch --parsable ${opts[*]} $1"; return; fi
  sbatch --parsable "${opts[@]}" "$1"
}
for w in "${WHAT[@]}"; do
  case $w in
    reference) echo "reference job $(submit 10_reference.sbatch)" ;;
    report)
      ls -d "$RESULTS_ROOT"/limits_* >/dev/null 2>&1 || echo "note: no results/limits_* yet: run ./50_limits_login.sh first, or the report has no limits section" >&2
      echo "report job $(submit 60_report.sbatch)" ;;
    *) echo "unknown job '$w' (reference | report)" >&2; exit 1 ;;
  esac
done
echo "watch: squeue -u \$USER    logs: $HERE/logs/    results: $RESULTS_ROOT/"
