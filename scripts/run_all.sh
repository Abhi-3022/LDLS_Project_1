#!/usr/bin/env bash
# The whole project, in order. Every step writes CSVs under results/ and is resumable
# (bench_sweep.py and profile_ops.py skip finished runs). Measurements run one at a time on
# socket $NODE; nothing else should run on that socket meanwhile.
#
#   bash scripts/run_all.sh            # everything
#   bash scripts/run_all.sh lengths    # one step
set -euo pipefail
source "$(dirname "$0")/env.sh"
S="$PROJ/scripts"
step() { [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ]; }
ONLY="${1:-}"

step build      && bash "$S/01_build.sh"
step hw         && bash "$S/00_hwinfo.sh"
step stream     && bash "$S/02_stream.sh"
step models     && bash "$S/03_get_models.sh"
step roofline   && "$PY" "$S/roofline.py"                                   # predict BEFORE measuring
step correct    && bash "$S/04_correctness.sh"
step threads    && "$PY" "$S/bench_sweep.py" threads
step lengths    && "$PY" "$S/bench_sweep.py" lengths
step profile    && "$PY" "$S/profile_ops.py"
step experts    && "$PY" "$S/experiment_experts.py"
step ik         && bash "$S/ik_compare.sh"
step reference  && bash "$S/reference_run.sh" "$LLAMA_BIN/llama-bench" "$MODELS/Llama-3.2-1B-Instruct-Q4_K_M.gguf" "$RESULTS/csv/reference_$(hostname -s).csv" 55 "$T_MAIN"
step bonus      && bash "$PROJ/bonus_mamba3/run_bonus.sh"
step plots      && "$PY" "$S/plots.py" && "$PY" "$S/summarize.py"
echo "done"
