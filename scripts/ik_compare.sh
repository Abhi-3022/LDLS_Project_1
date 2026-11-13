#!/usr/bin/env bash
# Option-6 stretch: llama.cpp vs ik_llama.cpp on the same GGUF files, same pinning, same threads.
#   llama.cpp     default (CPU repack of Q4_K on)
#   ik_llama.cpp  default (fused MoE kernel -fmoe 1)
#   ik_llama.cpp  -rtr 1 (run-time repack into its interleaved row formats)
# ik_llama.cpp is built with GCC 9 plus two local shims (patches/ik_compat), see 01_build.sh.
# Output: results/csv/bench_ik.csv
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODELS_IK="llama1b-q4,llama3b-q4,granite1b-q4,granite3b-q4"
OUT="$RESULTS/csv/bench_ik.csv"
"$PY" "$PROJ/scripts/bench_sweep.py" ik --models "$MODELS_IK" --engine llama --out "$OUT"
"$PY" "$PROJ/scripts/bench_sweep.py" ik --models "$MODELS_IK" --engine ik --out "$OUT"
"$PY" "$PROJ/scripts/bench_sweep.py" ik --models "$MODELS_IK" --engine ik --extra "-rtr 1" --tag rtr --out "$OUT"
