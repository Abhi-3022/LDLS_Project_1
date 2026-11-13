#!/usr/bin/env bash
# Mamba-3 decode-step bonus, end to end:
#   build -> reference data (official PyTorch refs on CPU) -> 1000-step verification
#   -> timing sweep (pinned like every other measurement) -> ggml integration -> analysis
set -euo pipefail
source "$(dirname "$0")/../scripts/env.sh"
B="$PROJ/bonus_mamba3"
cd "$B"
mkdir -p results

make -s
[ -f data/mimo4/config.txt ] || (cd ref && "$PY" make_reference.py)
# real-weight cases: layer 0 of the released 1.5B checkpoints (downloaded by 03_get_models.sh)
if [ -f "$MODELS/mamba3/mamba3-mimo-1.5b/pytorch_model.bin" ] && [ ! -f data/real_mimo4/config.txt ]; then
    (cd ref && "$PY" make_reference.py --real)
fi

# correctness: all ISAs, 1 and 8 threads, 1000 steps each
pinned 8 ./build/verify data results/verify.csv 1 8 | tee results/verify.txt

# timing: same socket/core placement as the llama.cpp runs
rm -f results/bench.csv
pinned 32 ./build/bench results/bench.csv --threads 1,2,4,8,16,32 --reps "$REPS" \
    --isa scalar,avx2,avx512 --variants m2,siso,mimo2,mimo4,mimo8 --regimes hot,cold | tee results/bench.txt

# ggml: SSM_SCAN baseline + Mamba-3 as a ggml custom op
pinned 32 ./build/ggml_m3 data results/ggml_m3.csv 1 8 32 | tee results/ggml_m3.txt

"$PY" analysis.py
