#!/bin/bash
# Shared reference test on a PARAM Rudra CPU node. Prepared, NOT submitted: edit the
# partition/account lines for your allocation, then from the repo root run
#     sbatch scripts/rudra_reference.sbatch
# It builds llama.cpp (CPU only, same commit as our runs) in a scratch build directory,
# reuses models/Llama-3.2-1B-Instruct-Q4_K_M.gguf (copy the repo incl. that file to Rudra),
# and writes results/csv/reference_<node>.csv.
#SBATCH --job-name=ldls-ref
#SBATCH --partition=cpu            # <-- edit: Rudra CPU partition name
##SBATCH --account=<your_account> # <-- edit if your site needs it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive                # whole node, so nothing else shares the memory bus
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/rudra_reference_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMIT=e30ad80fd7fa09a858bf8b8f4118e6479809486e
SRC=third_party/llama.cpp
if [ ! -d "$SRC" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
git -C "$SRC" checkout -q "$COMMIT" 2>/dev/null || echo "note: could not check out $COMMIT, using current tree"
BUILD="$SRC/build-rudra"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$BUILD" -j "$(nproc)" --target llama-bench

lscpu > "results/logs/rudra_lscpu_$(hostname -s).txt"
bash scripts/reference_run.sh "$BUILD/bin/llama-bench" models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
