# Shared settings for every script. Source it: `source scripts/env.sh`
# Override any variable from the environment before sourcing (e.g. NODE=1).

export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="$PROJ/.venv/bin/python"
export PATH="$PROJ/.venv/bin:$PATH"

export LLAMA_DIR="$PROJ/third_party/llama.cpp"
export LLAMA_BIN="$LLAMA_DIR/build-cpu/bin"        # benchmark build (no instrumentation)
export LLAMA_PROF_BIN="$LLAMA_DIR/build-prof/bin"  # GGML_OP_PROFILE build (patches/ggml_op_profile.patch)
export IK_BIN="$PROJ/third_party/ik_llama.cpp/build-cpu/bin"
export MODELS="$PROJ/models"
export RESULTS="$PROJ/results"

# --- CPU placement ---------------------------------------------------------
# Everything runs on one socket with memory bound to that socket's NUMA node,
# physical cores only (no SMT siblings). Core 1 on node 0 is permanently busy
# with the Weka storage daemon on this cluster, so it is excluded.
export NODE="${NODE:-0}"
export CORES="${CORES:-0 $(seq -s ' ' 2 55)}"     # 55 usable physical cores of socket 0
export THREAD_LIST="${THREAD_LIST:-1 2 4 8 16 24 32 48 55}"
export T_MAIN="${T_MAIN:-32}"                      # thread count for the length sweeps (chosen from thread scaling)
export REPS="${REPS:-3}"

# first N cores of $CORES as a numactl list, e.g. cores_for 4 -> 0,2,3,4
cores_for() {
    echo $CORES | tr ' ' '\n' | head -n "$1" | paste -sd, -
}

# command prefix that pins to the first N cores of $CORES with memory on $NODE;
# usable behind wrappers, e.g. /usr/bin/time $(pin_prefix 8) llama-bench ...
pin_prefix() {
    echo "env OMP_NUM_THREADS=$1 OMP_PROC_BIND=close OMP_PLACES=cores numactl --physcpubind=$(cores_for "$1") --membind=$NODE"
}

# run a command pinned to the first N cores of $CORES, memory on $NODE
pinned() {
    local n="$1"; shift
    $(pin_prefix "$n") "$@"
}

# --- Models (all quantized locally from F16 with the pinned llama.cpp) -----
# name|file|max_context
export MODEL_TABLE="
llama1b-q4|Llama-3.2-1B-Instruct-Q4_K_M.gguf|16384
llama1b-q8|Llama-3.2-1B-Instruct-Q8_0.gguf|16384
llama3b-q4|Llama-3.2-3B-Instruct-Q4_K_M.gguf|8192
granite1b-q4|granite-3.1-1b-a400m-instruct-Q4_K_M.gguf|16384
granite1b-q8|granite-3.1-1b-a400m-instruct-Q8_0.gguf|16384
granite3b-q4|granite-3.1-3b-a800m-instruct-Q4_K_M.gguf|8192
"
export MEM_BUDGET_GB=4.5
