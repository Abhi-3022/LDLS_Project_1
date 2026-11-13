#!/usr/bin/env bash
# Build everything from the pinned sources (CPU only). Safe to re-run.
#   llama.cpp      build-cpu  : benchmark build (gcc, -march=native, OpenMP)
#   llama.cpp      build-prof : same + GGML_OP_PROFILE per-operator timer (patches/ggml_op_profile.patch)
#   llama.cpp      build-exp  : + graph-node budget override for the expert-count experiment
#   ik_llama.cpp   build-cpu  : option-6 stretch comparison (GCC 9 compat shims in patches/ik_compat)
#   STREAM / readbw            : built by 02_stream.sh
#   bonus kernels              : bonus_mamba3/Makefile
set -euo pipefail
source "$(dirname "$0")/env.sh"
TP="$PROJ/third_party"
LLAMA_COMMIT=e30ad80fd7fa09a858bf8b8f4118e6479809486e
IK_COMMIT=656036d98503e154952f6c633cb3febc1a4f08b4
MAMBA_COMMIT=e9594ce1c732d97440f0332fdc43170a2294dbfa

# python env (torch CPU, plotting, gguf) and a CMake >= 3.18 (system cmake is 3.16)
if [ ! -x "$PY" ]; then
    uv venv --python 3.11 "$PROJ/.venv"
    uv pip install --python "$PY" -r "$PROJ/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cpu
fi

clone() {  # url dir commit
    [ -d "$TP/$2" ] || git clone -q "$1" "$TP/$2"
    git -C "$TP/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
    git -C "$TP/$2" checkout -q "$3" 2>/dev/null || echo "warning: $2 not at $3"
}
mkdir -p "$TP" "$RESULTS/logs"
clone https://github.com/ggml-org/llama.cpp.git llama.cpp "$LLAMA_COMMIT"
clone https://github.com/ikawrakow/ik_llama.cpp.git ik_llama.cpp "$IK_COMMIT"
clone https://github.com/state-spaces/mamba.git mamba "$MAMBA_COMMIT"
[ -f "$TP/stream/stream.c" ] || { mkdir -p "$TP/stream"; curl -sL https://www.cs.virginia.edu/stream/FTP/Code/stream.c -o "$TP/stream/stream.c"; }

# the op-profiler lives behind #ifdef GGML_OP_PROFILE; apply it if this checkout lacks it
if ! grep -q GGML_OP_PROFILE "$TP/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c"; then
    git -C "$TP/llama.cpp" apply "$PROJ/patches/ggml_op_profile.patch"
fi

COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF)
TARGETS=(llama-bench llama-completion llama-perplexity llama-quantize llama-simple llama-batched-bench llama-gguf test-backend-ops llama-tokenize)

cmake -S "$TP/llama.cpp" -B "$TP/llama.cpp/build-cpu" "${COMMON[@]}" -DLLAMA_BUILD_TESTS=ON > "$RESULTS/logs/build_llamacpp_cmake.log"
ninja -C "$TP/llama.cpp/build-cpu" "${TARGETS[@]}" > "$RESULTS/logs/build_llamacpp.log"

cmake -S "$TP/llama.cpp" -B "$TP/llama.cpp/build-prof" "${COMMON[@]}" -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_C_FLAGS="-DGGML_OP_PROFILE" > "$RESULTS/logs/build_prof_cmake.log"
ninja -C "$TP/llama.cpp/build-prof" llama-bench llama-completion > "$RESULTS/logs/build_prof.log"

# expert-count experiment build: graph node budget override (patches/llama_graph_nodes_experiment.patch)
if ! grep -q LDLS_EXPERIMENT_NODES "$TP/llama.cpp/src/llama-context.cpp"; then
    git -C "$TP/llama.cpp" apply "$PROJ/patches/llama_graph_nodes_experiment.patch"
fi
cmake -S "$TP/llama.cpp" -B "$TP/llama.cpp/build-exp" "${COMMON[@]}" -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-DLDLS_EXPERIMENT_NODES" > "$RESULTS/logs/build_exp_cmake.log"
ninja -C "$TP/llama.cpp/build-exp" llama-bench llama-perplexity > "$RESULTS/logs/build_exp.log"

C="$PROJ/patches/ik_compat"
cmake -S "$TP/ik_llama.cpp" -B "$TP/ik_llama.cpp/build-cpu" "${COMMON[@]}" -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS="-I$C -include $C/compat.h" > "$RESULTS/logs/build_ik_cmake.log"
ninja -C "$TP/ik_llama.cpp/build-cpu" llama-bench llama-perplexity > "$RESULTS/logs/build_ik.log"

make -C "$PROJ/bonus_mamba3" -s
echo "build OK: $LLAMA_BIN, $LLAMA_PROF_BIN, $IK_BIN, bonus_mamba3/build"
