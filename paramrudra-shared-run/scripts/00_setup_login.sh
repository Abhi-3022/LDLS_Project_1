#!/usr/bin/env bash
# Run ONCE on a PARAM Rudra LOGIN node (compute nodes have no internet and no C headers):
#   ./00_setup_login.sh            everything below, in order (about 10 min)
#   ./00_setup_login.sh fetch      llama.cpp at commit 827466b, the 1B Q4_K_M model (sha256 checked), stream.c
#   ./00_setup_login.sh env        folder-local uv + Python 3.11 in .env/ (no pip --user, nothing in ~/.local)
#   ./00_setup_login.sh build      llama.cpp, CPU backend only, gcc 12, -march=native (login and compute nodes are
#                                  the same Xeon Gold 6240R, so the build matches the compute nodes)
#   ./00_setup_login.sh tools      STREAM and the read-only probe, compiled here for the report job
#   ./00_setup_login.sh doctor     list the compilers, cmake and numactl this cluster offers
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"
WHAT=${1:-all}

fetch() {
  mkdir -p "$(dirname "$LLAMA_DIR")" "$MODELS_DIR" "$WORK/third_party/stream"
  [[ -d $LLAMA_DIR/.git ]] || git clone "$LLAMA_REPO" "$LLAMA_DIR"
  git -C "$LLAMA_DIR" fetch -q origin 2>/dev/null || true
  git -C "$LLAMA_DIR" checkout -q "$LLAMA_COMMIT"
  log "llama.cpp at $(git -C "$LLAMA_DIR" rev-parse --short=12 HEAD)"
  local f=Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [[ -s $MODELS_DIR/$f ]] || curl -fL --retry 3 -C - -o "$MODELS_DIR/$f" "$HF_BASE/Llama-3.2-1B-Instruct-GGUF/resolve/main/$f"
  (cd "$MODELS_DIR" && grep " $f\$" "$HERE/models.sha256" | sha256sum -c)
  [[ -s $WORK/third_party/stream/stream.c ]] || curl -fsSL -o "$WORK/third_party/stream/stream.c" "$STREAM_URL"
}

pyenv() {  # uv binary (pinned version, unpacked, not installed), uv's own Python, a venv; all inside $ENV_DIR
  uv_env
  mkdir -p "$ENV_DIR/bin"
  if [[ ! -x $UV ]]; then
    curl -fL --retry 3 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
      | tar xz -C "$ENV_DIR/bin" --strip-components=1
  fi
  [[ -x $VENV/bin/python ]] || "$UV" venv --python "$PY_VERSION" "$VENV"
  # shellcheck disable=SC2086
  "$UV" pip install --python "$VENV/bin/python" $PY_PACKAGES
  log "venv ready: $VENV ($("$VENV/bin/python" -V))"
}

build() {
  load_software
  local CMAKE="" c gcclib
  for c in "$(command -v cmake || true)" "$VENV/bin/cmake"; do   # first cmake >= 3.24 (needed for --fresh)
    if [[ -x $c && $("$c" --version | awk 'NR==1{split($3,v,"."); print v[1]*100+v[2]}') -ge 324 ]]; then CMAKE=$c; break; fi
  done
  [[ -n $CMAKE ]] || { echo "no cmake >= 3.24 from Spack or $VENV: run ./00_setup_login.sh env first" >&2; exit 1; }
  (( $(gcc_major) >= 9 )) || { echo "gcc $(gcc -dumpversion) is too old (need >= 9): see ./00_setup_login.sh doctor and MODULES in config.sh" >&2; exit 1; }
  # rpath to gcc 12's own libstdc++: the binaries then run in jobs even without the module's LD_LIBRARY_PATH
  gcclib=$(dirname "$("$CXX" -print-file-name=libstdc++.so)")
  log "building with $CXX ($("$CXX" -dumpversion)), $CMAKE"
  # --fresh: forget any compiler cached by an earlier configure (an old build picked /usr/bin/c++ = gcc 8.5)
  "$CMAKE" --fresh -S "$LLAMA_DIR" -B "$LLAMA_BUILD" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$gcclib" -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$gcclib" \
        -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
  "$CMAKE" --build "$LLAMA_BUILD" -j "$BUILD_JOBS" --target llama-bench llama-completion
  awk -F: '/model name/{sub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo > "$LLAMA_BUILD/built_on_cpu.txt"
  "$BENCH" --help > /dev/null && log "llama-bench built: $BENCH"
}

tools() {  # compiled here because compute nodes have no C headers (job 187674 failed on stdio.h)
  load_software
  mkdir -p "$TOOLS_BIN"
  [[ -s $WORK/third_party/stream/stream.c ]] || curl -fsSL -o "$WORK/third_party/stream/stream.c" "$STREAM_URL"
  # STREAM: array size and iteration count are compile-time constants; -mcmodel=medium allows arrays > 2 GB
  gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE="$STREAM_N" -DNTIMES=10 \
      "$WORK/third_party/stream/stream.c" -o "$TOOLS_BIN/stream"
  gcc -O3 -march=native -fopenmp "$HERE/readbw.c" -o "$TOOLS_BIN/readbw"
  echo "gcc=$(gcc -dumpversion) march=native on $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo) stream_n=$STREAM_N ntimes=10" > "$TOOLS_BIN/build_info.txt"
  log "tools built in $TOOLS_BIN: $(cat "$TOOLS_BIN/build_info.txt")"
}

doctor() {
  set +e +u
  echo "--- system:  gcc $(gcc -dumpversion 2>/dev/null)   python3 $(python3 -V 2>&1)   numactl: $(command -v numactl || echo none)"
  have module && { echo "--- module avail gcc cmake"; module -t avail gcc cmake 2>&1 | head -40; }
  load_software
  have spack && { echo "--- spack find gcc cmake numactl"; spack find gcc cmake numactl 2>&1 | head -40; }
  echo "--- after load_software: gcc $(gcc -dumpversion 2>/dev/null) ($(command -v gcc))   cmake: $(cmake --version 2>/dev/null | head -1)"
}

case $WHAT in
  all) fetch; pyenv; build; tools ;;
  fetch) fetch ;; env) pyenv ;; build) build ;; tools) tools ;;
  doctor) doctor; exit 0 ;;
  *) echo "usage: $0 [all|fetch|env|build|tools|doctor]" >&2; exit 1 ;;
esac
log "setup done. Next: ./50_limits_login.sh, then ./submit.sh reference report"
