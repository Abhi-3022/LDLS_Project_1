# Shared helpers, sourced by every script and job in this folder. In order:
#   1. find this folder (inside Slurm the job script runs from a spool copy) and load config.sh
#   2. load_software: gcc 12 module, Spack cmake + numactl, the folder-local Python
#   3. work out which physical cores this job got, socket by socket
#   4. place <threads>: the command prefix that pins threads to cores and memory to their socket
#   5. timev, logged, write_manifest: the records every result folder carries
set -euo pipefail
HERE=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
if [[ ! -f $HERE/config.sh ]]; then
  echo "config.sh not found in $HERE: submit from this folder (cd scripts; ./submit.sh ...)" >&2
  exit 1
fi
cd "$HERE"
# shellcheck disable=SC1091
source "$HERE/config.sh"
mkdir -p "$HERE/logs" "$RESULTS_ROOT"
log() { echo "[$(date +%T)] $*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
gcc_major() { gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0; }

# ---- 2. software ------------------------------------------------------------------------------------------
load_software() {  # module gcc 12.3 + Spack cmake / numactl (names in config.sh); falls back to whatever is on PATH
  set +u   # module and spack shell functions use unset variables
  local m p r
  # Spack writes per-user caches to ~/.spack by default: keep them inside this folder instead
  export SPACK_USER_CACHE_PATH=$ENV_DIR/spack-user SPACK_USER_CONFIG_PATH=$ENV_DIR/spack-user/config
  mkdir -p "$ENV_DIR/spack-user"
  if have module; then
    for m in ${MODULES:-}; do module load "$m" >/dev/null 2>&1 && log "module: loaded $m" || log "note: module load $m failed"; done
    module load spack >/dev/null 2>&1 || true
  fi
  # `spack load` needs the shell function from setup-env.sh, even when a spack module is loaded
  if ! type -t spack 2>/dev/null | grep -q function; then
    for r in ${SPACK_ROOT:-} $SPACK_ROOTS; do
      if [[ -f $r/share/spack/setup-env.sh ]]; then export SPACK_ROOT=$r; . "$r/share/spack/setup-env.sh"; break; fi
    done
  fi
  if have spack; then
    local pkgs=${SPACK_PACKAGES:-}
    [[ $(gcc_major) -lt 9 ]] && pkgs="${SPACK_GCC_FALLBACK:-} $pkgs"
    for p in $pkgs; do
      # not inside $(...): spack load changes this shell's PATH, and a subshell would throw that away
      if spack load --first "$p" 2>"$ENV_DIR/spack-load.err" || spack load "$p" 2>"$ENV_DIR/spack-load.err"; then log "spack: loaded $p"
      else log "note: spack load $p failed: $(head -2 "$ENV_DIR/spack-load.err" | tr '\n' ' ')"; fi
    done
  else
    log "note: spack not found"
  fi
  set -u
  use_venv
  # cmake and make call cc / c++ (= the system gcc 8.5 on Rudra): point them at the gcc now on PATH
  export CC=$(command -v gcc) CXX=$(command -v g++)
  log "using gcc $(gcc -dumpversion 2>/dev/null) ($(command -v gcc)), cmake $(cmake --version 2>/dev/null | awk 'NR==1{print $3}'), numactl $(command -v numactl || echo none)"
}
uv_env() {  # folder-local uv: binary, managed Python, cache and venv all under $ENV_DIR
  export UV_CACHE_DIR=$ENV_DIR/uv-cache UV_PYTHON_INSTALL_DIR=$ENV_DIR/python UV_TOOL_DIR=$ENV_DIR/uv-tools
  export UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed UV=$ENV_DIR/bin/uv VENV=$ENV_DIR/venv
}
use_venv() {  # PY = the venv's Python if ./00_setup_login.sh env made one, else python3
  uv_env
  if [[ -x $VENV/bin/python ]]; then PY=$VENV/bin/python; export PATH="$PATH:$VENV/bin"   # appended: Spack wins
  else PY=python3; fi
}

# ---- 3. the physical cores of this allocation, socket 0 first (Rudra: 1 hardware thread per core) ---------
expand_cpulist() { local IFS=,; local r; for r in $1; do if [[ $r == *-* ]]; then seq "${r%-*}" "${r#*-}"; else echo "$r"; fi; done; }
ALLOWED=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)      # what Slurm's cgroup lets us use
declare -A _ALLOW=() _SEEN=() _NODES_OF_SOCK=()
for c in $(expand_cpulist "$ALLOWED"); do _ALLOW[$c]=1; done
_list=()
while IFS=, read -r cpu core sock node; do                             # one CPU per physical core
  [[ -n ${_ALLOW[$cpu]:-} ]] || continue
  k="$sock-$core"; [[ -n ${_SEEN[$k]:-} ]] && continue; _SEEN[$k]=1
  _list+=("$sock $cpu ${node:-$sock}")
done < <(lscpu -p=CPU,CORE,SOCKET,NODE | grep -v '^#')
PHYS_CPUS=(); PHYS_SOCK=()
while read -r s c n; do
  PHYS_CPUS+=("$c"); PHYS_SOCK+=("$s")
  case ",${_NODES_OF_SOCK[$s]:-}," in *",$n,"*) ;; *) _NODES_OF_SOCK[$s]="${_NODES_OF_SOCK[$s]:+${_NODES_OF_SOCK[$s]},}$n" ;; esac
done < <(printf '%s\n' "${_list[@]}" | sort -k1,1n -k2,2n)
PHYS=${#PHYS_CPUS[@]}                                                  # physical cores we got
SOCK0=${PHYS_SOCK[0]}
SOCK0_CORES=0; for s in "${PHYS_SOCK[@]}"; do [[ $s == "$SOCK0" ]] && SOCK0_CORES=$((SOCK0_CORES + 1)); done
SOCK0_NODES=${_NODES_OF_SOCK[$SOCK0]}                                  # NUMA node(s) of that socket
NSOCK=$(printf '%s\n' "${PHYS_SOCK[@]}" | sort -u | wc -l)
cpus_first() { local n=$1 IFS=,; echo "${PHYS_CPUS[*]:0:$n}"; }        # the first n physical cores, comma-separated

# ---- 4. placement -----------------------------------------------------------------------------------------
# place <threads>  ->  PLACE (command prefix) and PLACE_TAG (its name in the results).
# Threads go one per physical core (taskset + OMP_PROC_BIND=close OMP_PLACES=cores) and memory is bound to that
# socket's NUMA node (numactl --membind), so every run reads from local DRAM. With 4 cores we never leave one socket.
place() {
  local T=$1
  (( T <= SOCK0_CORES )) || { echo "place: $T threads do not fit the $SOCK0_CORES cores of socket $SOCK0 in this allocation" >&2; return 1; }
  PLACE=(taskset -c "$(cpus_first "$T")"); PLACE_TAG=socket
  if have numactl; then PLACE=(numactl --membind="$SOCK0_NODES" "${PLACE[@]}"); else PLACE_TAG=socket-nonuma; fi
}
OMP_ENV=(OMP_PROC_BIND=close OMP_PLACES=cores)
KV_ARGS=(); [[ $KV != f16 ]] && KV_ARGS=(-ctk "$KV" -ctv "$KV" -fa 1)   # quantized KV cache needs flash attention
PY=python3
BENCH=$LLAMA_BUILD/bin/llama-bench
COMPLETION=$LLAMA_BUILD/bin/llama-completion
M1Q4=$MODELS_DIR/Llama-3.2-1B-Instruct-Q4_K_M.gguf                    # the shared reference model

# ---- 5. records -------------------------------------------------------------------------------------------
# timev <file> cmd... : run cmd and write its peak RSS + elapsed time to <file> (GNU time -v format). Rudra's compute
# nodes have no /usr/bin/time, so the fallback is Python's getrusage (ru_maxrss of the child, kB). stdio passes through.
timev() {
  local f=$1; shift
  if [[ -x /usr/bin/time && -z ${TIMEV_PY:-} ]]; then /usr/bin/time -v -o "$f" "$@"; return; fi
  "$PY" -c 'import resource, subprocess, sys, time
t = time.time(); rc = subprocess.call(sys.argv[2:]); el = time.time() - t
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
open(sys.argv[1], "w").write(f"Command: {sys.argv[2:]}\nElapsed (wall clock) time (seconds): {el:.2f}\n"
    f"User time (seconds): {ru.ru_utime:.2f}\nSystem time (seconds): {ru.ru_stime:.2f}\n"
    f"Maximum resident set size (kbytes): {ru.ru_maxrss}\nExit status: {rc}\n")
sys.exit(rc)' "$f" "$@"
}

# logged <outdir> <tag> cmd... : run cmd; append the exact command line, exit code and time to <outdir>/commands.txt
logged() {
  local out=$1 tag=$2; shift 2
  local t0 rc=0; t0=$(date +%s%N)
  printf '[%s] %s:' "$(date -Is)" "$tag" >> "$out/commands.txt"; printf ' %q' "$@" >> "$out/commands.txt"; echo >> "$out/commands.txt"
  "$@" || rc=$?
  local ms=$(( ($(date +%s%N) - t0) / 1000000 ))
  printf '[%s] %s: exit=%s elapsed_s=%d.%03d\n' "$(date -Is)" "$tag" "$rc" $((ms / 1000)) $((ms % 1000)) >> "$out/commands.txt"
  return $rc
}

check_build() {  # llama-bench exists and was built on this CPU model (-march=native)
  [[ -x $BENCH ]] || { echo "llama-bench missing at $BENCH: run ./00_setup_login.sh on the login node" >&2; exit 1; }
  local here built
  here=$(awk -F: '/model name/{sub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo)
  built=$(cat "$LLAMA_BUILD/built_on_cpu.txt" 2>/dev/null || echo unknown)
  [[ $here == "$built" ]] || log "WARNING: built on '$built', running on '$here' (-march=native may not match)"
}

refuse_login_node() {  # benchmarks belong on a compute node: on the login node they disturb others and mean nothing
  if [[ -z ${SLURM_JOB_ID:-} && $(hostname) == login* && -z ${ALLOW_LOGIN_RUN:-} ]]; then
    echo "This is the login node ($(hostname)): submit the job instead, e.g. ./submit.sh report" >&2; exit 1
  fi
}

job_banner() {
  log "host=$(hostname) job=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none} allowed_cpus=$ALLOWED"
  log "physical cores=$PHYS sockets=$NSOCK socket0 cores=$SOCK0_CORES numa nodes=$SOCK0_NODES numactl=$(have numactl && echo yes || echo no) KV=$KV"
}

# write_manifest <outdir> : software, build, model checksum, allocation and settings, as "key: value" lines
write_manifest() {
  local o=$1/manifest.txt cxx
  cxx=$(awk -F= '/^CMAKE_CXX_COMPILER:/{print $2}' "$LLAMA_BUILD/CMakeCache.txt" 2>/dev/null)
  {
    echo "date_start: $(date -Is)"
    echo "job_id: ${SLURM_JOB_ID:-none}"; echo "job_name: ${SLURM_JOB_NAME:-none}"
    echo "node: $(hostname)"; echo "partition: ${SLURM_JOB_PARTITION:-none}"; echo "account: ${SLURM_JOB_ACCOUNT:-none}"
    echo "slurm_cpus_on_node: ${SLURM_CPUS_ON_NODE:-n/a}"; echo "slurm_mem_per_node_MB: ${SLURM_MEM_PER_NODE:-n/a}"
    echo "allowed_cpus: $ALLOWED"; echo "physical_cores_used: $PHYS"; echo "sockets: $NSOCK"; echo "socket0_cores: $SOCK0_CORES"
    echo "llama_cpp_commit: $(git -C "$LLAMA_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo "$LLAMA_COMMIT (pinned in config.sh)")"
    echo "llama_cpp_local_changes: $(git -C "$LLAMA_DIR" status --porcelain --untracked-files=no 2>/dev/null | wc -l) files"
    echo "llama_bench: $BENCH"
    echo "built_on_cpu: $(cat "$LLAMA_BUILD/built_on_cpu.txt" 2>/dev/null || echo unknown)"
    echo "compiler_cxx: $cxx ($("${cxx:-false}" -dumpfullversion 2>/dev/null || echo unknown))"
    for k in CMAKE_BUILD_TYPE GGML_NATIVE GGML_OPENMP BUILD_SHARED_LIBS; do
      echo "cmake_$k: $(awk -F= -v k="$k" '$1 ~ "^"k":" {print $2}' "$LLAMA_BUILD/CMakeCache.txt" 2>/dev/null)"
    done
    echo "march_native_resolves_to: $(gcc -march=native -Q --help=target 2>/dev/null | awk '$1=="-march="{print $2; exit}')"
    echo "simd: $(cat "$1/node/simd.txt" 2>/dev/null)"
    echo "modules_loaded: $( (have module && module -t list 2>&1) | grep -v -E '^Currently|^$|:$' | tr '\n' ' ')"
    echo "spack_loaded: $( (have spack && spack find --loaded --format '{name}@{version}%{compiler}' 2>/dev/null) | tr '\n' ' ')"
    echo "numactl: $(command -v numactl || echo none)"
    echo "model: $M1Q4"
    echo "model_bytes: $(stat -c %s "$M1Q4" 2>/dev/null)"
    echo "model_sha256: $(sha256sum "$M1Q4" | cut -d' ' -f1)"
    echo "model_sha256_expected: $(awk '/1B-Instruct-Q4_K_M/{print $1}' "$HERE/models.sha256")"
    echo "kv_cache: $KV (llama-bench args: ${KV_ARGS[*]:-none})"
    echo "repeats: $REPEATS"
    echo "omp_env: ${OMP_ENV[*]}"
  } > "$o" 2>/dev/null
}

copy_job_logs() {  # copy this job's Slurm .out/.err into the result folder
  local f; for f in "$HERE"/logs/*-"${SLURM_JOB_ID:-none}".{out,err}; do [[ -f $f ]] && cp "$f" "$1/"; done; return 0
}
:
