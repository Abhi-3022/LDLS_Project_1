# PARAM Rudra settings for the Project 1 / Option 1 runs. Every script in this folder reads this file.
# Values come from the PARAM Rudra User Manual (C-DAC, 17 May 2024). Any value can be overridden for one
# submission from the environment, e.g.   REF_THREADS="24 48" ./submit.sh reference
#
# Node (partition "cpu"): 2 x Intel Xeon Gold 6240R (Cascade Lake, AVX-512 + VNNI), 2 x 24 = 48 cores, 2.4 GHz,
# 192 GB DDR4-2933 (6 channels per socket -> 140.8 GB/s theoretical per socket, 281.6 GB/s per node), 800 GB local SSD.

# --- Slurm ------------------------------------------------------------------------------------------
PARTITION=${PARTITION:-cpu}        # cpu = CPU-only nodes. Others: standard (default; CPU + high-memory nodes), hm, gpu
ACCOUNT=${ACCOUNT:-}               # only needed if you belong to several Slurm accounts (sacctmgr show assoc user=$USER)
QOS=${QOS:-}                       # cluster default QoS: up to 10 running jobs, 4 days max walltime
RESERVATION=${RESERVATION:-}       # set this if the course gets a reservation for the shared reference run
# QOS aicte_qip (ours) caps each user at cpu=4, node=1, 5 min, 2 jobs. So the reference job asks for 4 CPUs and
# 8 GB (shared node: other jobs may share its memory bandwidth; the manifest records the allocation).
# The optional sweeps/bonus jobs ask for a whole node and do not fit that QOS.
TIME_BUILD=${TIME_BUILD:-00:45:00}     # only if you build in a job instead of on the login node
TIME_REF=${TIME_REF:-00:04:30}         # Rudra limit for this job is 5 min; 1 run at 4 threads + node record take about 1-2 min
TIME_SWEEPS=${TIME_SWEEPS:-08:00:00}   # PER MODEL: the sweeps run as a 3-task job array, one node per model
TIME_BONUS=${TIME_BONUS:-01:30:00}
EXTRA_SBATCH=${EXTRA_SBATCH:-}     # any other sbatch flags, e.g. "--mail-type=END --mail-user=you@example.com"

# --- software ---------------------------------------------------------------------------------------
# What PARAM Rudra offers (checked on login02 with ./00_setup_login.sh doctor, Nov 2026):
#   module compiler/gcc/12.3          (system gcc is 8.5: too old for llama.cpp)
#   spack  cmake@3.29.2 (cascadelake, built with gcc@12.2.0), numactl@2.0.14, gcc@12.2.0, gcc@13.2.0
MODULES=${MODULES:-"compiler/gcc/12.3"}        # loaded first, with `module load`
SPACK_ROOTS=${SPACK_ROOTS:-"/home/apps/spack /home/apps/SPACK"}
SPACK_PACKAGES=${SPACK_PACKAGES:-"cmake@3.29.2%gcc@12.2.0 numactl@2.0.14%gcc@13.2.0"}   # numactl gives socket-local memory placement
SPACK_GCC_FALLBACK=${SPACK_GCC_FALLBACK:-"gcc@12.2.0"}   # used only if the gcc module is missing
# Python + cmake fallback live in a folder-local uv environment ($ENV_DIR): uv binary, uv-managed Python, cache and
# venv all stay inside this folder. Nothing goes to ~/.local, ~/.cache, ~/.bashrc or the system Python.
UV_VERSION=${UV_VERSION:-0.6.14}
PY_VERSION=${PY_VERSION:-3.11}
PY_PACKAGES=${PY_PACKAGES:-"numpy==1.26.4 cmake==3.29.6"}   # numpy: bonus correctness check; cmake: only used if Spack's fails
LLAMA_REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}
LLAMA_COMMIT=${LLAMA_COMMIT:-827466b}  # the commit used for every workstation result in the report
BUILD_JOBS=${BUILD_JOBS:-8}            # make -j on the login node (login nodes are shared: keep it modest)
HF_BASE=${HF_BASE:-https://huggingface.co/bartowski}
STREAM_URL=${STREAM_URL:-https://www.cs.virginia.edu/stream/FTP/Code/stream.c}

# --- where things go --------------------------------------------------------------------------------
# Default: inside this folder (in $HOME: 50 GB quota, needs ~6 GB). /scratch/$USER is faster but deletes files
# not accessed for 3 months, so copy results back if you use it.
WORK=${WORK:-$HERE}
LLAMA_DIR=${LLAMA_DIR:-$WORK/third_party/llama.cpp}
LLAMA_BUILD=${LLAMA_BUILD:-$LLAMA_DIR/build-rudra}
MODELS_DIR=${MODELS_DIR:-$WORK/models}
RESULTS_ROOT=${RESULTS_ROOT:-$WORK/results}
ENV_DIR=${ENV_DIR:-$WORK/.env}       # uv, its Python, caches and the venv (delete this folder to undo everything)
P1_DIR=${P1_DIR:-$HERE/..}         # the p1-dense repo (only the bonus job needs it: $P1_DIR/bonus_mamba3)

# --- run settings -----------------------------------------------------------------------------------
KV=${KV:-q8_0}                     # KV cache type. q8_0 = "No F16 anywhere" (adds -fa 1). f16 = llama.cpp default, for comparison only
REPEATS=${REPEATS:-3}              # the brief: every measurement 3 times
# Placement. Up to one socket (24 cores): threads on socket 0, memory bound to socket 0 (numactl --membind).
# Past one socket the model's memory has to live somewhere; CROSS_MODES lists what is tried for those runs:
#   interleave = numactl --interleave=all (pages round-robin over both sockets)
#   distribute = llama-bench --numa distribute (llama.cpp spreads its threads; memory where first touched)
#   none       = taskset only, Linux defaults
CROSS_MODES=${CROSS_MODES:-"interleave"}   # 48-thread memory placement(s); more modes = more runs ("interleave distribute none")
REF_THREADS=${REF_THREADS:-"4"}      # shared reference run: 4 threads = the 4 CPUs the QOS allows. Whole node: "24 48" (needs --exclusive)
REF_KVF16=${REF_KVF16:-0}            # 1 = one extra run with llama.cpp's default F16 KV cache (comparison only; rules say No F16)
REF_BANDWIDTH=${REF_BANDWIDTH:-0}     # 1 = also STREAM + read-only bandwidth in the reference job (not needed for the shared number)
SWEEP_THREADS=${SWEEP_THREADS:-socket}  # thread count of the prefill/decode sweeps: socket (= 24, local memory), all (= 48), or a number
SWEEP_CROSS_MODE=${SWEEP_CROSS_MODE:-interleave}       # used by the sweeps when SWEEP_THREADS is past one socket
THREAD_SWEEP=${THREAD_SWEEP:-"1 2 4 8 12 16 24 32 48"} # thread-scaling sweep (prefill 512 + decode 64)
STREAM_N=${STREAM_N:-400000000}    # STREAM elements: 3 x 8 B x N = 9.6 GB, far above 2 x 35.75 MB L3
READBW_GIB=${READBW_GIB:-8}        # readbw array size
BONUS_MAX_THREADS=${BONUS_MAX_THREADS:-48}
BONUS_LAYERS_MAX=${BONUS_LAYERS_MAX:-512}
