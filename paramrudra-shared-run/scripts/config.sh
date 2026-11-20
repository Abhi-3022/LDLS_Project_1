# Settings for the PARAM Rudra runs (Project 1, Option 1). Every script in this folder sources this file.
# Any value can be overridden for one command from the environment, e.g.   KV=f16 ./submit.sh reference
#
# What a Rudra "cpu" node is (PARAM Rudra User Manual, C-DAC, May 2024; checked with lscpu in job 187685):
#   2 x Intel Xeon Gold 6240R (Cascade Lake, AVX-512 + VNNI), 2 x 24 cores, 1 thread per core, 2.4 GHz base,
#   192 GB DDR4-2933, 6 channels per socket = 140.8 GB/s theoretical per socket, one NUMA node per socket.
#
# What we are allowed (QOS aicte_qip, recorded by ./50_limits_login.sh):
#   at most 4 CPUs on 1 node, 5 minutes per job, 2 jobs running at once; the node is shared with other users.
# So both jobs ask for 4 CPUs, 8 GB and 4.5 minutes (the #SBATCH lines in the .sbatch files).

# --- Slurm -------------------------------------------------------------------------------------------------
PARTITION=${PARTITION:-cpu}        # CPU-only nodes rdcn01-36
ACCOUNT=${ACCOUNT:-}               # only if you belong to several accounts (ours: nsmext, the default)
QOS=${QOS:-}                       # empty = your default QOS (ours: aicte_qip)
RESERVATION=${RESERVATION:-}       # set if the course gets a reservation
JOB_TIME=${JOB_TIME:-00:04:30}     # QOS limit is 5 min; both jobs need under 3
EXTRA_SBATCH=${EXTRA_SBATCH:-}     # any other sbatch flags, e.g. "--mail-type=END"

# --- software: what Rudra offers (found with ./00_setup_login.sh doctor) -----------------------------------
MODULES=${MODULES:-"compiler/gcc/12.3"}       # system gcc is 8.5, too old for llama.cpp
SPACK_ROOTS=${SPACK_ROOTS:-"/home/apps/spack /home/apps/SPACK"}
SPACK_PACKAGES=${SPACK_PACKAGES:-"cmake@3.29.2%gcc@12.2.0 numactl@2.0.14%gcc@13.2.0"}   # numactl: socket-local memory
SPACK_GCC_FALLBACK=${SPACK_GCC_FALLBACK:-"gcc@12.2.0"}  # used only if the gcc module is missing
# Python (for timing and the reports) lives in a folder-local uv environment, $ENV_DIR. Nothing is installed
# into ~/.local, ~/.cache, ~/.bashrc or the system Python.
UV_VERSION=${UV_VERSION:-0.6.14}
PY_VERSION=${PY_VERSION:-3.11}
PY_PACKAGES=${PY_PACKAGES:-"cmake==3.29.6"}   # cmake from PyPI is only a fallback if Spack's is missing
LLAMA_REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}
LLAMA_COMMIT=${LLAMA_COMMIT:-827466b}  # the same commit as every workstation result in the report
BUILD_JOBS=${BUILD_JOBS:-8}            # make -j on the shared login node: keep it modest
HF_BASE=${HF_BASE:-https://huggingface.co/bartowski}
STREAM_URL=${STREAM_URL:-https://www.cs.virginia.edu/stream/FTP/Code/stream.c}

# --- where things go (all inside this folder by default) ---------------------------------------------------
WORK=${WORK:-$HERE}
LLAMA_DIR=${LLAMA_DIR:-$WORK/third_party/llama.cpp}
LLAMA_BUILD=${LLAMA_BUILD:-$LLAMA_DIR/build-rudra}
MODELS_DIR=${MODELS_DIR:-$WORK/models}
RESULTS_ROOT=${RESULTS_ROOT:-$WORK/results}
TOOLS_BIN=${TOOLS_BIN:-$WORK/tools-bin}   # stream + readbw, compiled on the login node (compute nodes have no C headers)
ENV_DIR=${ENV_DIR:-$WORK/.env}            # uv, its Python, caches, venv: delete it to undo the setup

# --- measurement settings ----------------------------------------------------------------------------------
KV=${KV:-q8_0}                 # KV cache type. q8_0 + flash attention = the project rule "No F16 anywhere"
REPEATS=${REPEATS:-3}          # every measurement 3 times
REF_THREADS=${REF_THREADS:-"1 2 3 4"}   # thread counts of 10_reference.sbatch (the QOS allows 4 cores)
REPORT_THREADS=${REPORT_THREADS:-4}     # thread count of 60_report.sbatch
STREAM_N=${STREAM_N:-80000000}  # STREAM array length, fixed when compiled: 3 arrays x 8 B x 80M = 1.9 GB
                                # (fits --mem=8G and is 54x the 35.75 MB L3 of a socket)
READBW_GIB=${READBW_GIB:-2}     # read-probe array size
