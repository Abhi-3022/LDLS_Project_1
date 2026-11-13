#!/usr/bin/env bash
# Shared reference test (same for every team): Llama-3.2-1B Q4_K_M, 512-token prompt,
# then 128 tokens decoded on top of it. 3 repetitions, peak RSS per run.
# Runs anywhere: on this cluster through bench_sweep.py's placement, or stand-alone on the
# PARAM Rudra CPU node via scripts/rudra_reference.sbatch.
#
# usage: reference_run.sh [llama-bench binary] [model.gguf] [out.csv] [threads ...]
#   defaults: our build, models/Llama-3.2-1B-Instruct-Q4_K_M.gguf, results/csv/reference_<host>.csv,
#             threads = all physical cores of socket 0 (and half of them)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
BENCH="${1:-$PROJ/third_party/llama.cpp/build-cpu/bin/llama-bench}"
MODEL="${2:-$PROJ/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
HOST="$(hostname -s)"
OUT="${3:-$PROJ/results/csv/reference_${HOST}.csv}"
shift $(( $# > 3 ? 3 : $# )) || true

# physical cores of the first socket (one SMT sibling each), unless $CORES (space-separated, as set by
# scripts/env.sh) names them: on this cluster core 1 is permanently busy, and a single busy core in
# the pinned set slows every barrier-synchronised op (measured: ~10x, results/csv/busycore_*.csv)
if [ -n "${CORES:-}" ]; then
    SOCK0=$(echo $CORES | tr ' ' ',')
else
    SOCK0=$(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '$3==0 && !seen[$2]++ {print $1}' | paste -sd, -)
fi
NCORE=$(echo "$SOCK0" | tr ',' '\n' | wc -l)
THREADS=("$@")
[ ${#THREADS[@]} -eq 0 ] && THREADS=("$NCORE" "$(( NCORE / 2 ))")

mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || echo "host,cpu,test,n_prompt,n_gen,n_depth,threads,rep,tok_s,maxrss_gb,build_commit" > "$OUT"
CPU=$(lscpu | sed -n 's/^Model name:\s*//p' | tr ',' ' ')
for t in "${THREADS[@]}"; do
    PIN=$(echo "$SOCK0" | tr ',' '\n' | head -n "$t" | paste -sd, -)
    for rep in 1 2 3; do
        for spec in "pp 512 0 0" "tg 0 128 512"; do
            read -r test p n d <<< "$spec"
            tf=$(mktemp)
            json=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores /usr/bin/time -f "%M" -o "$tf" \
                numactl --physcpubind="$PIN" --localalloc "$BENCH" -m "$MODEL" -p "$p" -n "$n" -d "$d" \
                -t "$t" -r 1 -lm none -o json 2>/dev/null)
            ts=$(echo "$json" | grep -m1 '"avg_ts"' | grep -oE '[0-9.]+')
            commit=$(echo "$json" | grep -m1 '"build_commit"' | cut -d'"' -f4)
            rss=$(awk '{printf "%.3f", $1*1024/1e9}' "$tf"); rm -f "$tf"
            echo "$HOST,$CPU,$test,$p,$n,$d,$t,$rep,$ts,$rss,$commit" | tee -a "$OUT"
        done
    done
done
