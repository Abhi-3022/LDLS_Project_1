#!/usr/bin/env bash
# Memory bandwidth vs thread count: STREAM (copy/scale/add/triad) + a read-only
# reduction (tools/readbw.c). Same placement as the model runs (socket $NODE).
# Output: results/csv/stream.csv  (GB/s = 1e9 bytes/s, STREAM byte-counting convention)
set -euo pipefail
source "$(dirname "$0")/env.sh"
BIN="$PROJ/third_party/stream"
mkdir -p "$RESULTS/csv" "$RESULTS/raw/stream"

# 100M doubles per array = 800 MB each, 2.4 GB total: >> 105 MiB L3 per socket
gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=20 \
    "$BIN/stream.c" -o "$BIN/stream"
gcc -O3 -march=native -fopenmp "$PROJ/tools/readbw.c" -o "$PROJ/tools/readbw"

CSV="$RESULTS/csv/stream.csv"
echo "threads,rep,copy_GBps,scale_GBps,add_GBps,triad_GBps,read_GBps,loadavg" > "$CSV"
for t in $THREAD_LIST; do
    for r in $(seq 1 "$REPS"); do
        log="$RESULTS/raw/stream/stream_t${t}_r${r}.txt"
        pinned "$t" "$BIN/stream" > "$log"
        rd=$(pinned "$t" "$PROJ/tools/readbw" 2 20 | sed -E 's/.*read_GBps=([0-9.]+).*/\1/')
        get() { awk -v k="$1:" '$1==k {printf "%.2f", $2/1000}' "$log"; }
        la=$(cut -d' ' -f1 /proc/loadavg)
        echo "$t,$r,$(get Copy),$(get Scale),$(get Add),$(get Triad),$rd,$la" | tee -a "$CSV"
    done
done
echo "wrote $CSV"
