#!/bin/bash
#SBATCH --job-name=p1-report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:04:30
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Everything the Rudra report needs, in ONE 4-core job (the QOS allows 4 CPUs for 5 minutes). Every measurement
# runs 3 times on 4 threads, one per core on socket 0, with memory bound to socket 0 and a Q8_0 KV cache:
#   0. records: the node (lscpu, caches, SIMD, memory, NUMA), the software (manifest.txt), the Slurm limits
#      recorded earlier on the login node (./50_limits_login.sh)
#   1. memory bandwidth: STREAM and the read-only probe, at 1 and 4 threads
#   2. the shared reference run: llama-bench, 1B Q4_K_M, pp512 + tg128, 3 repeats
#   3. llama.cpp's own timings (llama_perf): llama-completion on a fixed 512-token prompt, 128 tokens decoded
#   4. profile: perf record of prefill and of decode (skipped, with the reason written down, if perf is missing)
#   5. report_rudra.py: tables (mean ± std, min-max) and the roofline check from this job's own bandwidth
# Before: ./00_setup_login.sh tools (STREAM and readbw binaries) and ./50_limits_login.sh, both on the login node.
# Submit: ./submit.sh report        About 75 s without perf.        Results: results/rudra_<jobid>/REPORT.md
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/lib.sh"
refuse_login_node
load_software
JOB_T0=$(date +%s)
job_banner
check_build
OUT=$RESULTS_ROOT/rudra_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"/{bw,reference,timings,perf}
T=$REPORT_THREADS; R=$REPEATS
PROMPT=$HERE/prompt_512.txt    # first 2,140 characters of wikitext-2 test = exactly 512 tokens with BOS (Llama 3 tokenizer)
step() { log "$*"; echo "[$(date -Is)] $*" >> "$OUT/steps.txt"; }

# ---- 0. records -------------------------------------------------------------------------------------------
cp config.sh "$OUT/config_used.sh"
./node_info.sh "$OUT/node"
write_manifest "$OUT"
latest_limits=$(ls -d "$RESULTS_ROOT"/limits_* 2>/dev/null | tail -1 || true)
[[ -n $latest_limits ]] && cp -r "$latest_limits" "$OUT/limits"

# ---- 1. memory bandwidth ----------------------------------------------------------------------------------
# STREAM (standard benchmark, reads + writes) and readbw.c (reads only, like decode). Each STREAM run reports the
# best of its 10 iterations (the STREAM rule); readbw reports the mean of its passes 2-10.
step "1. STREAM + read-only bandwidth at 1 and $T threads, $R runs each"
for b in stream readbw; do
  [[ -x $TOOLS_BIN/$b ]] || { echo "$TOOLS_BIN/$b missing: run ./00_setup_login.sh tools on the login node" >&2; exit 1; }
  cp "$TOOLS_BIN/$b" "$OUT/bw/"
done
cp "$TOOLS_BIN/build_info.txt" "$OUT/bw/tools_build_info.txt" 2>/dev/null || true
echo "threads,run,kernel,best_MBps" > "$OUT/bw/stream.csv"
echo "threads,run,read_GBps_best,read_GBps_mean" > "$OUT/bw/readbw.csv"
for t in 1 "$T"; do
  place "$t"
  for r in $(seq 1 "$R"); do
    env "${OMP_ENV[@]}" OMP_NUM_THREADS="$t" "${PLACE[@]}" "$OUT/bw/stream" > "$OUT/bw/stream_T${t}_r$r.txt"
    awk -v t="$t" -v r="$r" '/^(Copy|Scale|Add|Triad):/{printf "%d,%d,%s,%s\n", t, r, substr($1,1,length($1)-1), $2}' \
        "$OUT/bw/stream_T${t}_r$r.txt" >> "$OUT/bw/stream.csv"
    line=$(env "${OMP_ENV[@]}" OMP_NUM_THREADS="$t" "${PLACE[@]}" "$OUT/bw/readbw" "$READBW_GIB")
    echo "$t,$r,$(sed -E 's/.*best=([0-9.]+).*mean=([0-9.]+).*/\1,\2/' <<< "$line")" >> "$OUT/bw/readbw.csv"
  done
done
echo "placement=$PLACE_TAG cpus=$(cpus_first "$T") stream_n=$(grep -o 'stream_n=[0-9]*' "$TOOLS_BIN/build_info.txt" | cut -d= -f2) ntimes=10 readbw_gib=$READBW_GIB" \
  > "$OUT/bw/settings.txt"

# ---- 2. the shared reference run -------------------------------------------------------------------------
place "$T"
step "2. reference run: llama-bench 1B Q4_K_M -p 512 -n 128 -r $R, $T threads ($PLACE_TAG)"
logged "$OUT" reference timev "$OUT/reference/T${T}.time" env "${OMP_ENV[@]}" "${PLACE[@]}" \
  "$BENCH" -m "$M1Q4" -t "$T" -p 512 -n 128 -r "$R" "${KV_ARGS[@]}" -o csv -oe jsonl \
  > "$OUT/reference/T${T}.csv" 2> "$OUT/reference/T${T}.stderr.log" || log "reference run failed: see reference/T${T}.stderr.log"
grep '^{' "$OUT/reference/T${T}.stderr.log" > "$OUT/reference/T${T}.jsonl" || true

# ---- 3. llama.cpp's own timings ---------------------------------------------------------------------------
# llama-completion prints llama_perf lines at exit: load time, prompt eval time (prefill), eval time (decode).
# -c 1024: a context just big enough (the default would allocate the model's full 128k KV cache).
# --ignore-eos --temp 0 --seed 1: always exactly 128 decoded tokens, the same text every run.
step "3. llama-completion timings: 512-token prompt, 128 decoded tokens, $R runs"
for r in $(seq 1 "$R"); do
  logged "$OUT" "completion_r$r" timev "$OUT/timings/run$r.time" env "${OMP_ENV[@]}" "${PLACE[@]}" \
    "$COMPLETION" -m "$M1Q4" -f "$PROMPT" -c 1024 -n 128 --ignore-eos --temp 0 --seed 1 -t "$T" -ctk "$KV" -ctv "$KV" -fa on \
    -no-cnv --no-display-prompt > "$OUT/timings/run$r.out" 2> "$OUT/timings/run$r.log" < /dev/null || log "completion run $r failed"
  { grep -E 'common_perf_print|llama_perf' "$OUT/timings/run$r.log" || true; } | sed -E 's/^[0-9.]+ I //' > "$OUT/timings/run$r.perf.txt"
done
load_ms=$({ grep -ho 'load time = *[0-9.]*' "$OUT"/timings/run*.perf.txt || true; } | awk '{print $NF}' | sort -n | tail -1)
load_ms=${load_ms:-3000}

# ---- 4. profile (perf) ------------------------------------------------------------------------------------
# User-space cycles, 1999 samples/s. -D delays sampling past model load + weight repacking (load time + 0.5 s),
# so the profile shows only prefill (pp512) or only decode (tg128). Symbols are grouped into operators later.
PERF=${PERF_BIN:-$(command -v perf || true)}
par=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 9)
if [[ -z $PERF ]]; then
  echo "perf not found on $(hostname) (PERF_BIN unset)" > "$OUT/perf/NOT_RUN.txt"
elif (( par > 2 )) && [[ -z ${PERF_PARANOID_OK:-} ]]; then
  echo "perf_event_paranoid=$par on $(hostname): profiling own processes not allowed" > "$OUT/perf/NOT_RUN.txt"
else
  delay=$(( ${load_ms%.*} + 500 ))
  step "4. perf record -e cycles:u -F 1999 -D ${delay} ms, prefill and decode, $R runs each"
  for r in $(seq 1 "$R"); do
    for ph in pp tg; do
      if [[ $ph == pp ]]; then args=(-p 512 -n 0); else args=(-p 0 -n 128); fi
      env "${OMP_ENV[@]}" "${PLACE[@]}" "$PERF" record -e cycles:u -F 1999 -D "$delay" -o "$OUT/perf/${ph}_r$r.data" -- \
        "$BENCH" -m "$M1Q4" -t "$T" "${KV_ARGS[@]}" -r 1 --no-warmup "${args[@]}" > "$OUT/perf/${ph}_r$r.bench.txt" 2> "$OUT/perf/${ph}_r$r.log" \
        || echo "perf record failed (exit $?)" >> "$OUT/perf/${ph}_r$r.log"
      { "$PERF" report -i "$OUT/perf/${ph}_r$r.data" --no-children --sort symbol --stdio 2>/dev/null || true; } \
        | { grep -v -e '^#' -e '^$' || true; } | head -60 > "$OUT/perf/${ph}_r$r.report.txt" || true
      rm -f "$OUT/perf/${ph}_r$r.data"
    done
  done
fi

# ---- 5. report ---------------------------------------------------------------------------------------------
{ echo "date_end: $(date -Is)"; echo "job_runtime_s: $(( $(date +%s) - JOB_T0 ))"; echo "threads: $T"; echo "placement: $PLACE_TAG"; } >> "$OUT/manifest.txt"
"$PY" "$HERE/report_rudra.py" "$OUT" || true
copy_job_logs "$OUT"
log "report job done in $(( $(date +%s) - JOB_T0 )) s: $OUT/REPORT.md"
