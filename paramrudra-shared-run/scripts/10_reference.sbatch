#!/bin/bash
#SBATCH --job-name=p1-ref
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:04:30
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# THE SHARED REFERENCE RUN, with thread scaling over the 4 cores the QOS allows.
#   Test (fixed by the course): Llama-3.2-1B-Instruct Q4_K_M, llama-bench pp512 (512-token prompt) + tg128
#   (128 decoded tokens), 3 repeats, CPU only, KV cache Q8_0 + flash attention (No F16).
#   Thread counts: REF_THREADS (default 1 2 3 4), each pinned one thread per core on socket 0, memory on socket 0.
#   Also records the node (node_info.sh) and the software (manifest.txt). About 2.5 min.
# Submit: ./submit.sh reference          Results: results/ref_<jobid>/REPORT.md
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/lib.sh"
refuse_login_node
load_software
JOB_T0=$(date +%s)
job_banner
check_build
OUT=$RESULTS_ROOT/ref_${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT/reference"
cp config.sh "$OUT/config_used.sh"
./node_info.sh "$OUT/node"
write_manifest "$OUT"
echo "ref_threads: $REF_THREADS" >> "$OUT/manifest.txt"

# largest thread count first: if the job runs out of time, the headline 4-thread number is already saved
for T in $(printf '%s\n' $REF_THREADS | sort -rn); do
  place "$T"
  tag=T${T}_$PLACE_TAG
  log "reference run: $T threads ($PLACE_TAG, CPUs $(cpus_first "$T"))"
  # -o csv: mean and std per test (stdout);  -oe jsonl: the same run with every repeat's tok/s (stderr)
  logged "$OUT" "$tag" timev "$OUT/reference/$tag.time" env "${OMP_ENV[@]}" "${PLACE[@]}" \
    "$BENCH" -m "$M1Q4" -t "$T" -p 512 -n 128 -r "$REPEATS" "${KV_ARGS[@]}" -o csv -oe jsonl \
    > "$OUT/reference/$tag.csv" 2> "$OUT/reference/$tag.stderr.log" || log "run $tag failed: see reference/$tag.stderr.log"
  grep '^{' "$OUT/reference/$tag.stderr.log" > "$OUT/reference/$tag.jsonl" || true
  "$PY" "$HERE/report_reference.py" "$OUT" > /dev/null 2>&1 || true   # rewrite the report after every run
done

{ echo "date_end: $(date -Is)"; echo "job_runtime_s: $(( $(date +%s) - JOB_T0 ))"; } >> "$OUT/manifest.txt"
"$PY" "$HERE/report_reference.py" "$OUT" || true      # reference_summary.csv + REPORT.md
copy_job_logs "$OUT"
log "reference job done in $(( $(date +%s) - JOB_T0 )) s: $OUT/REPORT.md"
