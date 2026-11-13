#!/usr/bin/env bash
# Correctness: perplexity on the fixed WikiText-2 sample + a few greedy chat outputs per model.
#   results/csv/perplexity.csv, results/raw/samples/<model>.txt
# F16 rows are a reference for the quantization error only (never benchmarked; the 3B F16
# files exceed the 4.5 GB budget, which is why the benchmarks never use them).
set -euo pipefail
source "$(dirname "$0")/env.sh"
TEXT="$PROJ/data/wikitext2_test_100k.txt"
[ -f "$TEXT" ] || "$PY" "$PROJ/scripts/get_data.py"
mkdir -p "$RESULTS/raw/perplexity" "$RESULTS/raw/samples"
CSV="$RESULTS/csv/perplexity.csv"
echo "model,file,ppl,ppl_err,n_chunks,ctx,threads,maxrss_gb,role" > "$CSV"

ppl() {  # name file role
    local log="$RESULTS/raw/perplexity/$1.log"
    /usr/bin/time -f "%M" -o "$log.time" $(pin_prefix "$T_MAIN") \
        "$LLAMA_BIN/llama-perplexity" -m "$2" -f "$TEXT" -c 512 -b 512 -t "$T_MAIN" -lm none > "$log" 2>&1
    local p e n
    p=$(grep -oE "Final estimate: PPL = [0-9.]+" "$log" | awk '{print $NF}')
    e=$(grep -oE "\+/- [0-9.]+" "$log" | tail -1 | awk '{print $2}')
    n=$(grep -oE "calculating perplexity over [0-9]+ chunks" "$log" | grep -oE "[0-9]+")
    echo "$1,$(basename "$2"),$p,$e,$n,512,$T_MAIN,$(awk '{printf "%.3f", $1*1024/1e9}' "$log.time"),$3" | tee -a "$CSV"
}

while IFS='|' read -r name file ctx; do
    [ -z "$name" ] && continue
    ppl "$name" "$MODELS/$file" benchmarked
done <<< "$MODEL_TABLE"
for f in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct granite-3.1-1b-a400m-instruct granite-3.1-3b-a800m-instruct; do
    ppl "$f-f16" "$MODELS/f16/$f-f16.gguf" reference_only
done

# greedy chat samples (single turn, chat template from the GGUF)
PROMPTS=(
    "What is the capital of France? Answer in one sentence."
    "Write a Python function that returns the n-th Fibonacci number."
    "Explain in two sentences why the sky is blue."
)
while IFS='|' read -r name file ctx; do
    [ -z "$name" ] && continue
    out="$RESULTS/raw/samples/$name.txt"
    : > "$out"
    for p in "${PROMPTS[@]}"; do
        echo "### PROMPT: $p" >> "$out"
        $(pin_prefix "$T_MAIN") "$LLAMA_BIN/llama-completion" -m "$MODELS/$file" -cnv -st -p "$p" -n 96 \
            --temp 0 -s 42 -t "$T_MAIN" --no-display-prompt < /dev/null 2>/dev/null >> "$out" || true
        echo -e "\n" >> "$out"
    done
    echo "samples: $out"
done <<< "$MODEL_TABLE"
