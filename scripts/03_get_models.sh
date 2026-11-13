#!/usr/bin/env bash
# Download F16 GGUFs and quantize them locally with the pinned llama.cpp build
# (plain llama-quantize, no imatrix) so every model uses the same quantizer.
#   Q4_K_M for every model; Q8_0 only for models <= 1.5B parameters.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$MODELS/f16" "$RESULTS/logs"

"$PY" - <<'EOF'
from huggingface_hub import hf_hub_download
import os
files = [
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-f16.gguf"),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-f16.gguf"),
    ("bartowski/granite-3.1-1b-a400m-instruct-GGUF", "granite-3.1-1b-a400m-instruct-f16.gguf"),
    ("bartowski/granite-3.1-3b-a800m-instruct-GGUF", "granite-3.1-3b-a800m-instruct-f16.gguf"),
]
for repo, f in files:
    print(hf_hub_download(repo, f, local_dir=os.path.join(os.environ["MODELS"], "f16")))
EOF

# bonus: released Mamba-3 1.5B checkpoints (layer 0 is used for a real-weight kernel check).
# plain resumable curl: huggingface_hub's Xet path stalled on these files on this cluster
for r in mamba3-siso-1.5b mamba3-mimo-1.5b; do
    mkdir -p "$MODELS/mamba3/$r"
    for f in config.json pytorch_model.bin; do
        curl -sL --retry 5 -C - -o "$MODELS/mamba3/$r/$f" "https://huggingface.co/state-spaces/$r/resolve/main/$f"
    done
done

# quantization is a one-off: run it on the other socket so it never overlaps measurements
QNODE=$(( 1 - NODE ))
quant() {  # f16-file out-file type
    [ -f "$MODELS/$2" ] && { echo "exists: $2"; return; }
    numactl --cpunodebind=$QNODE --membind=$QNODE \
        "$LLAMA_BIN/llama-quantize" "$MODELS/f16/$1" "$MODELS/$2" "$3" 32 \
        > "$RESULTS/logs/quantize_${2%.gguf}.log" 2>&1
    echo "quantized: $2 ($(du -h "$MODELS/$2" | cut -f1))"
}
quant Llama-3.2-1B-Instruct-f16.gguf          Llama-3.2-1B-Instruct-Q4_K_M.gguf          Q4_K_M
quant Llama-3.2-1B-Instruct-f16.gguf          Llama-3.2-1B-Instruct-Q8_0.gguf            Q8_0
quant Llama-3.2-3B-Instruct-f16.gguf          Llama-3.2-3B-Instruct-Q4_K_M.gguf          Q4_K_M
quant granite-3.1-1b-a400m-instruct-f16.gguf  granite-3.1-1b-a400m-instruct-Q4_K_M.gguf  Q4_K_M
quant granite-3.1-1b-a400m-instruct-f16.gguf  granite-3.1-1b-a400m-instruct-Q8_0.gguf    Q8_0
quant granite-3.1-3b-a800m-instruct-f16.gguf  granite-3.1-3b-a800m-instruct-Q4_K_M.gguf  Q4_K_M
sha256sum "$MODELS"/*.gguf > "$MODELS/SHA256SUMS"
