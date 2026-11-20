"""Step 4: predict decode speed from memory bandwidth, before measuring.

  predicted tok/s = bandwidth / (weight bytes + context x KV-cache bytes per token)

Everything except the bandwidth is read from the GGUF files:
  weight bytes            = sum of all tensor data (what llama-bench reports as "size")
  KV bytes per token      = layers x KV heads x head dim x 2 (K and V) x 2 bytes (F16 cache)

Usage:
  pip install gguf
  python scripts\\step4_predict_decode.py --bandwidth 52103 --models-dir $HOME\\models --out results\\prediction.csv
"""
import argparse
import csv
from pathlib import Path

from gguf import GGUFReader

MODELS = [  # file, contexts to predict (the project's context limits)
    ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", [1024, 8192, 16384]),
    ("Llama-3.2-1B-Instruct-Q8_0.gguf", [1024, 8192, 16384]),
    ("Llama-3.2-3B-Instruct-Q4_K_M.gguf", [1024, 8192]),
]
KV_BYTES_PER_VALUE = 2  # llama.cpp default KV cache type is F16


def field(reader, key):
    f = reader.fields[key]
    return int(f.parts[f.data[0]][0])


def model_shape(path):
    r = GGUFReader(path)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    layers = field(r, f"{arch}.block_count")
    n_head = field(r, f"{arch}.attention.head_count")
    n_kv = field(r, f"{arch}.attention.head_count_kv")
    key = f"{arch}.attention.key_length"
    head_dim = field(r, key) if key in r.fields else field(r, f"{arch}.embedding_length") // n_head
    weight_bytes = sum(int(t.n_bytes) for t in r.tensors)
    return layers, n_kv, head_dim, weight_bytes, path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth", type=float, required=True, help="STREAM Triad in MB/s (1 MB = 1e6 bytes)")
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--out", default="prediction.csv")
    a = ap.parse_args()
    bw = a.bandwidth * 1e6

    rows = []
    print(f"Bandwidth (STREAM Triad): {a.bandwidth:,.0f} MB/s = {bw / 1e9:.1f} GB/s\n")
    for name, contexts in MODELS:
        path = a.models_dir / name
        layers, n_kv, head_dim, wb, file_bytes = model_shape(path)
        kv = layers * n_kv * head_dim * 2 * KV_BYTES_PER_VALUE
        print(name)
        print(f"  weight bytes = {wb:,} B = {wb / 2**20:.2f} MiB = {wb / 1e6:.1f} MB"
              f"   (whole file {file_bytes / 1e6:.1f} MB incl. tokenizer and metadata)")
        print(f"  KV per token = {layers} layers x {n_kv} KV heads x {head_dim} dims x 2 (K,V) x "
              f"{KV_BYTES_PER_VALUE} B = {kv:,} B")
        for ctx in contexts:
            total = wb + ctx * kv
            pred = bw / total
            pred_file = bw / (file_bytes + ctx * kv)
            print(f"  ctx {ctx:>6}: {wb / 1e6:.1f} MB + {ctx} x {kv:,} B = {total / 1e6:.1f} MB/token"
                  f"  ->  {a.bandwidth:,.0f} / {total / 1e6:.1f} = {pred:.1f} tok/s"
                  f"   (whole file: {pred_file:.1f})")
            rows.append({"model": name, "context": ctx, "weight_bytes": wb, "kv_bytes_per_token": kv,
                         "bytes_per_token": total, "predicted_tok_s": round(pred, 2),
                         "predicted_tok_s_whole_file": round(pred_file, 2)})
        print()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
