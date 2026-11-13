#!/usr/bin/env python3
"""Predict decode speed before measuring: bandwidth / bytes read per decoded token.

Bytes per decoded token (batch 1), from the GGUF tensor table:
  * every non-expert weight once (attention, norms, router, dense FFN)
  * MoE expert tensors (*_exps): only the routed experts, i.e. used/count of their bytes
    ("active"); the "total" variant counts all experts, for the active-vs-total question
  * LM head: output.weight, or token_embd when embeddings are tied (both families here)
  * token embedding lookup: one row
  * KV cache: 2 (K,V) x n_layer x n_head_kv x head_dim x 2 bytes (F16) per token of context,
    all of it read once per decoded token

Bandwidth: measured on the same socket and core set (results/csv/stream.csv); the read-only
reduction is the primary number (decode is read-dominated), STREAM triad/copy are reported too.

Outputs: results/csv/model_bytes.csv, results/csv/roofline_pred.csv
"""
import csv
import os
import pathlib

import numpy as np
from gguf import GGUFReader

PROJ = pathlib.Path(__file__).resolve().parents[1]
MODELS = pathlib.Path(os.environ.get("MODELS", PROJ / "models"))
RESULTS = pathlib.Path(os.environ.get("RESULTS", PROJ / "results"))


def model_table():
    rows = []
    for line in os.environ.get("MODEL_TABLE", "").strip().splitlines():
        name, f, ctx = line.strip().split("|")
        rows.append(dict(name=name, file=f, max_ctx=int(ctx)))
    return rows


def _field(r, key, default=None):
    for k, f in r.fields.items():
        if k.endswith(key):
            return f.parts[f.data[0]].tolist()[0]
    return default


def model_bytes(path):
    r = GGUFReader(str(path))
    n_layer = _field(r, ".block_count")
    n_embd = _field(r, ".embedding_length")
    n_head = _field(r, ".attention.head_count")
    n_head_kv = _field(r, ".attention.head_count_kv", n_head)
    head_dim = _field(r, ".attention.key_length", n_embd // n_head)
    n_exp = _field(r, ".expert_count", 0) or 0
    n_used = _field(r, ".expert_used_count", 0) or 0

    names = {t.name for t in r.tensors}
    tied = "output.weight" not in names
    b = dict(experts=0, other=0, embd=0, head=0)
    p = dict(experts=0, other=0, embd=0, head=0)
    vocab = 0
    for t in r.tensors:
        n_el = int(np.prod([int(x) for x in t.shape]))
        if t.name == "token_embd.weight":
            b["embd"], p["embd"], vocab = int(t.n_bytes), n_el, int(t.shape[-1])
        elif t.name == "output.weight":
            b["head"], p["head"] = int(t.n_bytes), n_el
        elif "_exps" in t.name:
            b["experts"] += int(t.n_bytes); p["experts"] += n_el
        else:
            b["other"] += int(t.n_bytes); p["other"] += n_el
    frac = n_used / n_exp if n_exp else 1.0
    head_bytes = b["embd"] if tied else b["head"]
    embd_row = b["embd"] / vocab
    d = dict(
        n_layer=n_layer, n_embd=n_embd, n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim,
        n_expert=n_exp, n_expert_used=n_used, tied_embeddings=int(tied),
        file_bytes=sum(int(t.n_bytes) for t in r.tensors),
        params_total=sum(p.values()),
        params_active=p["other"] + p["experts"] * frac + (p["embd"] if tied else p["head"]),
        bytes_lm_head=head_bytes, bytes_experts_total=b["experts"], bytes_other=b["other"],
        decode_bytes_active=b["other"] + b["experts"] * frac + head_bytes + embd_row,
        decode_bytes_total=b["other"] + b["experts"] + head_bytes + embd_row,
        kv_bytes_per_token=2 * n_layer * n_head_kv * head_dim * 2,
    )
    return d


def bandwidth():
    """mean GB/s per thread count and source from stream.csv"""
    rows = list(csv.DictReader(open(RESULTS / "csv" / "stream.csv")))
    out = {}
    for src in ("read_GBps", "triad_GBps", "copy_GBps"):
        for t in sorted({int(r["threads"]) for r in rows}):
            v = [float(r[src]) for r in rows if int(r["threads"]) == t]
            out[(src.split("_")[0], t)] = sum(v) / len(v)
    return out


def predict(mb, bw_gbps, context, active=True):
    w = mb["decode_bytes_active"] if active else mb["decode_bytes_total"]
    return bw_gbps * 1e9 / (w + context * mb["kv_bytes_per_token"])


def main():
    models = model_table()
    bw = bandwidth()
    out_b = RESULTS / "csv" / "model_bytes.csv"
    out_p = RESULTS / "csv" / "roofline_pred.csv"
    mbs = {}
    with open(out_b, "w", newline="") as f:
        w = None
        for m in models:
            mb = model_bytes(MODELS / m["file"])
            mbs[m["name"]] = (m, mb)
            row = dict(model=m["name"], file=m["file"], max_ctx=m["max_ctx"], **mb)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader()
            w.writerow(row)
            print(f"{m['name']:13s} active {mb['decode_bytes_active']/1e9:.3f} GB/tok  total {mb['decode_bytes_total']/1e9:.3f} GB/tok"
                  f"  KV {mb['kv_bytes_per_token']/1024:.0f} KiB/ctx-token  params {mb['params_total']/1e9:.2f}B"
                  f" (active {mb['params_active']/1e9:.2f}B)")
    threads = sorted({t for (_, t) in bw})
    with open(out_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "threads", "context", "bw_source", "bw_GBps", "bytes_per_token", "pred_tok_s",
                    "pred_tok_s_if_all_experts"])
        for name, (m, mb) in mbs.items():
            # decode tg128 at depth D sees an average context of D + 64; contexts from the sweeps
            ctxs = sorted({64, 1024 - 64, 8192 - 64, m["max_ctx"] - 64} | set(range(0, m["max_ctx"] + 1, 512)))
            for t in threads:
                for src in ("read", "triad", "copy"):
                    for c in ctxs:
                        b = bw[(src, t)]
                        w.writerow([name, t, c, src, round(b, 2),
                                    int(mb["decode_bytes_active"] + c * mb["kv_bytes_per_token"]),
                                    round(predict(mb, b, c), 2), round(predict(mb, b, c, active=False), 2)])
    print(f"wrote {out_b} and {out_p}")


if __name__ == "__main__":
    main()
