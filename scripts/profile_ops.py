#!/usr/bin/env python3
"""Per-operator time breakdown for prefill and decode.

`perf` is not usable on this cluster (kernel.perf_event_paranoid = 3), so profiling uses the
GGML_OP_PROFILE build of llama.cpp (patches/ggml_op_profile.patch). For every graph node,
thread 0 records:
  total = node start -> end of the barrier after it (= slowest thread + barrier wait)
  own   = node start -> thread 0 finished its own share
total - own is the synchronisation / load-imbalance cost of that op.
Graphs with n_tokens == 1 are decode graphs, the others prefill.

Runs (T_MAIN threads unless stated, 3 reps, warm-up disabled so no stray graphs):
  prefill: 512 and 4096 prompt tokens
  decode : 64 tokens at context 1k / 4k / 8k / 16k (capped by the model's max)
  decode at 1k with 1 thread too (llama1b-q4, granite1b-q4): no barriers -> pure compute
Outputs:
  results/raw/opprof/<run>.csv          raw per-op table from the profiler
  results/csv/opprof_ops.csv            per op/name, per run: ms per token, share
  results/csv/opprof_groups.csv         grouped categories (attention, FFN, MoE experts, routing, ...)

usage: profile_ops.py [--models ...] [--reps 3] [--aggregate-only]
"""
import argparse
import csv
import os
import pathlib
import subprocess

import pandas as pd

PROJ = pathlib.Path(__file__).resolve().parents[1]
ENV = os.environ
PROF_BIN = pathlib.Path(ENV.get("LLAMA_PROF_BIN", PROJ / "third_party/llama.cpp/build-prof/bin"))
MODELS = pathlib.Path(ENV.get("MODELS", PROJ / "models"))
RESULTS = pathlib.Path(ENV.get("RESULTS", PROJ / "results"))
RAW = RESULTS / "raw" / "opprof"
CORES = ENV.get("CORES", "0 " + " ".join(map(str, range(2, 56)))).split()
NODE = ENV.get("NODE", "0")
T_MAIN = int(ENV.get("T_MAIN", 32))

ROUTING = {"ffn_moe_logits", "ffn_moe_probs", "ffn_moe_argsort", "ffn_moe_topk", "ffn_moe_weights",
           "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped", "ffn_moe_weights_norm",
           "ffn_moe_weights_scaled", "ffn_moe_weighted", "ffn_moe_out", "ffn_moe_probs_biased"}


def model_table():
    rows = {}
    for line in ENV.get("MODEL_TABLE", "").strip().splitlines():
        name, f, ctx = line.strip().split("|")
        rows[name] = dict(name=name, file=f, max_ctx=int(ctx))
    return rows


def group(op, name, count, n_graphs, is_moe):
    """map a (op, node name) to a category for the breakdown plot"""
    if op == "FLASH_ATTN_EXT" or name in ("kq", "kq_soft_max", "kqv", "kqv_out"):
        return "attention (QK/softmax/V over KV cache)"
    if op == "SET_ROWS" and name.startswith("cache_"):
        return "KV cache write"
    if op == "MUL_MAT" and name in ("Qcur", "Kcur", "Vcur", "attn_out", "wo", "attn_output"):
        return "attention projections (QKV, O)"
    if op == "MUL_MAT_ID":
        return "MoE expert matmuls"
    if name in ROUTING or (is_moe and op == "ADD" and name == "node"):
        return "MoE routing + combine"
    if op == "MUL_MAT" and name.startswith("ffn"):
        return "dense FFN matmuls"
    if op == "MUL_MAT" and (name == "result_output" or count == n_graphs):
        return "LM head (tied embedding)"
    if op.startswith("RMS_NORM") or op in ("NORM", "MUL") and "norm" in name:
        return "norms"
    return "rope / elementwise / other"


def run(model, label, p, n, d, t, rep):
    out = RAW / f"{model['name']}__{label}__t{t}__r{rep}.csv"
    if out.exists():
        return out
    RAW.mkdir(parents=True, exist_ok=True)
    env = dict(ENV, GGML_OP_PROFILE=str(out), OMP_NUM_THREADS=str(t), OMP_PROC_BIND="close", OMP_PLACES="cores")
    cmd = ["numactl", f"--physcpubind={','.join(CORES[:t])}", f"--membind={NODE}",
           str(PROF_BIN / "llama-bench"), "-m", str(MODELS / model["file"]), "-p", str(p), "-n", str(n),
           "-d", str(d), "-t", str(t), "-r", "1", "-lm", "none", "--no-warmup", "-o", "json"]
    subprocess.run(cmd, check=True, capture_output=True, env=env)
    return out


def plan(models):
    for m in models:
        for p in (512, 4096):
            yield m, f"prefill_p{p}", p, 0, 0, T_MAIN, "prefill", p
        for c in sorted({1024, 4096, 8192, m["max_ctx"]}):
            if c <= m["max_ctx"]:
                yield m, f"decode_c{c}", 0, 64, c - 64, T_MAIN, "decode", c
        if m["name"] in ("llama1b-q4", "granite1b-q4"):
            yield m, "decode_c1024", 0, 64, 1024 - 64, 1, "decode", 1024


def aggregate(models, reps):
    ops, groups = [], []
    for (m, label, p, n, d, t, phase, ctx) in plan(models):
        for rep in range(1, reps + 1):
            f = RAW / f"{m['name']}__{label}__t{t}__r{rep}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            g = df[(df.op == "__GRAPH__") & (df.phase == phase)].iloc[0]
            info = dict(kv.split("=") for kv in g["name"].split(";"))
            n_graphs, n_tok, graph_ms = int(info["graphs"]), int(info["tokens"]), float(g["total_ms"])
            df = df[(df.op != "__GRAPH__") & (df.phase == phase)].copy()
            is_moe = (df.op == "MUL_MAT_ID").any()
            df["group"] = [group(o, nm, c, n_graphs, is_moe) for o, nm, c in zip(df.op, df.name, df["count"])]
            df["ms_per_token"] = df.total_ms / n_tok
            df["own_ms_per_token"] = df.own_ms / n_tok
            df["share"] = df.total_ms / graph_ms
            base = dict(model=m["name"], run=label, phase=phase, context=ctx, threads=t, rep=rep,
                        tokens=n_tok, graph_ms_per_token=graph_ms / n_tok)
            for r in df.itertuples():
                ops.append(dict(base, op=r.op, name=r.name, group=r.group, count=r.count,
                                ms_per_token=r.ms_per_token, own_ms_per_token=r.own_ms_per_token, share=r.share))
            gg = df.groupby("group")[["ms_per_token", "own_ms_per_token", "share"]].sum().reset_index()
            for r in gg.itertuples():
                groups.append(dict(base, group=r.group, ms_per_token=r.ms_per_token,
                                   own_ms_per_token=r.own_ms_per_token, share=r.share))
    pd.DataFrame(ops).to_csv(RESULTS / "csv" / "opprof_ops.csv", index=False)
    pd.DataFrame(groups).to_csv(RESULTS / "csv" / "opprof_groups.csv", index=False)
    print(f"wrote {RESULTS/'csv'/'opprof_ops.csv'} ({len(ops)} rows) and opprof_groups.csv ({len(groups)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama1b-q4,llama3b-q4,granite1b-q4,granite3b-q4")
    ap.add_argument("--reps", type=int, default=int(ENV.get("REPS", 3)))
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    table = model_table()
    models = [table[x] for x in a.models.split(",")]
    if not a.aggregate_only:
        jobs = list(plan(models))
        for i, (m, label, p, n, d, t, phase, ctx) in enumerate(jobs):
            for rep in range(1, a.reps + 1):
                run(m, label, p, n, d, t, rep)
            print(f"[{i+1}/{len(jobs)}] {m['name']} {label} t={t} done", flush=True)
    aggregate(models, a.reps)


if __name__ == "__main__":
    main()
