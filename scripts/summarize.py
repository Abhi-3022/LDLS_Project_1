#!/usr/bin/env python3
"""Collect every key number into results/summary.md (mean ± std over the 3 repetitions).
Sections whose CSV is missing are skipped."""
import json
import os
import pathlib

import pandas as pd

PROJ = pathlib.Path(__file__).resolve().parents[1]
CSV = PROJ / "results" / "csv"
T_MAIN = int(os.environ.get("T_MAIN", 32))
OUT = PROJ / "results" / "summary.md"


def load(n):
    p = CSV / n
    return pd.read_csv(p) if p.exists() else None


def ms(df, keys, col="tok_s"):
    g = df.groupby(keys)[col].agg(["mean", "std"]).reset_index()
    g["value"] = [f"{m:.1f} ± {s:.1f}" if s == s else f"{m:.1f}" for m, s in zip(g["mean"], g["std"])]
    return g


def main():
    L = ["# Results summary\n", f"All runs: socket 0, physical cores only (core 1 excluded), memory bound to node 0, "
         f"main thread count T_MAIN = {T_MAIN}. Values are mean ± std over 3 separate processes.\n"]
    hw = PROJ / "hw" / "hardware.json"
    if hw.exists():
        h = json.loads(hw.read_text())
        L.append(f"**Hardware:** {h['cpu_model']}, {h['sockets']} sockets x {h['cores_per_socket']} cores "
                 f"(SMT {h['threads_per_core']}), caches {h['cache_per_instance']}, RAM {h['ram_gb']} GB, "
                 f"SIMD: {' '.join(h['simd'])}.\n")

    s = load("stream.csv")
    if s is not None:
        g = s.groupby("threads")[["read_GBps", "copy_GBps", "triad_GBps"]].agg(["mean", "std"])
        rows = [{"threads": t, **{c: f"{g.loc[t, (c, 'mean')]:.1f} ± {g.loc[t, (c, 'std')]:.1f}"
                                  for c in ["read_GBps", "copy_GBps", "triad_GBps"]}} for t in g.index]
        L += ["## Memory bandwidth (GB/s)\n", pd.DataFrame(rows).to_markdown(index=False), ""]

    mb = load("model_bytes.csv")
    if mb is not None:
        t = mb[["model", "params_total", "params_active", "file_bytes", "decode_bytes_active", "decode_bytes_total",
                "bytes_lm_head", "kv_bytes_per_token", "n_expert", "n_expert_used"]].copy()
        for c in ["params_total", "params_active"]:
            t[c] = (t[c] / 1e9).round(3)
        for c in ["file_bytes", "decode_bytes_active", "decode_bytes_total", "bytes_lm_head"]:
            t[c] = (t[c] / 1e9).round(3)
        t["kv_bytes_per_token"] = (t.kv_bytes_per_token / 1024).round(0)
        t.columns = ["model", "params (B)", "active params (B)", "file GB", "bytes/token active (GB)",
                     "bytes/token all experts (GB)", "LM head GB", "KV KiB per ctx token", "experts", "used"]
        L += ["## Bytes read per decoded token (roofline input)\n", t.to_markdown(index=False), ""]

    p = load("perplexity.csv")
    if p is not None:
        L += ["## Correctness: WikiText-2 perplexity (first 100k chars of test, ctx 512)\n",
              p[["model", "ppl", "ppl_err", "n_chunks", "maxrss_gb", "role"]].to_markdown(index=False), ""]

    d = load("bench_threads.csv")
    if d is not None:
        g = ms(d, ["model", "test", "threads"])
        piv = g.pivot_table(index=["model", "test"], columns="threads", values="value", aggfunc="first")
        L += ["## Thread scaling (tok/s; pp512 prefill, tg128 decode)\n", piv.to_markdown(), ""]
        best = g.loc[g.groupby(["model", "test"])["mean"].idxmax()][["model", "test", "threads", "value"]]
        L += ["Best thread count per model/test:\n", best.to_markdown(index=False), ""]

    d = load("bench_lengths.csv")
    if d is not None:
        g = ms(d, ["model", "test", "context"])
        r = d.groupby(["model", "test", "context"]).maxrss_gb.max().round(2).reset_index()
        g = g.merge(r, on=["model", "test", "context"])
        g["over budget"] = (g.maxrss_gb > 4.5).map({True: "YES", False: ""})
        L += [f"## Prefill vs prompt length and decode vs context ({T_MAIN} threads)\n",
              g[["model", "test", "context", "value", "maxrss_gb", "over budget"]].to_markdown(index=False), ""]
        sw = d[(d.swap_in > 0) | (d.swap_out > 0)]
        L.append(f"Runs with swap activity: {len(sw)}; runs over the 4.5 GB budget: {int(d.over_budget.sum())}.\n")

    pv = load("pred_vs_meas.csv")
    if pv is not None:
        t = pv[(pv.threads == T_MAIN)].copy()
        t["efficiency"] = (100 * t.measured / t.predicted).round(0)
        L += ["## Roofline: predicted vs measured decode\n",
              t[["model", "suite", "threads", "context", "predicted", "predicted_all_experts", "measured", "efficiency"]]
              .round(1).to_markdown(index=False), ""]
        e = pv.groupby("threads").apply(lambda x: (x.measured / x.predicted).mean()).round(2)
        L.append("Mean measured/predicted by thread count: " + ", ".join(f"{k}: {v}" for k, v in e.items()) + "\n")

    g = load("opprof_groups.csv")
    o = load("opprof_ops.csv")
    if g is not None:
        gg = g.groupby(["model", "phase", "context", "threads", "group"])[["ms_per_token", "own_ms_per_token", "share"]].mean().reset_index()
        gg["share %"] = (100 * gg.share).round(1)
        gg["sync %"] = (100 * (1 - gg.own_ms_per_token / gg.ms_per_token)).round(0)
        L += ["## Operator groups (ms per token; sync % = share of the op's wall time spent waiting at its barrier)\n",
              gg[["model", "phase", "context", "threads", "group", "ms_per_token", "share %", "sync %"]].round(3).to_markdown(index=False), ""]
        rt = gg[gg.group == "MoE routing + combine"]
        if len(rt):
            L += ["### Expert routing cost\n", rt[["model", "phase", "context", "threads", "ms_per_token", "share %"]].round(3).to_markdown(index=False), ""]
    if o is not None:
        oo = o[o.threads == T_MAIN].groupby(["model", "phase", "context", "op", "name"])[["ms_per_token", "share"]].mean().reset_index()
        top = oo.sort_values("ms_per_token", ascending=False).groupby(["model", "phase", "context"]).head(3)
        top = top.sort_values(["model", "phase", "context", "ms_per_token"], ascending=[True, True, True, False])
        top["share %"] = (100 * top.share).round(1)
        L += ["## Top-3 operators per phase\n", top[["model", "phase", "context", "op", "name", "ms_per_token", "share %"]].round(3).to_markdown(index=False), ""]

    e = load("bench_experts.csv")
    if e is not None:
        g = ms(e, ["model", "test", "n_depth"])
        piv = g.pivot_table(index="model", columns=["test", "n_depth"], values="value", aggfunc="first")
        L += ["## Own experiment: experts per token k (tok/s)\n", piv.to_markdown(), ""]
        ep = load("experts_ppl.csv")
        if ep is not None:
            L += [ep.to_markdown(index=False), ""]

    ik = load("bench_ik.csv")
    if ik is not None:
        ik["cfg"] = ik.engine + ik.tag.fillna("").apply(lambda t: f" ({t})" if t else "")
        g = ms(ik, ["model", "test", "cfg"])
        L += ["## llama.cpp vs ik_llama.cpp (tok/s)\n",
              g.pivot_table(index=["model", "test"], columns="cfg", values="value", aggfunc="first").to_markdown(), ""]

    for f in sorted(CSV.glob("reference_*.csv")):
        r = pd.read_csv(f)
        g = ms(r, ["host", "test", "threads"])
        L += [f"## Shared reference test ({f.name})\n", g[["host", "test", "threads", "value"]].to_markdown(index=False), ""]

    bs = PROJ / "bonus_mamba3" / "results" / "summary.md"
    if bs.exists():
        L.append("## Bonus\nSee bonus_mamba3/results/summary.md.\n")
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
