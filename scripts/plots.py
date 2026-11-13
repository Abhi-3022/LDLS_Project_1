#!/usr/bin/env python3
"""All figures + a summary of key numbers, from the CSVs in results/csv.

Required plots (project brief): decode speed vs context, prefill speed vs prompt length,
thread scaling, operator time breakdown, predicted vs measured decode speed.
Option-6 plots: active vs total parameters, expert-count experiment, routing cost,
llama.cpp vs ik_llama.cpp. Plus memory bandwidth and peak RSS.
Each figure is skipped (with a note) when its input CSV does not exist yet.

Colours: categorical slots 1-6 of the dataviz reference palette in fixed order, one per model;
marker shape encodes architecture (circle = dense, square = MoE) so identity is never colour-only.
"""
import os
import pathlib
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = pathlib.Path(__file__).resolve().parents[1]
CSV = PROJ / "results" / "csv"
PLOTS = PROJ / "plots"
PLOTS.mkdir(exist_ok=True)
T_MAIN = int(os.environ.get("T_MAIN", 32))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["llama1b-q4", "llama1b-q8", "llama3b-q4", "granite1b-q4", "granite1b-q8", "granite3b-q4"]
COLOR = dict(zip(MODELS, SERIES))
LABEL = {"llama1b-q4": "Llama-3.2-1B Q4_K_M", "llama1b-q8": "Llama-3.2-1B Q8_0", "llama3b-q4": "Llama-3.2-3B Q4_K_M",
         "granite1b-q4": "Granite-1B-A400M Q4_K_M", "granite1b-q8": "Granite-1B-A400M Q8_0",
         "granite3b-q4": "Granite-3B-A800M Q4_K_M"}
MARK = {m: ("s" if m.startswith("granite") else "o") for m in MODELS}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def load(name):
    p = CSV / name
    return pd.read_csv(p) if p.exists() else None


def stats(df, keys):
    return df.groupby(keys).tok_s.agg(["mean", "std", "count"]).reset_index()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=160)
    plt.close(fig)
    print("  wrote plots/" + name)


def pred_table():
    return load("roofline_pred.csv")


# ---------------------------------------------------------------- figures
def fig_bandwidth():
    s = load("stream.csv")
    g = s.groupby("threads").agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for col, lab, c in [("read_GBps", "read-only reduction", SERIES[0]), ("copy_GBps", "STREAM Copy", SERIES[1]),
                        ("triad_GBps", "STREAM Triad", SERIES[2])]:
        ax.errorbar(g.index, g[col]["mean"], yerr=g[col]["std"], marker="o", color=c, label=lab, capsize=2)
    ax.set_xlabel("threads (physical cores, socket 0)"); ax.set_ylabel("GB/s")
    ax.set_title("Memory bandwidth vs threads"); ax.legend(); ax.set_ylim(0)
    save(fig, "bandwidth.png")


def fig_thread_scaling():
    d = load("bench_threads.csv")
    pr = pred_table()
    st = stats(d, ["model", "test", "threads"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, test, title in [(axes[0], "pp", "Prefill (pp512)"), (axes[1], "tg", "Decode (tg128)")]:
        for m in MODELS:
            s = st[(st.model == m) & (st.test == test)]
            if s.empty:
                continue
            ax.errorbar(s.threads, s["mean"], yerr=s["std"], marker=MARK[m], color=COLOR[m], label=LABEL[m], capsize=2)
            if test == "tg" and pr is not None:
                p = pr[(pr.model == m) & (pr.bw_source == "read") & (pr.context == 64)].sort_values("threads")
                ax.plot(p.threads, p.pred_tok_s, ":", color=COLOR[m], lw=1.2)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 55]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 55])
        ax.set_xlabel("threads"); ax.set_ylabel("tokens / s"); ax.set_title(title)
    axes[1].text(0.02, 0.02, "dotted: roofline prediction (read BW / bytes per token)", transform=axes[1].transAxes,
                 fontsize=7, color=INK2)
    axes[0].legend(fontsize=7)
    save(fig, "thread_scaling.png")


def fig_lengths():
    d = load("bench_lengths.csv")
    pr = pred_table()
    st = stats(d, ["model", "test", "context"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in MODELS:
        s = st[(st.model == m) & (st.test == "tg")]
        if s.empty:
            continue
        ax.errorbar(s.context, s["mean"], yerr=s["std"], marker=MARK[m], color=COLOR[m], label=LABEL[m], capsize=2)
        if pr is not None:
            p = pr[(pr.model == m) & (pr.bw_source == "read") & (pr.threads == T_MAIN)].sort_values("context")
            p = p[(p.context >= 960) & (p.context <= s.context.max())]
            ax.plot(p.context, p.pred_tok_s, ":", color=COLOR[m], lw=1.2)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length (tokens, KV cache size at the end of the 128 decoded tokens)")
    ax.set_ylabel("decode tokens / s"); ax.set_title(f"Decode speed vs context ({T_MAIN} threads; dotted = roofline)")
    ax.legend(fontsize=7)
    save(fig, "decode_vs_context.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in MODELS:
        s = st[(st.model == m) & (st.test == "pp")]
        if s.empty:
            continue
        ax.errorbar(s.context, s["mean"], yerr=s["std"], marker=MARK[m], color=COLOR[m], label=LABEL[m], capsize=2)
    ax.set_xscale("log", base=2); ax.set_xlabel("prompt length (tokens)"); ax.set_ylabel("prefill tokens / s")
    ax.set_ylim(0); ax.set_title(f"Prefill speed vs prompt length ({T_MAIN} threads)"); ax.legend(fontsize=7)
    save(fig, "prefill_vs_prompt.png")

    r = d.groupby(["model", "test", "context"]).maxrss_gb.max().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in MODELS:
        s = r[(r.model == m) & (r.test == "tg")]
        if not s.empty:
            ax.plot(s.context, s.maxrss_gb, marker=MARK[m], color=COLOR[m], label=LABEL[m])
    ax.axhline(4.5, color=INK2, ls="--", lw=1); ax.text(1100, 4.55, "4.5 GB budget", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xlabel("context length"); ax.set_ylabel("peak RSS (GB)"); ax.set_ylim(0, 5)
    ax.set_title("Peak memory during decode"); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "memory_vs_context.png")


def fig_pred_vs_meas():
    pr = pred_table()
    rows = []
    for name, sub in [("bench_threads.csv", "threads"), ("bench_lengths.csv", "lengths")]:
        d = load(name)
        if d is None:
            continue
        d = d[d.test == "tg"]
        st = d.groupby(["model", "threads", "n_depth"]).tok_s.agg(["mean", "std"]).reset_index()
        for r in st.itertuples():
            ctx = r.n_depth + 64
            p = pr[(pr.model == r.model) & (pr.threads == r.threads) & (pr.bw_source == "read")]
            if p.empty:
                continue
            p = p.iloc[(p.context - ctx).abs().argsort()[:1]]
            rows.append(dict(model=r.model, threads=r.threads, context=ctx, measured=r.mean, measured_std=r.std,
                             predicted=float(p.pred_tok_s.iloc[0]), predicted_all_experts=float(p.pred_tok_s_if_all_experts.iloc[0]),
                             suite=sub))
    df = pd.DataFrame(rows)
    df["efficiency"] = df.measured / df.predicted
    df.to_csv(CSV / "pred_vs_meas.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    for m in MODELS:
        s = df[df.model == m]
        if not s.empty:
            ax.scatter(s.predicted, s.measured, marker=MARK[m], color=COLOR[m], label=LABEL[m], s=28,
                       edgecolor=SURFACE, linewidth=1)
    lim = [3, max(df.predicted.max(), df.measured.max()) * 1.3]
    ax.plot(lim, lim, color=INK2, lw=1); ax.plot(lim, [x * 0.5 for x in lim], color=INK2, lw=0.8, ls="--")
    ax.text(lim[1] / 1.6, lim[1] / 1.45, "measured = predicted", fontsize=7, color=INK2, rotation=38)
    ax.text(lim[1] / 1.6, lim[1] / 3.4, "50 %", fontsize=7, color=INK2, rotation=38)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted decode tok/s (read BW / bytes per token)"); ax.set_ylabel("measured decode tok/s")
    ax.set_title("Predicted vs measured decode\n(all thread counts and contexts)"); ax.legend(fontsize=7)
    save(fig, "pred_vs_measured.png")
    return df


def fig_active_vs_total(pvm):
    d = pvm[(pvm.threads == T_MAIN) & (pvm.suite == "threads")]
    if d.empty:
        return
    mb = load("model_bytes.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(d))
    w = 0.26
    for i, (col, lab, c) in enumerate([("measured", "measured", SERIES[0]),
                                       ("predicted", "predicted from ACTIVE bytes", SERIES[1]),
                                       ("predicted_all_experts", "predicted from TOTAL bytes", SERIES[2])]):
        ax.bar(xs + (i - 1) * w, d[col], width=w - 0.02, color=c, label=lab)
    ax.set_xticks(xs)
    short = {"llama1b-q4": "Llama-1B Q4", "llama1b-q8": "Llama-1B Q8", "llama3b-q4": "Llama-3B Q4",
             "granite1b-q4": "Granite-1B Q4", "granite1b-q8": "Granite-1B Q8", "granite3b-q4": "Granite-3B Q4"}
    ax.set_xticklabels([f"{short[m]}\n{mb.loc[m,'params_active']/1e9:.2f} / {mb.loc[m,'params_total']/1e9:.2f} B"
                        for m in d.model], fontsize=7.5)
    ax.set_xlabel("model  (active / total parameters)")
    ax.set_ylabel("decode tokens / s"); ax.set_title(f"Does decode follow active or total parameters? (tg128, {T_MAIN} threads)")
    ax.legend(fontsize=7)
    save(fig, "active_vs_total.png")


def fig_ops():
    g = load("opprof_groups.csv")
    if g is None:
        print("  (no opprof_groups.csv yet)"); return
    order = ["MoE expert matmuls", "dense FFN matmuls", "attention projections (QKV, O)",
             "attention (QK/softmax/V over KV cache)", "LM head (tied embedding)", "MoE routing + combine",
             "KV cache write", "norms", "rope / elementwise / other"]
    colors = dict(zip(order, SERIES + ["#9a9890"]))
    g = g[g.threads == T_MAIN]
    agg = g.groupby(["model", "phase", "context", "group"]).ms_per_token.mean().reset_index()
    for phase, fname in [("decode", "op_breakdown_decode.png"), ("prefill", "op_breakdown_prefill.png")]:
        s = agg[agg.phase == phase]
        runs = s[["model", "context"]].drop_duplicates().sort_values(["model", "context"])
        runs = runs[runs.model.isin(MODELS)]
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(runs) + 1.4))
        for i, r in enumerate(runs.itertuples()):
            left = 0.0
            tot = s[(s.model == r.model) & (s.context == r.context)].ms_per_token.sum()
            for grp in order:
                v = s[(s.model == r.model) & (s.context == r.context) & (s.group == grp)].ms_per_token.sum()
                if v <= 0:
                    continue
                ax.barh(i, v / tot * 100, left=left, color=colors[grp], edgecolor=SURFACE, linewidth=1,
                        label=grp if i == 0 or grp not in ax.get_legend_handles_labels()[1] else None)
                left += v / tot * 100
            ax.text(101, i, f"{tot:.2f} ms/tok" if phase == "decode" else f"{tot*1000:.0f} µs/tok", va="center",
                    fontsize=7, color=INK2)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([f"{LABEL[r.model]}, ctx {r.context}" for r in runs.itertuples()], fontsize=7)
        ax.invert_yaxis(); ax.set_xlim(0, 115); ax.set_xlabel("% of graph time (thread 0 wall incl. barrier wait)")
        ax.set_title(f"Operator time breakdown: {phase} ({T_MAIN} threads)")
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / max(1, len(runs))))
        save(fig, fname)

    # attention share vs context (decode)
    s = agg[(agg.phase == "decode")]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in MODELS:
        sm = s[s.model == m]
        if sm.empty:
            continue
        tot = sm.groupby("context").ms_per_token.sum()
        att = sm[sm.group.isin(["attention (QK/softmax/V over KV cache)", "KV cache write"])].groupby("context").ms_per_token.sum()
        ax.plot(tot.index, 100 * att / tot, marker=MARK[m], color=COLOR[m], label=LABEL[m])
    ax.axhline(50, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log", base=2); ax.set_xlabel("context length"); ax.set_ylabel("% of decode time in attention over KV")
    ax.set_ylim(0, 100); ax.set_title("When does attention dominate decode?"); ax.legend(fontsize=7)
    save(fig, "attention_share.png")


def fig_experts():
    d = load("bench_experts.csv")
    if d is None:
        print("  (no bench_experts.csv yet)"); return
    ppl = load("experts_ppl.csv")
    mb = load("model_bytes.csv").set_index("model")
    d["base"] = d.model.str.rsplit("-k", n=1).str[0]
    d["k"] = d.model.str.rsplit("-k", n=1).str[1].astype(int)
    st = d.groupby(["base", "k", "test", "n_depth"]).tok_s.agg(["mean", "std"]).reset_index()
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for base in sorted(st.base.unique()):
        s = st[(st.base == base) & (st.test == "tg") & (st.n_depth == 0)].sort_values("k")
        ms = 1000 / s["mean"]
        ax.errorbar(s.k, ms, yerr=ms * s["std"] / s["mean"], marker="s", color=COLOR[base], label=f"{LABEL[base]} measured", capsize=2)
        b = mb.loc[base]
        per_expert = b.bytes_experts_total / b.n_expert
        fixed = b.decode_bytes_active - b.bytes_experts_total * b.n_expert_used / b.n_expert
        slope, icpt = np.polyfit(s.k, ms, 1)
        for kk, mm, sd in zip(s.k, ms, s["std"]):
            rows.append(dict(model=base, k=kk, decode_ms_per_tok=mm, active_bytes=fixed + per_expert * kk))
        ax.plot(s.k, icpt + slope * s.k, ":", color=COLOR[base], lw=1.2)
        ax.text(s.k.max(), icpt + slope * s.k.max(), f" {slope*1000:.0f} µs per extra expert", fontsize=7, color=INK2,
                va="center")
        dense = "llama1b-q4" if base.startswith("granite1b") else "llama3b-q4"
        dd = load("bench_threads.csv")
        if dd is not None:
            v = dd[(dd.model == dense) & (dd.test == "tg") & (dd.threads == T_MAIN)].tok_s.mean()
            if v == v:
                ax.axhline(1000 / v, color=COLOR[dense], lw=1, ls="--")
                ax.text(1, 1000 / v, f" dense {LABEL[dense]}", fontsize=7, color=INK2, va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 40]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 40])
    ax.set_xlabel("experts used per token k (default 8)"); ax.set_ylabel("ms per decoded token (tg128)")
    ax.set_title(f"Own experiment: decode time vs routed experts ({T_MAIN} thr)"); ax.legend(fontsize=7)
    ax = axes[1]
    if ppl is not None:
        ppl["base"] = ppl.model.str.rsplit("-k", n=1).str[0]
        for base in sorted(ppl.base.unique()):
            s = ppl[ppl.base == base].sort_values("k")
            ax.errorbar(s.k, s.ppl, yerr=s.ppl_err, marker="s", color=COLOR[base], label=LABEL[base], capsize=2)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 2, 4, 8, 16, 32, 40])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 40]); ax.set_xlabel("experts used per token k")
        ax.set_ylabel("WikiText-2 perplexity"); ax.set_title("Quality cost of changing k"); ax.legend(fontsize=7)
    pd.DataFrame(rows).to_csv(CSV / "experts_summary.csv", index=False)
    # the left panel alone, for the report
    fig2, ax2 = plt.subplots(figsize=(5.2, 4))
    for base in sorted(st.base.unique()):
        s = st[(st.base == base) & (st.test == "tg") & (st.n_depth == 0)].sort_values("k")
        ms = 1000 / s["mean"]
        ax2.errorbar(s.k, ms, yerr=ms * s["std"] / s["mean"], marker="s", color=COLOR[base], label=LABEL[base], capsize=2)
        slope, icpt = np.polyfit(s.k, ms, 1)
        ax2.plot(s.k, icpt + slope * s.k, ":", color=COLOR[base], lw=1.2)
        dense = "llama1b-q4" if base.startswith("granite1b") else "llama3b-q4"
        dd = load("bench_threads.csv")
        v = dd[(dd.model == dense) & (dd.test == "tg") & (dd.threads == T_MAIN)].tok_s.mean()
        ax2.axhline(1000 / v, color=COLOR[dense], lw=1, ls="--", label=f"dense {LABEL[dense].split(' ')[0]}")
    ax2.set_xscale("log", base=2); ax2.set_xticks([1, 2, 4, 8, 16, 32, 40]); ax2.set_xticklabels([1, 2, 4, 8, 16, 32, 40])
    ax2.set_xlabel("experts used per token k (trained: 8)"); ax2.set_ylabel("ms per decoded token")
    ax2.set_title(f"Decode time vs routed experts ({T_MAIN} thr)"); ax2.legend(fontsize=7)
    save(fig2, "experts_k_left.png")
    save(fig, "experts_k.png")


def fig_ik():
    d = load("bench_ik.csv")
    if d is None:
        print("  (no bench_ik.csv yet)"); return
    d["cfg"] = d.engine + d.tag.fillna("").apply(lambda t: f" ({t})" if t else "")
    st = d.groupby(["model", "test", "cfg"]).tok_s.agg(["mean", "std"]).reset_index()
    cfgs = list(dict.fromkeys(st.cfg))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, test, title in [(axes[0], "pp", "Prefill pp512"), (axes[1], "tg", "Decode tg128")]:
        s = st[st.test == test]
        models = [m for m in MODELS if m in set(s.model)]
        xs = np.arange(len(models)); w = 0.8 / len(cfgs)
        for i, c in enumerate(cfgs):
            v = [s[(s.model == m) & (s.cfg == c)]["mean"].mean() for m in models]
            e = [s[(s.model == m) & (s.cfg == c)]["std"].mean() for m in models]
            ax.bar(xs + (i - (len(cfgs) - 1) / 2) * w, v, yerr=e, width=w - 0.02, color=SERIES[i], label=c, capsize=2)
        ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in models], fontsize=7)
        ax.set_ylabel("tokens / s"); ax.set_title(f"{title}, {T_MAIN} threads")
    axes[0].legend(fontsize=7)
    save(fig, "ik_vs_llamacpp.png")


def main():
    for f in [fig_bandwidth, fig_thread_scaling, fig_lengths]:
        try:
            f()
        except Exception:
            print(f"  skipped {f.__name__}:"); traceback.print_exc(limit=1)
    pvm = None
    try:
        pvm = fig_pred_vs_meas()
        fig_active_vs_total(pvm)
    except Exception:
        print("  skipped prediction plots:"); traceback.print_exc(limit=1)
    for f in [fig_ops, fig_experts, fig_ik]:
        try:
            f()
        except Exception:
            print(f"  skipped {f.__name__}:"); traceback.print_exc(limit=1)


if __name__ == "__main__":
    main()
