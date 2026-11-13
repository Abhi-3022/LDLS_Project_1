#!/usr/bin/env python3
"""Bonus analysis: does MIMO's extra compute come at little cost on this CPU?

Reads results/{verify,bench,ggml_m3}.csv (+ ../results/csv/stream.csv for bandwidth) and writes
  results/summary.md            tables + the answer paragraph with numbers
  ../plots/m3_*.png             time vs rank, roofline, thread scaling
Full-layer estimate: the SSM core is only part of a Mamba-3 layer; in_proj/out_proj are
GEMVs over Q4_K_M weights (0.5625 B/weight) whose time is bytes / measured read bandwidth.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from plots import SERIES, INK2  # noqa: E402  (shared rcParams + palette with the main figures)

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
RES = HERE / "results"
PLOTS = PROJ / "plots"
PLOTS.mkdir(exist_ok=True)

V_ORDER = ["m2", "siso", "mimo2", "mimo4", "mimo8"]
V_LABEL = {"m2": "Mamba-2", "siso": "Mamba-3 SISO", "mimo2": "MIMO R=2", "mimo4": "MIMO R=4", "mimo8": "MIMO R=8"}
Q4K_BYTES_PER_W = 144 / 256  # Q4_K block: 256 weights in 144 bytes


def layer_proj_bytes(R, d_model=2048, d_inner=4096, N=128, H=64, S=32):
    d_in_proj = 2 * d_inner + 2 * N * R + 3 * H + S          # [z, x, B, C, dt, A, trap, angle]
    return (d_model * d_in_proj + d_inner * d_model) * Q4K_BYTES_PER_W


def main():
    b = pd.read_csv(RES / "bench.csv")
    v = pd.read_csv(RES / "verify.csv")
    g = pd.read_csv(RES / "ggml_m3.csv") if (RES / "ggml_m3.csv").exists() else None
    stream = pd.read_csv(PROJ / "results/csv/stream.csv").groupby("threads").mean(numeric_only=True)

    agg = (b.groupby(["variant", "R", "isa", "threads", "regime"])
             .agg(us=("us_per_step", "mean"), us_std=("us_per_step", "std"), gflops=("gflops", "mean"),
                  gbps=("gbps", "mean"), ai=("ai_flop_per_byte", "first"), flops=("flops_per_step", "first"),
                  bytes=("bytes_per_step", "first"))
             .reset_index())
    agg.to_csv(RES / "bench_summary.csv", index=False)

    def us(var, isa, t, reg):
        r = agg[(agg.variant == var) & (agg.isa == isa) & (agg.threads == t) & (agg.regime == reg)]
        return float(r.us.iloc[0]) if len(r) else float("nan")

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, reg in zip(axes, ["cold", "hot"]):
        for isa, ls in [("avx512", "-"), ("avx2", "--")]:
            for t, c in [(1, SERIES[0]), (8, SERIES[1]), (32, SERIES[2])]:
                ys = [us(x, isa, t, reg) for x in ["siso", "mimo2", "mimo4", "mimo8"]]
                ax.plot([1, 2, 4, 8], ys, ls, marker="o", color=c, label=f"{isa}, {t} thr")
                if isa == "avx512":
                    ax.axhline(us("m2", isa, t, reg), color=c, ls=":", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["SISO", "2", "4", "8"])
        ax.set_xlabel("MIMO rank R"); ax.set_ylabel("µs per decode step (one layer)")
        ax.set_title(f"{'DRAM-resident state (pool 512 MiB)' if reg == 'cold' else 'cache-resident state (1 layer)'}")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("Mamba-3 decode step vs MIMO rank (solid AVX-512, dashed AVX2; dotted: Mamba-2 AVX-512, same state size)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "m3_time_vs_rank.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for t, c in [(1, SERIES[0]), (8, SERIES[1]), (32, SERIES[2])]:
        bw = stream.loc[t, "read_GBps"] if t in stream.index else None
        peak = agg[(agg.threads == t) & (agg.isa == "avx512")].gflops.max()
        xs = [0.1, 20]
        if bw:
            ax.plot(xs, [min(x * bw, peak) for x in xs], color=c, lw=0.8)
            ax.plot([peak / bw, 20], [peak, peak], color=c, lw=0.8, ls=":")
        sub = agg[(agg.threads == t) & (agg.isa == "avx512") & (agg.regime == "cold")]
        for r in sub.itertuples():
            ax.scatter(r.ai, r.gflops, color=c, s=25)
            ax.annotate(V_LABEL[r.variant].replace("Mamba-3 ", ""), (r.ai, r.gflops), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.3, 10)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of state traffic)"); ax.set_ylabel("GFLOP/s")
    ax.set_title("Roofline, DRAM-resident state, AVX-512\n(lines: read BW x AI capped at best measured GFLOP/s; blue 1, orange 8, green 32 thr)",
                 fontsize=8)
    ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(PLOTS / "m3_roofline.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for var, c in zip(V_ORDER, SERIES):
        sub = agg[(agg.variant == var) & (agg.isa == "avx512") & (agg.regime == "cold")].sort_values("threads")
        ax.plot(sub.threads, sub.us, marker="o", color=c, label=V_LABEL[var])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("threads"); ax.set_ylabel("µs per step")
    ax.set_title("Thread scaling, AVX-512, DRAM-resident state"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(PLOTS / "m3_thread_scaling.png", dpi=150); plt.close(fig)

    # ---------------- tables + answer ----------------
    L = []
    L.append("# Mamba-3 decode step on CPU: results\n")
    L.append("Layer shape: d_model 2048, d_inner 4096, 64 heads x head dim 64, d_state 128, 1 B/C group, "
             "RoPE on half the state (32 angles). State per layer = 2 MiB fp32 (same for Mamba-2).\n")
    L.append("## Correctness (1000 decode steps vs official fp64 reference, tolerance 1e-5 relative)\n")
    vv = v[v.threads == 1][["case", "isa", "rel_err", "state_rel_err", "torch_fp32_rel_err", "torch_fp32_state_rel_err", "result"]]
    L.append(vv.to_markdown(index=False, floatfmt=".2e"))
    L.append(f"\nAll {len(v)} runs (1 and 8 threads): {(v.result == 'PASS').sum()} PASS, {(v.result != 'PASS').sum()} FAIL.\n")

    L.append("## Time per decode step (µs, mean of 3 reps), AVX-512\n")
    rows = []
    for reg in ["cold", "hot"]:
        for t in sorted(agg.threads.unique()):
            row = {"state": reg, "threads": t}
            for var in V_ORDER:
                row[V_LABEL[var]] = round(us(var, "avx512", t, reg), 1)
            row["MIMO4 / SISO"] = round(us("mimo4", "avx512", t, reg) / us("siso", "avx512", t, reg), 2)
            rows.append(row)
    L.append(pd.DataFrame(rows).to_markdown(index=False))

    L.append("\n## FLOPs, bytes and arithmetic intensity per step\n")
    ai = agg[(agg.isa == "avx512") & (agg.threads == 1) & (agg.regime == "cold")][["variant", "flops", "bytes", "ai"]]
    ai["variant"] = ai.variant.map(V_LABEL)
    L.append(ai.to_markdown(index=False, floatfmt=".3g"))

    L.append("\n## Same comparison per ISA, DRAM-resident state, 1 thread (µs)\n")
    rows = []
    for isa in ["scalar", "avx2", "avx512"]:
        row = {"isa": isa}
        for var in V_ORDER:
            row[V_LABEL[var]] = round(us(var, isa, 1, "cold"), 1)
        row["MIMO4 / SISO"] = round(us("mimo4", isa, 1, "cold") / us("siso", isa, 1, "cold"), 2)
        rows.append(row)
    L.append(pd.DataFrame(rows).to_markdown(index=False))

    # full-layer estimate
    L.append("\n## Full layer estimate (core measured + in/out projections at measured read bandwidth, Q4_K_M weights)\n")
    rows = []
    for t in [1, 8, 32]:
        if t not in stream.index:
            continue
        bw = stream.loc[t, "read_GBps"] * 1e9
        for var, R in [("siso", 1), ("mimo4", 4)]:
            pb = layer_proj_bytes(R)
            core = us(var, "avx512", t, "cold")
            proj = pb / bw * 1e6
            rows.append({"threads": t, "variant": V_LABEL[var], "proj MB": round(pb / 1e6, 2),
                         "proj µs": round(proj, 1), "core µs": round(core, 1), "layer µs": round(proj + core, 1)})
    lay = pd.DataFrame(rows)
    L.append(lay.to_markdown(index=False))

    if g is not None:
        L.append("\n## ggml integration\n")
        L.append(g.to_markdown(index=False, floatfmt=".3g"))

    # the answer paragraph, numbers filled in
    c1s, c1m = us("siso", "avx512", 1, "cold"), us("mimo4", "avx512", 1, "cold")
    c8s, c8m = us("siso", "avx512", 8, "cold"), us("mimo4", "avx512", 8, "cold")
    h8s, h8m = us("siso", "avx512", 8, "hot"), us("mimo4", "avx512", 8, "hot")
    a2s, a2m = us("siso", "avx2", 1, "cold"), us("mimo4", "avx2", 1, "cold")
    m2c1 = us("m2", "avx512", 1, "cold")
    fl = agg[(agg.variant == "mimo4")].flops.iloc[0] / agg[(agg.variant == "siso")].flops.iloc[0]

    def layer(t, var):
        r = lay[(lay.threads == t) & (lay.variant == V_LABEL[var])]
        return float(r["layer µs"].iloc[0]) if len(r) else float("nan")
    L.append("\n## Answer: does MIMO's extra compute come at little cost on this CPU?\n")
    L.append(
        f"Partly. MIMO at rank 4 does {fl:.1f}x the FLOPs of SISO per step (25 vs 7 FLOP per state element) on "
        f"the same 2 MiB state, so its arithmetic intensity rises from {ai[ai.variant=='Mamba-3 SISO'].ai.iloc[0]:.2f} to "
        f"{ai[ai.variant=='MIMO R=4'].ai.iloc[0]:.2f} FLOP/B. When the state must stream from DRAM, as it does for a "
        f"deep model, the extra compute is mostly hidden behind the memory traffic: one AVX-512 core takes "
        f"{c1s:.0f} µs (SISO) vs {c1m:.0f} µs (MIMO-4), {c1m/c1s:.2f}x, and 8 cores take {c8s:.0f} vs {c8m:.0f} µs "
        f"({c8m/c8s:.2f}x), far below the {fl:.1f}x FLOP ratio. Once enough cores stream the state at the "
        f"socket's bandwidth (32 threads: {us('siso','avx512',32,'cold'):.0f} vs {us('mimo4','avx512',32,'cold'):.0f} µs, "
        f"{agg[(agg.variant=='siso')&(agg.isa=='avx512')&(agg.threads==32)&(agg.regime=='cold')].gbps.iloc[0]:.0f} GB/s of "
        f"state traffic) MIMO-4 is {us('mimo4','avx512',32,'cold')/us('siso','avx512',32,'cold'):.2f}x, i.e. effectively "
        f"free, which is the GPU argument holding on a CPU. With AVX2 only, as on most laptops, the ratio grows to "
        f"{a2m/a2s:.2f}x on one core ({a2s:.0f} vs {a2m:.0f} µs), because half-width vectors halve the compute "
        f"headroom. When the state is cache-resident (8 threads, 256 KiB of state per core in L2), the step "
        f"becomes compute-bound and MIMO-4 costs {h8m/h8s:.2f}x ({h8s:.0f} vs {h8m:.0f} µs). For reference, Mamba-2 "
        f"with the same state takes {m2c1:.0f} µs on one core (DRAM-resident). For a full layer, the Q4_K_M "
        f"in_proj/out_proj reads (~{layer_proj_bytes(1)/1e6:.1f} MB for SISO, ~{layer_proj_bytes(4)/1e6:.1f} MB for "
        f"MIMO-4, whose B/C projections are 4x wider) dominate: per layer, SISO and MIMO-4 come to about "
        f"{layer(1,'siso'):.0f} vs {layer(1,'mimo4'):.0f} µs on 1 thread (+{100*(layer(1,'mimo4')/layer(1,'siso')-1):.0f}%) "
        f"and {layer(8,'siso'):.0f} vs {layer(8,'mimo4'):.0f} µs on 8 threads "
        f"(+{100*(layer(8,'mimo4')/layer(8,'siso')-1):.0f}%). So on this CPU MIMO's extra compute is not free inside "
        f"the SSM kernel, but at the level of a decode step it costs on the order of 10%, and less when the kernel "
        f"is memory-bound.\n")
    (RES / "summary.md").write_text("\n".join(L))
    print("\n".join(L[-2:]))
    print(f"wrote {RES/'summary.md'} and plots/m3_*.png")


if __name__ == "__main__":
    main()
