"""Plot bench.exe results. Usage: python plot_bench.py bench_t1.csv bench_t6.csv [...] --out mamba3_bench.png"""
import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--out", default="mamba3_bench.png")
a = ap.parse_args()

frames = []
for f in a.csv:
    lines = [l for l in open(f, encoding="utf-8-sig") if not l.startswith("#") and l.strip()]
    header, rows = lines[0], [l for l in lines[1:] if not l.startswith("kernel,")]
    frames.append(pd.read_csv(io.StringIO(header + "".join(rows))))
d = pd.concat(frames, ignore_index=True)
d["us"] = d["ns_per_step_mean"] / 1000
d["regime"] = d["layers"].map(lambda L: "in cache" if L == 1 else f"DRAM, {L} layers")
d["group"] = d["threads"].astype(str) + " thread(s)\n" + d["regime"]

COL = {"mamba2": "#9e9e9e", "siso": "#2a6fdb", "mimo": "#d9534f"}
LAB = {"mamba2": "Mamba-2", "siso": "Mamba-3 SISO", "mimo": "Mamba-3 MIMO (R=4)"}
groups = list(dict.fromkeys(d.sort_values(["threads", "layers"])["group"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 2]})
ax = axes[0]
av = d[d["impl"] == "avx2"]
for j, k in enumerate(["mamba2", "siso", "mimo"]):
    for i, g in enumerate(groups):
        r = av[(av["group"] == g) & (av["kernel"] == k)]
        if r.empty:
            continue
        r = r.iloc[0]
        x = i * 4 + j
        ax.bar(x, r.us, color=COL[k], label=LAB[k] if i == 0 else None)
        ax.errorbar(x, r.us, yerr=[[r.us - r.ns_per_step_min / 1000], [r.ns_per_step_max / 1000 - r.us]],
                    color="black", capsize=2, lw=1)
ax.set_xticks([i * 4 + 1 for i in range(len(groups))]); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("µs per layer per decode step (AVX2)")
ax.set_title("Decode-step time at equal state size (H=64, P=64, N=128)")
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.3)

ax = axes[1]
flop_ratio = None
for i, g in enumerate(groups):
    for impl, mk in (("avx2", "o"), ("plain", "s")):
        s = d[(d["group"] == g) & (d["impl"] == impl)].set_index("kernel")
        if not {"siso", "mimo"} <= set(s.index):
            continue
        flop_ratio = s.loc["mimo", "flops_per_step"] / s.loc["siso", "flops_per_step"]
        ax.scatter(i, s.loc["mimo", "us"] / s.loc["siso", "us"], marker=mk, s=60,
                   color="#d9534f" if impl == "avx2" else "#f0a8a6", label=impl if i == 0 else None, zorder=3)
if flop_ratio:
    ax.axhline(flop_ratio, color="black", ls="--", lw=1, label=f"FLOP ratio ({flop_ratio:.2f}x): fully compute-bound")
ax.axhline(1.0, color="black", ls=":", lw=1, label="1.0x: extra compute free")
ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("MIMO time / SISO time"); ax.set_ylim(0, (flop_ratio or 4) * 1.15)
ax.set_title("Cost of MIMO's extra compute")
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(a.out, dpi=150)

cols = ["kernel", "impl", "threads", "layers", "us", "arith_intensity", "GFLOPs", "GBps"]
print(d[cols].round(2).to_string(index=False))
