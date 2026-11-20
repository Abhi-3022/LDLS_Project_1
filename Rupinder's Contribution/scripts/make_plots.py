"""Draw the five plots the project requires, plus the own-experiment plot.

Usage:  python scripts/make_plots.py results
Reads   results/summary.csv, roofline.csv, operator_breakdown.csv, exp_pinning/summary.csv,
        boost_off/capped_scaling.csv
Writes  results/plots/*.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # draw to files, no window
import matplotlib.pyplot as plt

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
plots = results / "plots"
plots.mkdir(exist_ok=True)

MODELS = ["1B Q4_K_M", "1B Q8_0", "3B Q4_K_M"]
COLOR = {"1B Q4_K_M": "tab:blue", "1B Q8_0": "tab:orange", "3B Q4_K_M": "tab:green"}


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(plots / name, dpi=150)
    plt.close(fig)
    print("saved", plots / name)


runs = read(results / "summary.csv")
for r in runs:
    for k in ("threads", "prompt", "context"):
        r[k] = int(r[k])
    for k in ("tok_s", "stddev"):
        r[k] = float(r[k])


def select(kind, model, threads=None, prompt=None, context=None):
    """Runs matching the given settings, sorted by threads, prompt and context."""
    out = [r for r in runs if r["kind"] == kind and r["model"] == model
           and (threads is None or r["threads"] == threads)
           and (prompt is None or r["prompt"] == prompt)
           and (context is None or r["context"] == context)]
    return sorted(out, key=lambda r: (r["threads"], r["prompt"], r["context"]))


# ---- 1. decode speed vs context length (6 threads) ------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m in MODELS:
    pts = select("decode", m, threads=6)
    ax.errorbar([r["context"] for r in pts], [r["tok_s"] for r in pts], yerr=[r["stddev"] for r in pts],
                marker="o", capsize=3, color=COLOR[m], label=m)
ax.set_xscale("log", base=2)
ax.set_xticks([1024, 8192, 16384], ["1k", "8k", "16k"])
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Decode speed (tokens/s)")
ax.set_title("Decode speed vs context length (6 P-cores)")
ax.grid(alpha=0.3); ax.legend()
save(fig, "1_decode_vs_context.png")

# ---- 2. prefill speed vs prompt length (6 threads) ------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m in MODELS:
    pts = select("prefill", m, threads=6)
    ax.errorbar([r["prompt"] for r in pts], [r["tok_s"] for r in pts], yerr=[r["stddev"] for r in pts],
                marker="o", capsize=3, color=COLOR[m], label=m)
ax.set_xscale("log", base=2)
ax.set_xticks([512, 4096, 8192, 16384], ["512", "4k", "8k", "16k"])
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("Prefill speed (tokens/s)")
ax.set_title("Prefill speed vs prompt length (6 P-cores)")
ax.grid(alpha=0.3); ax.legend()
save(fig, "2_prefill_vs_prompt.png")

# ---- 3. thread scaling: speed-up over 1 thread ------------------------------------------
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for m in MODELS:
    for ax, kind, setting, value, style in [(left, "prefill", "prompt", 512, "-"),
                                            (left, "prefill", "prompt", 4096, "--"),
                                            (right, "decode", "context", 1024, "-"),
                                            (right, "decode", "context", 8192, "--")]:
        pts = select(kind, m, **{setting: value})
        one_thread = pts[0]["tok_s"]
        ax.plot([r["threads"] for r in pts], [r["tok_s"] / one_thread for r in pts],
                marker="o", ls=style, color=COLOR[m], label=f"{m}, {'512' if value == 512 else str(value // 1024) + 'k'}")
for ax, title in [(left, "Prefill"), (right, "Decode")]:
    ax.plot([1, 6], [1, 6], ":", color="grey", label="perfect scaling")
    ax.set_title(f"{title} thread scaling")
    ax.set_xlabel("Threads (one per P-core)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
left.set_ylabel("Speed-up over 1 thread")
save(fig, "3_thread_scaling.png")

# ---- 4. operator time breakdown -----------------------------------------------------------
ops = read(results / "operator_breakdown.csv")
profiles = ["prefill_4k", "decode_1k", "decode_8k"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
bottom = [0, 0, 0]
for row in ops:
    values = [float(row[p]) for p in profiles]
    ax.bar(["Prefill\n4k prompt", "Decode\n1k context", "Decode\n8k context"], values, bottom=bottom,
           width=0.6, label=row["operator_type"])
    for i, v in enumerate(values):
        if v >= 4:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
    bottom = [b + v for b, v in zip(bottom, values)]
ax.set_ylabel("% of CPU time"); ax.set_ylim(0, 100)
ax.set_title("Operator time breakdown, 1B Q4_K_M, 6 P-cores (VTune)")
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "4_operator_breakdown.png")

# ---- 5. predicted vs measured decode speed ------------------------------------------------
roof = read(results / "roofline.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = range(len(roof))
ax.bar([i - 0.2 for i in x], [float(r["predicted_tok_s"]) for r in roof], width=0.4, color="lightgrey",
       label="Predicted (bandwidth / bytes per token)")
ax.bar([i + 0.2 for i in x], [float(r["measured_tok_s"]) for r in roof], width=0.4,
       color=[COLOR[r["model"]] for r in roof], label="Measured (6 P-cores)")
for i, r in enumerate(roof):
    ax.text(i + 0.2, float(r["measured_tok_s"]) + 1, f"{float(r['measured_pct']):.0f}%", ha="center", fontsize=8)
ax.set_xticks(list(x), [f"{r['model']}\n{int(r['context']) // 1024}k" for r in roof], fontsize=8)
ax.set_ylabel("Decode speed (tokens/s)")
ax.set_title("Predicted vs measured decode speed (% = measured / predicted)")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
save(fig, "5_predicted_vs_measured.png")

# ---- 6. own experiment: thread mask with vs without core 0 -------------------------------
exp_file = results / "exp_pinning" / "summary.csv"
if exp_file.exists():
    exp = read(exp_file)
    groups = [("Llama-3.2-1B-Instruct-Q8_0", "prefill512"), ("Llama-3.2-1B-Instruct-Q8_0", "decode1k"),
              ("Llama-3.2-1B-Instruct-Q4_K_M", "prefill512"), ("Llama-3.2-1B-Instruct-Q4_K_M", "decode1k")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (model, test) in zip(axes, groups):
        for i, threads in enumerate(["2", "5"]):
            for j, case in enumerate(["with_core0", "without_core0"]):
                r = next(e for e in exp if e["model"] == model and e["test"] == test
                         and e["threads"] == threads and e["case"] == case)
                mean, lo, hi = float(r["mean_ts"]), float(r["min_ts"]), float(r["max_ts"])
                ax.bar(i * 3 + j, mean, color="tab:red" if j == 0 else "tab:blue",
                       label=case.replace("_", " ") if i == 0 else None)
                ax.errorbar(i * 3 + j, mean, yerr=[[mean - lo], [hi - mean]], color="black", capsize=3)
        ax.set_xticks([0.5, 3.5], ["2 threads", "5 threads"])
        ax.set_title(f"{model.replace('Llama-3.2-', '').replace('-Instruct-', ' ')}, {test}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("tokens/s (bar = mean, line = min-max of 5)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Own experiment: thread mask with vs without core 0")
    save(fig, "6_experiment_pinning.png")

# ---- 7. own experiment part B: thread scaling with turbo boost on vs off -----------------
boost_file = results / "boost_off" / "capped_scaling.csv"
if boost_file.exists():
    off = {int(r["threads"]): float(r["tok_s"]) for r in read(boost_file) if r["test"] == "prefill512"}
    on = {r["threads"]: r["tok_s"] for r in select("prefill", "1B Q4_K_M", prompt=512)}
    t = sorted(off)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(t, [on[i] / on[1] for i in t], marker="o", label=f"boost on  ({on[6] / on[1]:.2f}x at 6)")
    ax.plot(t, [off[i] / off[1] for i in t], marker="s", label=f"boost off ({off[6] / off[1]:.2f}x at 6)")
    ax.plot(t, t, "k--", alpha=0.5, label="ideal (linear)")
    ax.set_xlabel("Threads (one per P-core)")
    ax.set_ylabel("Speed-up over 1 thread")
    ax.set_title("1B Q4_K_M prefill 512: turbo boost on vs off")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "7_boost_off_scaling.png")
