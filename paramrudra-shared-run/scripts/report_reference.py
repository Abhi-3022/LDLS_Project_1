#!/usr/bin/env python3
"""REPORT.md and reference_summary.csv for one 10_reference.sbatch result folder.

  python3 report_reference.py results/ref_<jobid>

Reads reference/T<threads>_<placement>.{csv,jsonl,time}: llama-bench's mean/std, every repeat, peak RSS.
Writes the headline (most threads), the thread-scaling table, and the hardware/software record.
"""
import csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W_1B_Q4, kv, peak_rss_gb, record, samples  # noqa: E402

HEADER = ["test", "threads", "placement", "kv_type", "tok_s_mean", "tok_s_std", "tok_s_min", "tok_s_max",
          "per_repeat_tok_s", "peak_rss_GB", "source"]


def collect(d):
    """one row per (test, thread count) from llama-bench's CSV, with the per-repeat values from its JSONL"""
    out = []
    for p in glob.glob(os.path.join(d, "reference", "T*.csv")):
        tag = os.path.basename(p)[:-4]                      # e.g. T4_socket
        smp = samples(p[:-4] + ".jsonl")
        try:                                                # older folders copied some runs in from another job
            src = open(p[:-4] + ".source").read().strip()
        except OSError:
            src = "this job"
        for r in csv.DictReader(open(p)):
            test = f"tg{r['n_gen']}" if int(r["n_gen"]) else f"pp{r['n_prompt']}"
            sm = smp.get(test, [])
            out.append([test, int(r["n_threads"]), tag.split("_", 1)[1], r["type_k"], round(float(r["avg_ts"]), 2),
                        round(float(r["stddev_ts"]), 2), min(sm) if sm else "", max(sm) if sm else "",
                        " ".join(str(x) for x in sm), peak_rss_gb(p[:-4] + ".time"), src])
    return sorted(out, key=lambda x: (x[0], x[1]))


def main(d):
    d = d.rstrip("/")
    rows = collect(d)
    if not rows:
        print(f"{d}: no reference/T*.csv yet"); return
    with open(os.path.join(d, "reference_summary.csv"), "w", newline="") as f:
        csv.writer(f).writerows([HEADER] + rows)
    m = kv(os.path.join(d, "manifest.txt"))
    by_t = {}
    for r in rows:
        by_t.setdefault(r[1], {})[r[0]] = r
    top = max(by_t)
    L = ["# Shared reference run on PARAM Rudra\n",
         "Test (fixed by the course): Llama-3.2-1B-Instruct Q4_K_M, llama-bench pp512 (512-token prompt) and tg128 "
         f"(128 decoded tokens), {m.get('repeats', '3')} repeats, CPU only. KV cache: {m.get('kv_cache', '?')}.\n",
         f"## Headline ({top} threads)\n",
         "| Test | tok/s (mean ± std) | min–max | Each repeat | Peak RSS (GB) | Run |\n| --- | --- | --- | --- | --- | --- |"]
    for test in ("pp512", "tg128"):
        r = by_t[top].get(test)
        if r:
            L.append(f"| {test} | {r[4]} ± {r[5]} | {r[6]}–{r[7]} | {r[8].replace(' ', ' / ')} | {r[9]} | {r[10]} |")
    if len(by_t) > 1:
        t1 = min(by_t)
        L += [f"\n## Thread scaling\n",
              f"| Threads | Prefill pp512 (tok/s) | Speed-up vs {t1} | Decode tg128 (tok/s) | Speed-up vs {t1} | Decode reads (GB/s) | Peak RSS (GB) | Run |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t in sorted(by_t):
            pp, tg = by_t[t].get("pp512"), by_t[t].get("tg128")
            val = lambda r: f"{r[4]} ± {r[5]}" if r else "?"
            up = lambda r, k: f"{r[4] / by_t[t1][k][4]:.2f}×" if r and by_t[t1].get(k) else "?"
            L.append(f"| {t} | {val(pp)} | {up(pp, 'pp512')} | {val(tg)} | {up(tg, 'tg128')} | "
                     f"{tg[4] * W_1B_Q4 / 1e9:.1f} | {(pp or tg)[9]} | {(pp or tg)[10]} |" if tg else f"| {t} | {val(pp)} | {up(pp, 'pp512')} | ? | ? | ? | | |")
        L.append("\nDecode reads = tg128 tok/s × 0.80 GB of weights per token (the KV cache adds under 0.2% at this context).")
    L.append("\nPlacement `socket`: one thread per physical core on socket 0 (`taskset`, `OMP_PROC_BIND=close OMP_PLACES=cores`), "
             "memory bound to socket 0 (`numactl --membind`).\n")
    L += record(d)
    L.append("\nExact command lines with exit codes and times: `commands.txt`. Raw data: `reference/*.csv` (mean/std), "
             "`reference/*.jsonl` (every repeat), `reference/*.time` (peak RSS), `node/`, `manifest.txt`.")
    with open(os.path.join(d, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"report written: {os.path.join(d, 'REPORT.md')}")


if __name__ == "__main__":
    for a in sys.argv[1:] or sorted(glob.glob("results/ref_*"))[-1:]:
        main(a)
