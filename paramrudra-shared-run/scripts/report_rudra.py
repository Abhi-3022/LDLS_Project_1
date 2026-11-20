#!/usr/bin/env python3
"""REPORT.md for one 60_report.sbatch result folder.

  python3 report_rudra.py results/rudra_<jobid>

Sections, in order: allocation limits, hardware and software, memory bandwidth, shared reference run, llama.cpp's
own timings, per-operator profile (perf), roofline check. Every measured value is "mean ± sample std (min–max, n)".

The roofline check: decoding one token reads every weight once plus the KV cache, so
    predicted tok/s = bandwidth / bytes per token.
It is computed with the bandwidth this same job measured, and the time per token is split into "reading the bytes"
and "everything else", which is the gap the report has to explain.
"""
import csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KV_1B_F16, KV_REL, THEORY_SOCKET, W_1B_Q4, fmt, kv, peak_rss_gb, record, samples, stats  # noqa: E402

# perf symbols -> operator classes, the same grouping as p1-dense/scripts/perf_breakdown.py; first match wins
CLASSES = [
    ("Attention (flash-attn, softmax, KV dequant/dot)", r"dequantize_row_q8_0|vec_dot_q8_0_q8_0|flash_attn|expf|f16_f32|soft_max"),
    ("Q4_K weight matmul", r"q4_K"),
    ("Q6_K weight matmul (output / embedding, some FFN)", r"q6_K|tiled_run_micro"),
    ("Q8_0 matmul", r"q8_0"),
    ("Thread sync (OpenMP spin / barrier)", r"^0x0000|libgomp|gomp|omp_|barrier|sched_yield"),
    ("Norm, SwiGLU, RoPE, add, quantize activations", r"rms_norm|swiglu|rope|add|quantize|silu|mul\b|get_rows|cpy|scale"),
]
LIMIT_ROWS = [("user", "User"), ("account", "Account"), ("qos", "QOS"),
              ("qos_max_tres_per_user", "Max resources per user (QOS MaxTRESPU)"),
              ("qos_maxwall", "Max wall time per job (QOS MaxWall)"),
              ("qos_max_jobs_per_user", "Max running jobs per user"), ("qos_max_submit_per_user", "Max submitted jobs per user"),
              ("partition", "Partition"), ("partition_maxtime", "Partition max time"),
              ("partition_nodes", "Partition nodes"), ("partition_cpus", "Partition CPUs")]


def classify(sym):
    return next((name for name, pat in CLASSES if re.search(pat, sym)), "Other")


def perf_shares(path):
    """perf report --stdio --sort symbol -> ({class: % of samples}, [(%, symbol), ...])"""
    acc, top = {}, []
    for line in open(path):
        m = re.match(r"\s*([\d.]+)%\s+\[\S\]\s+(.+?)\s*$", line)
        if m:
            pct, sym = float(m.group(1)), re.split(r"\s{2,}", m.group(2))[0]   # drop trailing IPC columns
            acc[classify(sym)] = acc.get(classify(sym), 0) + pct
            top.append((pct, sym))
    tot = sum(acc.values()) or 1
    return {k: 100 * v / tot for k, v in acc.items()}, top


def read_csv(path, key, val):
    """{key(row): [val(row), ...]} for a bw/*.csv file, {} if missing"""
    out = {}
    try:
        for r in csv.DictReader(open(path)):
            out.setdefault(key(r), []).append(val(r))
    except OSError:
        pass
    return out


def llama_perf(path):
    """the llama_perf lines of one llama-completion run -> load ms, prefill and decode ms/token and tok/s"""
    t = open(path).read()
    f = lambda pat: float(re.search(pat, t).group(1)) if re.search(pat, t) else None
    return {"load": f(r"load time =\s*([\d.]+) ms"),
            "pp_n": f(r"prompt eval time =.*?/\s*(\d+) tokens"), "pp_tpt": f(r"prompt eval time =.*?([\d.]+) ms per token"),
            "pp_ts": f(r"prompt eval time =.*?([\d.]+) tokens per second"),
            # "eval time" also appears inside "prompt eval time": anchor on the ':' before it
            "tg_n": f(r":\s+eval time =.*?/\s*(\d+) runs"), "tg_tpt": f(r":\s+eval time =.*?([\d.]+) ms per token"),
            "tg_ts": f(r":\s+eval time =.*?([\d.]+) tokens per second"),
            "rss": peak_rss_gb(path.replace(".perf.txt", ".time"))}


def main(d):
    d = d.rstrip("/")
    m = kv(os.path.join(d, "manifest.txt"))
    g = lambda k: m.get(k, "?")
    kvt = g("kv_cache").split()[0]               # manifest: "q8_0 (llama-bench args: ...)"
    T = int(m.get("threads", "4"))
    L = ["# PARAM Rudra: hardware, profile, roofline and the shared reference run\n",
         f"Job {g('job_id')} on node {g('node')}, {g('date_start')}. All runs: {T} threads, one per physical core, pinned to "
         f"socket 0 with memory bound to socket 0 (`{g('placement')}`), llama.cpp CPU backend, KV cache {kvt}. "
         "Every measurement is repeated 3 times: mean ± sample std (min–max, n).\n"]

    # ---- allocation limits (recorded on the login node by 50_limits_login.sh) ----
    lim = kv(os.path.join(d, "limits", "limits_summary.txt"))
    if lim and "date" not in lim:                # older summaries: date and host are in the full record
        full = kv(os.path.join(d, "limits", "limits.txt"))
        lim["date"] = f"{full.get('host', '?')}, {full.get('date', '?')}"
    L.append("## Allocation limits (recorded on the login node)\n")
    if lim:
        L.append("| Limit | Value |\n| --- | --- |")
        L += [f"| {lab} | {lim.get(k, '?')} |" for k, lab in LIMIT_ROWS]
        L.append("\nSo every job gets at most 4 cores of one node for 5 minutes, on a node shared with other jobs. "
                 f"Full `sacctmgr` / `sinfo` / `scontrol` / `sacct` output: `limits/limits.txt` (recorded on {lim['date']}).\n")
    else:
        L.append("Not recorded: run `./50_limits_login.sh` on the login node before submitting the job.\n")

    # ---- hardware, software ----
    L += record(d)

    # ---- memory bandwidth ----
    st = read_csv(os.path.join(d, "bw", "stream.csv"), lambda r: (int(r["threads"]), r["kernel"]), lambda r: float(r["best_MBps"]) / 1000)
    rb = read_csv(os.path.join(d, "bw", "readbw.csv"), lambda r: int(r["threads"]), lambda r: float(r["read_GBps_mean"]))
    if st or rb:
        L.append("\n## Memory bandwidth (GB/s)\n")
        L.append("| Threads | STREAM Copy | STREAM Scale | STREAM Add | STREAM Triad | Read-only probe |\n| --- | --- | --- | --- | --- | --- |")
        for t in sorted({k[0] for k in st} | set(rb)):
            L.append(f"| {t} | " + " | ".join(fmt(stats(st.get((t, k), [])), 1) for k in ("Copy", "Scale", "Add", "Triad"))
                     + f" | {fmt(stats(rb.get(t, [])), 1)} |")
        settings = open(os.path.join(d, "bw", "settings.txt")).read().strip()
        L.append(f"\nSTREAM: standard stream.c ({settings}); each run reports its best of 10 iterations (the STREAM rule). "
                 "Read-only probe: `readbw.c`, AVX-512 loads, mean of passes 2–10. Decode only reads weights, so the probe is "
                 f"the roofline's bandwidth. Theoretical: {THEORY_SOCKET} GB/s per socket (6 × DDR4-2933).\n")

    # ---- shared reference run ----
    ref = {}
    ref_s = samples(os.path.join(d, "reference", f"T{T}.jsonl"))
    if ref_s:
        rss = peak_rss_gb(os.path.join(d, "reference", f"T{T}.time"))
        L.append("## Shared reference run\n")
        L.append("| Test | tok/s | Each repeat | Peak RSS (GB) |\n| --- | --- | --- | --- |")
        for test in ("pp512", "tg128"):
            ref[test] = stats(ref_s.get(test, []))
            L.append(f"| {test} | {fmt(ref[test])} | {' / '.join(str(x) for x in ref_s.get(test, []))} | {rss} |")
        L.append(f"\n`llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -t {T} -p 512 -n 128 -r 3 -ctk {kvt} -ctv {kvt} -fa 1`"
                 " (exact command: `commands.txt`).\n")

    # ---- llama.cpp's own timings ----
    runs = [llama_perf(p) for p in sorted(glob.glob(os.path.join(d, "timings", "run*.perf.txt")))]
    if runs:
        col = lambda k: [r[k] for r in runs if r[k] is not None]
        L.append("## llama.cpp's own timings (llama_perf, llama-completion)\n")
        L.append("| Run | Load (ms) | Prompt eval: tokens | ms/token | tok/s | Decode: tokens | ms/token | tok/s | Peak RSS (GB) |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, r in enumerate(runs, 1):
            L.append(f"| {i} | {r['load']} | {r['pp_n'] and int(r['pp_n'])} | {r['pp_tpt']} | {r['pp_ts']} | "
                     f"{r['tg_n'] and int(r['tg_n'])} | {r['tg_tpt']} | {r['tg_ts']} | {r['rss']} |")
        L.append(f"| mean ± std | {fmt(stats(col('load')), 0)} | | {fmt(stats(col('pp_tpt')))} | {fmt(stats(col('pp_ts')))} | | "
                 f"{fmt(stats(col('tg_tpt')))} | {fmt(stats(col('tg_ts')))} | |")
        L.append("\nSame 512-token prompt (first 2,140 characters of wikitext-2 test, `prompt_512.txt`), 128 tokens decoded "
                 "(`-c 1024 --ignore-eos --temp 0`). Decode here runs at context 512–640; llama-bench's tg128 runs at 0–128. "
                 "Generated text: `timings/run*.out`.\n")

    # ---- per-operator profile ----
    L.append("## Per-operator time breakdown (perf)\n")
    nr = os.path.join(d, "perf", "NOT_RUN.txt")
    if os.path.exists(nr):
        L.append(f"Not run: {open(nr).read().strip()}. llama.cpp's own timings above still split prefill from decode.\n")
    else:
        sh, tops = {"pp": [], "tg": []}, {"pp": {}, "tg": {}}
        for ph in sh:
            for p in sorted(glob.glob(os.path.join(d, "perf", f"{ph}_r*.report.txt"))):
                s, top = perf_shares(p)
                if s:
                    sh[ph].append(s)
                    for pct, sym in top:
                        tops[ph].setdefault(sym, []).append(pct)
        if sh["pp"] or sh["tg"]:
            cls = sorted({c for ph in sh for s in sh[ph] for c in s}, key=lambda c: -sum(s.get(c, 0) for s in sh["tg"]))
            L.append("| Operator class | Prefill pp512 (% of samples) | Decode tg128 (% of samples) |\n| --- | --- | --- |")
            for c in cls:
                L.append(f"| {c} | {fmt(stats([s.get(c, 0) for s in sh['pp']]), 1)} | {fmt(stats([s.get(c, 0) for s in sh['tg']]), 1)} |")
            for ph, name in (("pp", "prefill"), ("tg", "decode")):
                n = len(sh[ph]) or 1
                best = sorted(tops[ph].items(), key=lambda kv_: -sum(kv_[1]) / n)[:3]
                L.append(f"\nTop 3 symbols, {name}: " + "; ".join(f"`{s}` {sum(v) / n:.1f}%" for s, v in best) + ".")
            L.append("\n`perf record -e cycles:u -F 1999 -D <load + 500 ms>` on `llama-bench -r 1 --no-warmup` (pp512, or tg128), "
                     "3 runs each; sampling starts after model load and weight repacking. Raw reports: `perf/*.report.txt`.\n")
        else:
            L.append("perf ran but produced no samples: see `perf/*.log`.\n")

    # ---- roofline check ----
    tg = ref.get("tg128")
    if tg:
        kv_bytes = 64 * KV_1B_F16 * KV_REL.get(kvt, 1.0)      # tg128 runs at context 0..128: mean 64 tokens
        b = W_1B_Q4 + kv_bytes
        L.append("## Roofline check (decode, 1B Q4_K_M)\n")
        L.append(f"Bytes read per decoded token = {W_1B_Q4:,} B of weights (GGUF tensor sizes) + the KV cache at tg128's mean "
                 f"context of 64 tokens ({64 * KV_1B_F16:,} B × {KV_REL.get(kvt, 1.0):.3f} for {kvt} = {kv_bytes / 1e6:.1f} MB) "
                 f"= {b / 1e9:.4f} GB.\n")
        L.append("| Bandwidth used for the prediction | GB/s | Predicted tg128 (tok/s) | Measured tg128 (tok/s) | Measured / predicted |\n"
                 "| --- | --- | --- | --- | --- |")
        cand = []
        if rb.get(T):
            cand.append((f"Read-only probe, {T} threads (this job)", stats(rb[T])[0]))
        cand += [(f"STREAM {k}, {T} threads", stats(st[(T, k)])[0]) for k in ("Copy", "Triad") if st.get((T, k))]
        if rb.get(1):
            cand.append((f"Read-only probe, 1 thread × {T} (if bandwidth grew linearly with cores)", stats(rb[1])[0] * T))
        cand.append(("Theoretical, one socket", THEORY_SOCKET))
        for name, bw in cand:
            pred = bw * 1e9 / b
            L.append(f"| {name} | {bw:.1f} | {pred:.1f} | {tg[0]:.2f} ± {tg[1]:.2f} | {tg[0] / pred:.2f} |")
        ach = tg[0] * b / 1e9
        L.append(f"\nDecode achieves {ach:.1f} GB/s (measured tok/s × bytes per token)"
                 + (f": {100 * ach / stats(rb[T])[0]:.0f}% of what {T} cores read in the probe" if rb.get(T) else "")
                 + f" and {100 * ach / THEORY_SOCKET:.0f}% of the socket's theoretical bandwidth.")
        if rb.get(T):   # where the time per token goes
            ms_tok, ms_read = 1000 / tg[0], 1000 * b / (stats(rb[T])[0] * 1e9)
            L.append(f"\nTime per token: {ms_tok:.1f} ms measured, of which {ms_read:.1f} ms is reading {b / 1e9:.3f} GB at the probe's "
                     f"{stats(rb[T])[0]:.1f} GB/s; the other {ms_tok - ms_read:.1f} ms is the gap to explain (compute, thread "
                     "synchronisation, small operators). The per-operator profile is what splits it.\n")
        else:
            L.append("")

    L.append("## Files\n")
    L.append("`manifest.txt` (software, build, model checksum, allocation), `node/` (lscpu, caches, NUMA, memory), `bw/` "
             "(STREAM, readbw), `reference/`, `timings/`, `perf/`, `commands.txt` (every command with its exit code and time), `limits/`.")
    with open(os.path.join(d, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"report written: {os.path.join(d, 'REPORT.md')}")


if __name__ == "__main__":
    for a in sys.argv[1:] or sorted(glob.glob("results/rudra_*"))[-1:]:
        main(a)
