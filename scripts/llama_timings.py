#!/usr/bin/env python3
"""llama.cpp's own timings (llama_perf) next to llama-bench, same model, threads and pinning.

The brief asks for "llama.cpp's own timings and llama-bench". llama-bench measures the forward
pass only; an end-to-end llama-completion run also pays for tokenisation, sampling (greedy argmax
over the full vocabulary, every token) and the per-token loop in common/. This step runs, per
model and repetition, as separate pinned processes:

  llama_perf   llama-completion on an exact 512-token WikiText-2 prompt (BOS included), then 128
               greedy tokens with --ignore-eos; parses llama_perf_context_print ("prompt eval time",
               "eval time") and llama_perf_sampler_print ("sampling time") from stderr
  llama-bench  pp512 and tg128 at depth 512, i.e. the same KV-cache range as the completion run

Output: results/csv/llama_timings.csv (one row per source x phase x rep, with peak RSS),
raw logs in results/raw/timings/. Rows already present are skipped (resumable).

usage: llama_timings.py [--models llama1b-q4,...] [--threads 32] [--reps 3] [--out csv] [--raw dir]
"""
import argparse
import csv
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench_sweep as bs  # noqa: E402  (same placement, model table, budget and RSS accounting)

PROJ = bs.PROJ
TEXT = PROJ / "data" / "wikitext2_test_100k.txt"
N_PROMPT, N_GEN = 512, 128
FIELDS = ["model", "quant", "threads", "rep", "source", "phase", "n_tokens", "ms", "ms_per_token", "tok_s",
          "maxrss_gb", "over_budget", "majflt", "loadavg_1m", "mem_avail_gb", "timestamp"]

PERF = {  # phase -> regex on the perf summary (common_perf_print at the pinned commit, llama_perf_* before)
    "prefill": re.compile(r"prompt eval time\s*=\s*([\d.]+) ms /\s*(\d+) tokens"),
    "decode": re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs"),
    # one sample per generated token; older builds print "/ N runs", the pinned one prints no count
    "sampling": re.compile(r"sampling time\s*=\s*([\d.]+) ms(?: /\s*(\d+) runs)?"),
    "load": re.compile(r"load time\s*=\s*([\d.]+) ms"),
    "total": re.compile(r"total time\s*=\s*([\d.]+) ms /\s*(\d+) tokens"),
}


def n_tokens(model_file, text):
    """token count incl. BOS, with the model's own tokenizer (vocab only, fast)"""
    r = subprocess.run([str(bs.LLAMA_BIN / "llama-tokenize"), "-m", str(model_file), "--stdin", "--ids",
                        "--log-disable"], input=text, capture_output=True, text=True, check=True)
    return len(re.findall(r"-?\d+", r.stdout.strip().splitlines()[-1]))


def exact_prompt(model, raw):
    """longest prefix of the WikiText sample (in words) that tokenises to exactly N_PROMPT tokens"""
    path = raw / f"prompt_{model['name']}.txt"
    if path.exists():
        return path
    words = TEXT.read_text().split(" ")
    lo, hi = 1, 4 * N_PROMPT
    while lo < hi:  # largest word count with <= N_PROMPT tokens
        mid = (lo + hi + 1) // 2
        if n_tokens(bs.MODELS / model["file"], " ".join(words[:mid])) <= N_PROMPT:
            lo = mid
        else:
            hi = mid - 1
    text = " ".join(words[:lo])
    got = n_tokens(bs.MODELS / model["file"], text)
    if got != N_PROMPT:  # a word that splits into several tokens can straddle the boundary
        print(f"note: {model['name']} prompt is {got} tokens (closest word boundary to {N_PROMPT})", flush=True)
    path.write_text(text)
    return path


def run_completion(model, prompt, t, log):
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        cmd = (["/usr/bin/time", "-f", "%M %F", "-o", tf.name] + bs.pin_prefix(t) +
               [str(bs.LLAMA_BIN / "llama-completion"), "-m", str(bs.MODELS / model["file"]), "-f", str(prompt),
                "-n", str(N_GEN), "--ignore-eos", "--temp", "0", "-s", "42", "-no-cnv", "--no-display-prompt",
                "-c", str(N_PROMPT + N_GEN), "-b", str(N_PROMPT), "-ub", str(N_PROMPT), "-t", str(t),
                "-lm", "none", "--perf"])
        la = os.getloadavg()[0]
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        log.write_text(res.stderr + "\n=== stdout ===\n" + res.stdout)
        if res.returncode != 0:
            sys.stderr.write(res.stderr[-2000:])
            raise RuntimeError(f"llama-completion failed: {' '.join(cmd)}")
        rss_kb, majflt = tf.read().split()[-2:]
    rss_gb = int(rss_kb) * 1024 / 1e9
    common = dict(maxrss_gb=round(rss_gb, 4), over_budget=int(rss_gb > bs.BUDGET_GB), majflt=int(majflt),
                  loadavg_1m=la, mem_avail_gb=round(bs.mem_avail_gb(), 1),
                  timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rows = []
    for phase, rx in PERF.items():
        m = rx.search(res.stderr)
        if not m:
            raise RuntimeError(f"no '{phase}' llama_perf line in {log}")
        ms = float(m.group(1))
        n = int(m.group(2)) if rx.groups >= 2 and m.group(2) else (N_GEN if phase == "sampling" else "")
        per = ms / n if n else ""
        rows.append(dict(source="llama_perf", phase=phase, n_tokens=n, ms=round(ms, 3),
                         ms_per_token=round(per, 4) if n else "",
                         tok_s=round(1000 / per, 3) if n and per else "", **common))
    return rows


def run_bench(model, t):
    rows = []
    for phase, test, p, n, d in (("prefill", "pp", N_PROMPT, 0, 0), ("decode", "tg", 0, N_GEN, N_PROMPT)):
        r = bs.run_one(model, test, p, n, d, t, [])
        ntok = p or n
        ms = r["avg_ns"] / 1e6
        rows.append(dict(source="llama-bench", phase=phase, n_tokens=ntok, ms=round(ms, 3),
                         ms_per_token=round(ms / ntok, 4), tok_s=r["tok_s"], maxrss_gb=r["maxrss_gb"],
                         over_budget=r["over_budget"], majflt=r["majflt"], loadavg_1m=r["loadavg_1m"],
                         mem_avail_gb=r["mem_avail_gb"], timestamp=r["timestamp"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--threads", default=os.environ.get("T_MAIN", "32"))
    ap.add_argument("--reps", type=int, default=int(os.environ.get("REPS", 3)))
    ap.add_argument("--out", default=str(bs.RESULTS / "csv" / "llama_timings.csv"))
    ap.add_argument("--raw", default=str(bs.RESULTS / "raw" / "timings"))
    a = ap.parse_args()

    table = bs.model_table()
    models = [table[x] for x in (a.models.split(",") if a.models else table)]
    threads = [int(x) for x in a.threads.replace(",", " ").split()]
    out, raw = pathlib.Path(a.out), pathlib.Path(a.raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        done = {(r["model"], r["threads"], r["rep"], r["source"]) for r in csv.DictReader(open(out))}
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    for m in models:
        prompt = exact_prompt(m, raw)
        for t in threads:
            for rep in range(1, a.reps + 1):
                for source in ("llama_perf", "llama-bench"):
                    if (m["name"], str(t), str(rep), source) in done:
                        continue
                    rows = (run_completion(m, prompt, t, raw / f"{m['name']}__t{t}__r{rep}.log")
                            if source == "llama_perf" else run_bench(m, t))
                    for r in rows:
                        r.update(model=m["name"], quant=m["quant"], threads=t, rep=rep)
                        w.writerow(r)
                    f.flush()
                    s = {r["phase"]: r for r in rows}
                    print(f"{m['name']:13s} t={t:<2d} rep={rep} {source:11s} prefill {s['prefill']['tok_s']:>9} tok/s  "
                          f"decode {s['decode']['tok_s']:>8} tok/s"
                          + (f"  sampling {s['sampling']['ms_per_token']} ms/tok" if "sampling" in s else "")
                          + f"  rss={rows[0]['maxrss_gb']:.2f} GB", flush=True)
    f.close()


if __name__ == "__main__":
    main()
