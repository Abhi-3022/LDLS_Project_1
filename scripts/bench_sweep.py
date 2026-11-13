#!/usr/bin/env python3
"""Scripted llama-bench sweeps. Every repetition is its own process, pinned with numactl
and wrapped in /usr/bin/time, so each one gets its own peak RSS.

Suites (results/csv/bench_<suite>.csv):
  threads   pp512 and tg128 for every model, threads in $THREAD_LIST
  lengths   prefill at 512 / 4096 / max prompt, decode (tg128) at context 1k / 8k / max, $T_MAIN threads
  reference Llama-3.2-1B Q4_K_M, 512-token prompt + 128-token decode (the shared PARAM Rudra test)
  experts   own experiment: pp512, tg128 and tg128 at context 1k (used with --model-table for the
            expert-count variants built by experiment_experts.py)
  ik        pp512 and tg128, for the llama.cpp vs ik_llama.cpp comparison (--engine ik)

Context caps follow the project rules: 1B-class models up to 16k, 3B-class up to 8k.
Decode "at context C" is tg128 starting at depth C-128, so the KV cache never exceeds C.
Rows already present in the CSV are skipped, so an interrupted sweep can be resumed.

usage: bench_sweep.py <suite> [--models llama1b-q4,...] [--threads 1,2,..] [--reps 3] [--extra "-fa off"] [--tag x]
                      [--engine llama|ik] [--model-table file]
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

PROJ = pathlib.Path(__file__).resolve().parents[1]
ENV = os.environ
LLAMA_BIN = pathlib.Path(ENV.get("LLAMA_BIN", PROJ / "third_party/llama.cpp/build-cpu/bin"))
IK_BIN = pathlib.Path(ENV.get("IK_BIN", PROJ / "third_party/ik_llama.cpp/build-cpu/bin"))
MODELS = pathlib.Path(ENV.get("MODELS", PROJ / "models"))
RESULTS = pathlib.Path(ENV.get("RESULTS", PROJ / "results"))
CORES = ENV.get("CORES", "0 " + " ".join(map(str, range(2, 56)))).split()
NODE = ENV.get("NODE", "0")
BUDGET_GB = float(ENV.get("MEM_BUDGET_GB", 4.5))

FIELDS = ["suite", "tag", "engine", "model", "quant", "file", "test", "n_prompt", "n_gen", "n_depth", "context",
          "threads", "rep", "tok_s", "avg_ns", "maxrss_gb", "over_budget", "majflt", "swap_in", "swap_out",
          "wall_s", "loadavg_1m", "mem_avail_gb", "extra", "build_commit", "timestamp"]


def model_table(path=None):
    rows = {}
    text = open(path).read() if path else ENV.get("MODEL_TABLE", "")
    for line in text.strip().splitlines():
        name, f, ctx = line.strip().split("|")
        rows[name] = dict(name=name, file=f, max_ctx=int(ctx), quant=name.split("-")[1])
    return rows


def pin_prefix(n):
    cores = ",".join(CORES[:n])
    return ["env", f"OMP_NUM_THREADS={n}", "OMP_PROC_BIND=close", "OMP_PLACES=cores",
            "numactl", f"--physcpubind={cores}", f"--membind={NODE}"]


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in ("pswpin", "pswpout"):
            d[k] = int(v)
    return d


def mem_avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return float("nan")


def run_one(model, test, p, n, d, t, extra, engine="llama"):
    # both engines load the weights into anonymous memory (no mmap) so RSS is not double counted
    binary, load = (LLAMA_BIN, ["-lm", "none"]) if engine == "llama" else (IK_BIN, ["-mmp", "0"])
    depth = ["-d", str(d)] if d or engine == "llama" else []
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        cmd = (["/usr/bin/time", "-f", "%M %F %e", "-o", tf.name] + pin_prefix(t) +
               [str(binary / "llama-bench"), "-m", str(MODELS / model["file"]), "-p", str(p), "-n", str(n)] +
               depth + ["-t", str(t), "-r", "1"] + load + ["-o", "json"] + extra)
        v0, la = vmstat(), os.getloadavg()[0]
        res = subprocess.run(cmd, capture_output=True, text=True)
        v1 = vmstat()
        if res.returncode != 0:
            sys.stderr.write(res.stderr[-2000:])
            raise RuntimeError(f"llama-bench failed: {' '.join(cmd)}")
        rss_kb, majflt, wall = tf.read().split()[-3:]
    j = json.loads(res.stdout)[0]
    rss_gb = int(rss_kb) * 1024 / 1e9
    return dict(test=test, n_prompt=p, n_gen=n, n_depth=d, context=p if test == "pp" else d + n, threads=t,
                tok_s=round(j["avg_ts"], 4), avg_ns=j["avg_ns"], maxrss_gb=round(rss_gb, 4),
                over_budget=int(rss_gb > BUDGET_GB), majflt=int(majflt),
                swap_in=v1["pswpin"] - v0["pswpin"], swap_out=v1["pswpout"] - v0["pswpout"],
                wall_s=float(wall), loadavg_1m=la, mem_avail_gb=round(mem_avail_gb(), 1),
                build_commit=j.get("build_commit", ""), timestamp=j.get("test_time", ""))


def plan(suite, models, threads, t_main):
    """yield (model, test, n_prompt, n_gen, n_depth, threads)"""
    for m in models:
        if suite == "threads":
            for t in threads:
                yield m, "pp", 512, 0, 0, t
                yield m, "tg", 0, 128, 0, t
        elif suite == "lengths":
            for p in sorted({512, 4096, m["max_ctx"]}):
                yield m, "pp", p, 0, 0, t_main
            for c in sorted({1024, 8192, m["max_ctx"]}):
                yield m, "tg", 0, 128, c - 128, t_main
        elif suite == "reference":
            # 512-token prompt, then 128 tokens decoded on top of it
            for t in threads:
                yield m, "pp", 512, 0, 0, t
                yield m, "tg", 0, 128, 512, t
        elif suite == "experts":
            yield m, "pp", 512, 0, 0, t_main
            yield m, "tg", 0, 128, 0, t_main
            yield m, "tg", 0, 128, 1024 - 128, t_main
        elif suite == "ik":
            yield m, "pp", 512, 0, 0, t_main
            yield m, "tg", 0, 128, 0, t_main
        else:
            raise SystemExit(f"unknown suite {suite}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--models", default=None)
    ap.add_argument("--threads", default=None)
    ap.add_argument("--reps", type=int, default=int(ENV.get("REPS", 3)))
    ap.add_argument("--extra", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="llama", choices=["llama", "ik"])
    ap.add_argument("--model-table", default=None)
    a = ap.parse_args()

    table = model_table(a.model_table)
    names = a.models.split(",") if a.models else list(table)
    if a.suite == "reference" and not a.models:
        names = ["llama1b-q4"]
    models = [table[x] for x in names]
    threads = [int(x) for x in (a.threads or ENV.get("THREAD_LIST", "1 2 4 8 16 32")).replace(",", " ").split()]
    t_main = int(ENV.get("T_MAIN", 32))
    extra = a.extra.split()
    out = pathlib.Path(a.out) if a.out else RESULTS / "csv" / f"bench_{a.suite}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            done.add((r.get("engine") or "llama", r["tag"], r["model"], r["test"], r["n_prompt"], r["n_gen"], r["n_depth"], r["threads"], r["rep"]))
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    jobs = list(plan(a.suite, models, threads, t_main))
    print(f"[{a.suite}] {len(jobs)} configs x {a.reps} reps -> {out}", flush=True)
    for (m, test, p, n, d, t) in jobs:
        for rep in range(1, a.reps + 1):
            key = (a.engine, a.tag, m["name"], test, str(p), str(n), str(d), str(t), str(rep))
            if key in done:
                continue
            r = run_one(m, test, p, n, d, t, extra, a.engine)
            r.update(suite=a.suite, tag=a.tag, engine=a.engine, model=m["name"], quant=m["quant"], file=m["file"], rep=rep,
                     extra=a.extra, test=test)
            w.writerow(r)
            f.flush()
            flag = " OVER-BUDGET" if r["over_budget"] else ""
            flag += " SWAP" if r["swap_in"] or r["swap_out"] else ""
            print(f"{a.engine:5s} {m['name']:13s} {test} p={p:<5d} n={n:<3d} d={d:<5d} t={t:<2d} rep={rep}  "
                  f"{r['tok_s']:9.2f} tok/s  rss={r['maxrss_gb']:.2f} GB  load={r['loadavg_1m']:.1f}{flag}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
