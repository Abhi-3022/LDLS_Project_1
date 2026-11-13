#!/usr/bin/env python3
"""Own experiment: change the number of routed experts per token (k = expert_used_count).

Granite-3.1 MoE routes every token to k = 8 of E experts (E = 32 for 1B-A400M, 40 for
3B-A800M). Re-writing k in a copy of the GGUF changes the *active* parameters while the
*total* parameters (the file) stay the same. If batch-1 decode is bound by the bytes it
reads, decode time should be linear in k with slope = one expert's bytes per layer, and
the k = E point should read as many bytes as the dense model of the same size.
Perplexity on the fixed WikiText-2 sample shows the quality cost of each k.

Steps:
  1. models/experiment/<model>-k<k>.gguf  (copy + in-place edit of <arch>.expert_used_count)
  2. bench_sweep.py experts  -> results/csv/bench_experts.csv   (pp512, tg128 @0, tg128 @1k; 3 reps)
  3. llama-perplexity per k  -> results/csv/experts_ppl.csv
usage: experiment_experts.py [--models granite1b-q4,granite3b-q4] [--ks 1,2,4,8,16,32]
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess

from gguf import GGUFReader

PROJ = pathlib.Path(__file__).resolve().parents[1]
ENV = os.environ
MODELS = pathlib.Path(ENV.get("MODELS", PROJ / "models"))
RESULTS = pathlib.Path(ENV.get("RESULTS", PROJ / "results"))
# build-exp = benchmark build + patches/llama_graph_nodes_experiment.patch: llama.cpp sizes the graph
# node table for the trained k, so k > 8 overflows it; the patch adds LLAMA_GRAPH_NODES_EXTRA nodes.
# Every k (including k <= 8) runs on this one binary for consistency.
EXP_BIN = PROJ / "third_party/llama.cpp/build-exp/bin"
EXP_ENV = {"LLAMA_BIN": str(EXP_BIN), "LLAMA_GRAPH_NODES_EXTRA": "4096"}
CORES = ENV.get("CORES", "0 " + " ".join(map(str, range(2, 56)))).split()
NODE = ENV.get("NODE", "0")
T_MAIN = int(ENV.get("T_MAIN", 32))
EXP = MODELS / "experiment"


def table():
    rows = {}
    for line in ENV.get("MODEL_TABLE", "").strip().splitlines():
        name, f, ctx = line.strip().split("|")
        rows[name] = (f, int(ctx))
    return rows


def set_expert_used(src, dst, k):
    if not dst.exists():
        shutil.copyfile(src, dst)
    r = GGUFReader(str(dst), "r+")
    key = next(k_ for k_ in r.fields if k_.endswith(".expert_used_count"))
    n_exp = next(int(f.parts[f.data[0]][0]) for k_, f in r.fields.items() if k_.endswith(".expert_count"))
    if k > n_exp:
        raise SystemExit(f"k={k} > expert_count={n_exp}")
    f = r.fields[key]
    f.parts[f.data[0]][0] = k       # memory-mapped: writes through to the file
    del r
    return n_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="granite1b-q4,granite3b-q4")
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    ap.add_argument("--reps", default=ENV.get("REPS", "3"))
    a = ap.parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    tab = table()
    lines = []
    for name in a.models.split(","):
        f, ctx = tab[name]
        n_exp = None
        for k in [int(x) for x in a.ks.split(",")]:
            dst = EXP / f"{name}-k{k}.gguf"
            n_exp = set_expert_used(MODELS / f, dst, k)
            lines.append(f"{name}-k{k}|experiment/{dst.name}|{ctx}")
        if n_exp not in [int(x) for x in a.ks.split(",")]:  # always include k = E (all experts)
            dst = EXP / f"{name}-k{n_exp}.gguf"
            set_expert_used(MODELS / f, dst, n_exp)
            lines.append(f"{name}-k{n_exp}|experiment/{dst.name}|{ctx}")
    mt = EXP / "model_table.txt"
    mt.write_text("\n".join(lines) + "\n")
    print(mt.read_text())

    subprocess.run([str(PROJ / ".venv/bin/python"), str(PROJ / "scripts/bench_sweep.py"), "experts",
                    "--model-table", str(mt), "--reps", a.reps], check=True, env=dict(ENV, **EXP_ENV))

    text = PROJ / "data" / "wikitext2_test_100k.txt"
    out = RESULTS / "csv" / "experts_ppl.csv"
    done = set(l.split(",")[0] for l in open(out).read().splitlines()[1:]) if out.exists() else set()
    with open(out, "a") as fo:
        if not done:
            fo.write("model,k,ppl,ppl_err\n")
        for line in lines:
            name, rel, _ = line.split("|")
            if name in done:
                continue
            env = dict(ENV, OMP_NUM_THREADS=str(T_MAIN), OMP_PROC_BIND="close", OMP_PLACES="cores", **EXP_ENV)
            res = subprocess.run(["numactl", f"--physcpubind={','.join(CORES[:T_MAIN])}", f"--membind={NODE}",
                                  str(EXP_BIN / "llama-perplexity"), "-m", str(MODELS / rel), "-f", str(text),
                                  "-c", "512", "-b", "512", "-t", str(T_MAIN), "-lm", "none"],
                                 capture_output=True, text=True, env=env)
            log = res.stdout + res.stderr
            (RESULTS / "raw" / "perplexity").mkdir(parents=True, exist_ok=True)
            (RESULTS / "raw" / "perplexity" / f"{name}.log").write_text(log)
            ppl = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log)
            k = name.rsplit("-k", 1)[1]
            fo.write(f"{name},{k},{ppl.group(1) if ppl else ''},{ppl.group(2) if ppl else ''}\n")
            fo.flush()
            print(name, "PPL", ppl.group(1) if ppl else "FAILED", flush=True)


if __name__ == "__main__":
    main()
