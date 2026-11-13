# CPU inference project: dense vs. small MoE on a CPU (options 1 + 6) and a Mamba-3 decode step (bonus)

This repository runs Llama-3.2-1B/3B (option 1, the dense baseline) and Granite-3.1
1B-A400M / 3B-A800M (option 6, small mixture of experts) on a CPU with llama.cpp, measures
speed and memory, explains the speed with a bandwidth roofline and a per-operator profile,
runs one experiment of our own (number of routed experts), compares against ik_llama.cpp,
and implements and verifies a Mamba-3 SISO and MIMO decode step in C++ with SIMD (bonus).

Everything is scripted. Raw results are CSV under `results/csv`, figures under `plots/`,
and every key number with its spread is collected in `results/summary.md`.

**Deliverables:** `report/report.pdf` (4 pages; source `report/report.tex`, built with tectonic) ·
`slides/talk.pptx` (12 slides with speaker notes, built by `slides/make_slides.py` from the CSVs) ·
`bonus_mamba3/` (kernels, verification, `results/summary.md` with the MIMO answer).

## Quick start

```bash
bash scripts/run_all.sh          # build -> hardware -> bandwidth -> models -> roofline -> correctness
                                 # -> sweeps -> profile -> own experiment -> ik_llama.cpp -> reference
                                 # -> bonus -> plots + summary
bash scripts/run_all.sh lengths  # or any single step: build hw stream models roofline correct threads
                                 # lengths profile experts ik reference bonus plots
```

Requirements: Linux x86-64, gcc >= 9, git, `numactl`, `/usr/bin/time`, `uv` (or any Python 3.11 venv
with `requirements.txt`; torch is the CPU wheel). CMake >= 3.18 and Ninja come from the venv.
About 25 GB disk (F16 sources, quantized models, expert-count variants). No GPU is used anywhere.

Settings live in `scripts/env.sh` (overridable from the environment): socket `NODE`, core list
`CORES`, `THREAD_LIST`, `T_MAIN`, `REPS`, and the model table.

## What runs where

| Step | Script | Output |
|---|---|---|
| Build llama.cpp (bench + profiling build), ik_llama.cpp, bonus kernels | `scripts/01_build.sh` | `third_party/*/build-*` |
| Hardware record | `scripts/00_hwinfo.sh` | `hw/hardware.txt`, `hw/hardware.json` |
| Memory bandwidth: STREAM + read-only reduction, 1..55 threads | `scripts/02_stream.sh`, `tools/readbw.c` | `results/csv/stream.csv` |
| Models: F16 GGUF download, local quantization (Q4_K_M; Q8_0 for <= 1.5B) | `scripts/03_get_models.sh` | `models/*.gguf`, `models/SHA256SUMS` |
| Roofline prediction (before measuring) | `scripts/roofline.py` | `results/csv/model_bytes.csv`, `roofline_pred.csv` |
| Correctness: perplexity + sample generations | `scripts/04_correctness.sh`, `scripts/get_data.py` | `results/csv/perplexity.csv`, `results/raw/samples/` |
| Sweeps: thread scaling; prompt/context length | `scripts/bench_sweep.py threads|lengths` | `results/csv/bench_threads.csv`, `bench_lengths.csv` |
| Per-operator profile (prefill + decode, several contexts) | `scripts/profile_ops.py` | `results/csv/opprof_*.csv`, `results/raw/opprof/` |
| Own experiment: experts per token k (runs on `build-exp`, see below) | `scripts/experiment_experts.py` | `results/csv/bench_experts.csv`, `experts_ppl.csv` |
| Stretch: llama.cpp vs ik_llama.cpp | `scripts/ik_compare.sh` | `results/csv/bench_ik.csv` |
| Shared reference test (this machine / PARAM Rudra) | `scripts/reference_run.sh`, `scripts/rudra_reference.sbatch` | `results/csv/reference_<host>.csv` |
| Bonus: Mamba-3 decode step | `bonus_mamba3/run_bonus.sh` | `bonus_mamba3/results/` |
| Figures and summary of all numbers | `scripts/plots.py`, `scripts/summarize.py` | `plots/*.png`, `results/summary.md` |
| Report / slides | `report/report.tex` (tectonic), `slides/make_slides.py` | `report/report.pdf`, `slides/talk.pptx` |

## Pinned versions

| Component | Version |
|---|---|
| llama.cpp | `e30ad80fd7fa09a858bf8b8f4118e6479809486e` (+ `patches/ggml_op_profile.patch`, compiled only into `build-prof`; + `patches/llama_graph_nodes_experiment.patch`, compiled only into `build-exp`) |
| ik_llama.cpp | `656036d98503e154952f6c633cb3febc1a4f08b4` (+ `patches/ik_compat/` GCC 9 shims) |
| state-spaces/mamba | `e9594ce1c732d97440f0332fdc43170a2294dbfa` (reference code only, run on CPU) |
| STREAM | `stream.c` 5.10 from cs.virginia.edu |
| Models | bartowski F16 GGUFs of Llama-3.2-{1B,3B}-Instruct and granite-3.1-{1b-a400m,3b-a800m}-instruct, quantized here with `llama-quantize` (no imatrix) |

## Measurement protocol

* **Placement.** All runs use socket 0 only, physical cores only (no SMT siblings), memory bound
  to NUMA node 0 (`numactl --physcpubind --membind`, `OMP_PROC_BIND=close OMP_PLACES=cores`).
  Core 1 is excluded: the cluster's storage daemon keeps it at 100 %. So "all performance cores"
  means the 55 usable physical cores of one socket. The CPU is not a hybrid design, so there
  are no E-cores to exclude.
* **Repetitions.** Every configuration runs 3 times as 3 separate processes. Each records its own
  peak RSS (`/usr/bin/time`), major faults, swap-in/out (from `/proc/vmstat`) and 1-minute load average.
* **Memory accounting.** Weights are loaded with `-lm none` (no mmap). With mmap, RSS counts the
  mapped file pages *and* the CPU-repacked copy, double counting the model. Runs over 4.5 GB
  are flagged in the CSV and none of the benchmark runs is.
* **Prefill** = `llama-bench -p L -n 0`. **Decode at context C** = `-n 128 -d C-128`, so the KV
  cache never exceeds the context cap (1B-class: 16k, 3B-class: 8k; the Granite MoE models
  take the cap of the Llama model of the same total size).
* **Laptop-hygiene items that do not apply to a cluster node** (charger, power mode, closing the
  browser) are replaced by: governor/driver recorded in `hw/hardware.txt`, load average logged
  per run, and the measured socket kept free of our own other work during sweeps.

## Profiling without `perf`

`kernel.perf_event_paranoid = 3` on this cluster, so `perf` cannot attach for normal users (VTune
and uProf are not available either). Instead `patches/ggml_op_profile.patch` adds an opt-in
per-operator timer to the ggml CPU graph loop, compiled only into `build-prof`
(`-DGGML_OP_PROFILE`) and enabled with `GGML_OP_PROFILE=<csv>`. Thread 0 timestamps each graph node
before it starts and after the barrier that ends it. That gives:

* `total`: the op's wall time, i.e. the slowest thread plus barrier wait;
* `own`: thread 0's own compute;
* `total - own`: the synchronisation and load-imbalance cost.

Graphs are tagged as decode (1 token) or prefill. Node names (`ffn_moe_topk`, `Qcur`, ...) are
kept, which is what makes the expert-routing cost measurable. The benchmark binaries are built
without the macro, so their code is unchanged.

## Bonus: Mamba-3 decode step

See `bonus_mamba3/README.md`. In short:

* The official SISO, MIMO and Mamba-2 step references are extracted verbatim from the pinned
  state-spaces/mamba checkout and run on CPU in PyTorch.
* One C++ source compiles to scalar, AVX2, AVX-512 and NEON kernels.
* The kernels match the fp64 reference to <= 3.5e-7 relative over 1,000 steps, for every rank
  and ISA.
* They are timed against Mamba-2 at the same state size, and wrapped as a ggml custom op.

## Notes and deviations

* **Hardware.** The brief targets an 8 GB laptop. At the user's request this project runs on a
  cluster node (2 x Xeon Platinum 8480+, 503 GB). To stay comparable, it uses one socket and keeps
  every run under the 4.5 GB budget. Bandwidth (≈136 GB/s read on one socket) is ~2-3x a laptop's,
  so absolute tok/s are higher, but the methodology and the roofline reasoning are unchanged.
* The F16 GGUFs are used only as the perplexity reference for quantization error. They are never
  benchmarked. The 3B F16 files exceed the 4.5 GB budget, which is noted in `perplexity.csv`.
* The node is shared. Load average is logged with every run; spreads larger than ~5 % at
  4-24 threads come from other users' jobs.
* **Expert-count experiment build.** llama.cpp sizes its graph node table for the trained k = 8, so
  k > 8 overflows it (`ggml_backend_sched_split_graph` aborts). An `#ifdef`-guarded patch adds
  `LLAMA_GRAPH_NODES_EXTRA` nodes in a separate `build-exp`. Every k runs on that binary; its k = 8
  numbers are within 3-5 % of the main build (`results/csv/bench_experts_mainbuild_check.csv`).
* **Busy-core run kept as evidence.** The first reference run pinned all 56 cores of socket 0,
  including the daemon's core 1, and ran ~10x slower (`results/csv/busycore_reference_precision.csv`).
  `reference_run.sh` now takes the core list from `$CORES` when set.
* `/proc/vmstat` swap counters are system-wide, so the `swap_in/out` columns can show other users'
  paging. Per-process major faults (`majflt`) are the reliable per-run signal; they were ~0 throughout.
* GPU jobs are never submitted. The PARAM Rudra reference test is prepared as
  `scripts/rudra_reference.sbatch` for the team to submit.
