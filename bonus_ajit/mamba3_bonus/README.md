# Bonus: Mamba-3 decode step on a CPU (SISO and MIMO), AMD Ryzen 9 5950X

**Question (from the brief):** MIMO adds decode compute without growing the state. On GPUs that comes at little
cost, because decode is limited by memory reads. Does the same hold on a CPU?

**Answer:** yes inside a real decode, no when the state is in cache.

* MIMO-4 does 3.57x the FLOPs of SISO.
* A full layer (SSM step plus streaming its Q4_K_M weights) takes **1.07x** as long on 16 cores and **1.22x** on 1 core.
* With the state cache-resident, the step is compute-bound and MIMO-4 costs **1.8-2.9x**.
* All **48/48** correctness runs match the official fp64 reference over 1,000 steps (worst relative error 4.2e-7).
* Peak RSS is at most 445 MB in every configuration, under the brief's 0.5 GB.

## Contents

| What | Where |
|---|---|
| Report (4 pages) | `report/Mamba3_Bonus_Report.pdf` (editable: `.docx`) |
| Slides (9, 16:9) | `slides/Mamba3_Bonus_Talk.pptx` |
| Every number, as tables | `results/summary.md` (machine-readable: `results/numbers.json`) |
| Raw results (CSV) | `results/` (see below) |
| Figures | `plots/*.png` |
| Kernel source | `src/` + `Makefile` |

### Raw results

| File | Content |
|---|---|
| `results/verify.csv`, `verify.txt` | 1,000-step correctness of every kernel vs the official fp64 reference (scalar/AVX2 × 1/8/16 threads × 8 cases) |
| `results/bench.csv`, `bench.txt`, `bench_summary.csv` | Time per step: 5 variants × 2 ISAs × 1-16 threads × 4 memory regimes × 3 repetitions |
| `results/bench_ccd_one_ccd.csv`, `bench_ccd_two_ccd.csv`, `bench_ccd.txt` | Own experiment: 8 threads on one CCD vs split over two |
| `results/ggml_m3.csv`, `ggml_m3.txt` | ggml `SSM_SCAN` baseline and the Mamba-3 ggml custom op |
| `results/stream.csv` | STREAM + read-only bandwidth, 1-16 cores, and 8 cores split over both CCDs |
| `results/memory.csv` | Peak RSS of every binary and regime, 3 runs |
| `results/xeon_team_run/` | The team's Xeon 8480+ numbers, used in the three-CPU comparison |

## Kernel source (`src/`)

| File | Content |
|---|---|
| `m3_kernel.inc` | SISO and MIMO Mamba-3 decode steps and the Mamba-2 step, written once over a small vector layer |
| `m3_{scalar,avx2,avx512,neon}.cpp`, `m3_dispatch.cpp` | Each ISA compiled separately and picked at run time (scalar = plain C++, no auto-vectorisation) |
| `verify.cpp` | Correctness driver |
| `bench.cpp` | Timing driver; regimes `hot`, `layersN`, `modelN` (state + per-layer Q4 weight traffic), `cold` |
| `ggml_m3.cpp` | ggml integration (stretch goal) |

The kernel source is shared with the team's `bonus_mamba3/`. The `layersN` and `modelN` regimes in `bench.cpp`
are new here. The Core Ultra numbers in the comparison come from Rupinder's README and were measured with his own AVX2 kernels.

## Memory regimes

| Regime | What it models |
|---|---|
| `hot` | one 2 MiB layer state reused, so it stays in L2/L3 |
| `layers24` | 24 layer states (48 MiB), the depth of mamba3-1.5b |
| `model24` | 24 states + each layer's Q4_K_M in_proj/out_proj bytes (14.7 / 15.6 MB) streamed before its step: the cache traffic of a full decode |
| `cold` | 128 states (256 MiB, 4x the L3), always from DRAM (the team Xeon run used 512 MiB) |

## Known limitations

* The kernel is measured, not a whole model. `model24` reads the weights without dequantising them, so it is a lower bound on projection time.
* NEON compiles but is untested (no ARM hardware). AVX-512 cannot run on Zen 3.
* ggml's `SSM_SCAN` call took 102 µs at 16 threads against 16 µs at 8. That was not investigated, and is probably per-call thread start-up.
