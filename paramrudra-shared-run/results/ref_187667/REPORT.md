# Shared reference run on PARAM Rudra

Test (fixed by the course): Llama-3.2-1B-Instruct Q4_K_M, llama-bench pp512 (512-token prompt) and tg128 (128 decoded tokens), 3 repeats, CPU only. KV cache: q8_0 (llama-bench args: -ctk q8_0 -ctv q8_0 -fa 1).

## Headline (4 threads)

| Test | tok/s (mean ± std) | min–max | Each repeat | Peak RSS (GB) | Run |
| --- | --- | --- | --- | --- | --- |
| pp512 | 211.18 ± 0.44 | 210.91–211.69 | 210.95 / 210.91 / 211.69 |  | this job |
| tg128 | 42.75 ± 0.01 | 42.74–42.76 | 42.76 / 42.76 / 42.74 |  | this job |

Placement `socket`: one thread per physical core on socket 0 (`taskset`, `OMP_PROC_BIND=close OMP_PLACES=cores`), memory bound to socket 0 (`numactl --membind`).

## Hardware

| | |
| --- | --- |
| Node | rdcn20 (partition `cpu`, job 187667) |
| Allocation | 4 cores (CPUs 0-3), 8192 MB; shared node: other jobs may use the same memory bandwidth |
| CPU | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz |
| Cores | 2 sockets × 24 cores, one core type, 1 thread(s) per core |
| Max clock | 4000.0000 MHz |
| Caches | L1d 32K, L2 1024K, L3 36608K |
| NUMA | 2 nodes (one per socket) |
| Memory | 188 GB DDR4-2933, 6 channels per socket (Rudra manual) |
| SIMD | avx avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma sse4_2 |
| Governor / SMT / THP | performance / on / [always] madvise never |
| OS | ? |

## Software

| | |
| --- | --- |
| llama.cpp | commit `unknown` (0 files locally changed) |
| Build | CMake Release, GGML_NATIVE=ON (-march=native = `cascadelake`), OpenMP=ON, CPU backend only |
| Compiler | /home/apps/gcc-12.3/bin/g++ (12.3.0) |
| Modules / Spack | compiler/gcc/12.3 spack/spack / cmake@3.29.2%gcc@=12.2.0 numactl@2.0.14%gcc@=13.2.0 |
| Model | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes, sha256 `6f85a640a97cf2bf…` (matches the expected checksum) |
| Run time | 2026-11-20T17:11:01+05:30 to 2026-11-20T17:11:23+05:30 (22 s) |

Exact command lines with exit codes and times: `commands.txt`. Raw data: `reference/*.csv` (mean/std), `reference/*.jsonl` (every repeat), `reference/*.time` (peak RSS), `node/`, `manifest.txt`.
