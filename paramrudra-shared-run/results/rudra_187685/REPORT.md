# PARAM Rudra: hardware, profile, roofline and the shared reference run

Job 187685 on node rdcn25, 2026-11-20T20:24:18+05:30. All runs: 4 threads, one per physical core, pinned to socket 0 with memory bound to socket 0 (`socket`), llama.cpp CPU backend, KV cache q8_0. Every measurement is repeated 3 times: mean ± sample std (min–max, n).

## Allocation limits (recorded on the login node)

| Limit | Value |
| --- | --- |
| User | bapic_iiitd |
| Account | nsmext |
| QOS | aicte_qip |
| Max resources per user (QOS MaxTRESPU) | cpu=4,node=1 |
| Max wall time per job (QOS MaxWall) | 00:05:00 |
| Max running jobs per user | 2 |
| Max submitted jobs per user | none |
| Partition | cpu |
| Partition max time | 4-00:00:00 |
| Partition nodes | 36 |
| Partition CPUs | 1728 |

So every job gets at most 4 cores of one node for 5 minutes, on a node shared with other jobs. Full `sacctmgr` / `sinfo` / `scontrol` / `sacct` output: `limits/limits.txt` (recorded on login02, 2026-11-20T18:15:02+05:30).

## Hardware

| | |
| --- | --- |
| Node | rdcn25 (partition `cpu`, job 187685) |
| Allocation | 4 cores (CPUs 0-3), 8192 MB; shared node: other jobs may use the same memory bandwidth |
| CPU | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz |
| Cores | 2 sockets × 24 cores, one core type, 1 thread(s) per core |
| Max clock | 4000.0000 MHz |
| Caches | L1d 32K, L2 1024K, L3 36608K |
| NUMA | 2 nodes (one per socket) |
| Memory | 188 GB DDR4-2933, 6 channels per socket (Rudra manual) |
| SIMD | avx avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma sse4_2 |
| Governor / SMT / THP | performance / on / [always] madvise never |
| OS | AlmaLinux 8.9 (Midnight Oncilla) |

## Software

| | |
| --- | --- |
| llama.cpp | commit `827466b9efa0` (0 files locally changed) |
| Build | CMake Release, GGML_NATIVE=ON (-march=native = `cascadelake`), OpenMP=ON, CPU backend only |
| Compiler | /home/apps/gcc-12.3/bin/g++ (12.3.0) |
| Modules / Spack | compiler/gcc/12.3 spack/spack / cmake@3.29.2%gcc@=12.2.0 numactl@2.0.14%gcc@=13.2.0 |
| Model | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, 807694464 bytes, sha256 `6f85a640a97cf2bf…` (matches the expected checksum) |
| Run time | 2026-11-20T20:24:18+05:30 to 2026-11-20T20:25:33+05:30 (75 s) |

## Memory bandwidth (GB/s)

| Threads | STREAM Copy | STREAM Scale | STREAM Add | STREAM Triad | Read-only probe |
| --- | --- | --- | --- | --- | --- |
| 1 | 10.5 ± 0.0 (10.5–10.5, n=3) | 12.6 ± 0.0 (12.6–12.6, n=3) | 13.3 ± 0.0 (13.3–13.3, n=3) | 13.3 ± 0.0 (13.3–13.3, n=3) | 12.5 ± 0.1 (12.5–12.6, n=3) |
| 4 | 40.6 ± 0.0 (40.5–40.6, n=3) | 46.5 ± 0.1 (46.4–46.5, n=3) | 49.3 ± 0.1 (49.2–49.5, n=3) | 49.2 ± 0.0 (49.2–49.3, n=3) | 49.6 ± 0.1 (49.5–49.7, n=3) |

STREAM: standard stream.c (placement=socket cpus=0,1,2,3 stream_n=80000000 ntimes=10 readbw_gib=2); each run reports its best of 10 iterations (the STREAM rule). Read-only probe: `readbw.c`, AVX-512 loads, mean of passes 2–10. Decode only reads weights, so the probe is the roofline's bandwidth. Theoretical: 140.8 GB/s per socket (6 × DDR4-2933).

## Shared reference run

| Test | tok/s | Each repeat | Peak RSS (GB) |
| --- | --- | --- | --- |
| pp512 | 209.09 ± 0.08 (209.01–209.17, n=3) | 209.17 / 209.08 / 209.01 | 1.324 |
| tg128 | 39.65 ± 0.06 (39.59–39.70, n=3) | 39.65 / 39.7 / 39.59 | 1.324 |

`llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -t 4 -p 512 -n 128 -r 3 -ctk q8_0 -ctv q8_0 -fa 1` (exact command: `commands.txt`).

## llama.cpp's own timings (llama_perf, llama-completion)

| Run | Load (ms) | Prompt eval: tokens | ms/token | tok/s | Decode: tokens | ms/token | tok/s | Peak RSS (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 535.73 | 512 | 4.76 | 210.18 | 127 | 28.71 | 34.83 | 1.338 |
| 2 | 536.44 | 512 | 4.77 | 209.84 | 127 | 28.81 | 34.71 | 1.336 |
| 3 | 537.13 | 512 | 4.76 | 210.17 | 127 | 28.76 | 34.77 | 1.338 |
| mean ± std | 536 ± 1 (536–537, n=3) | | 4.76 ± 0.01 (4.76–4.77, n=3) | 210.06 ± 0.19 (209.84–210.18, n=3) | | 28.76 ± 0.05 (28.71–28.81, n=3) | 34.77 ± 0.06 (34.71–34.83, n=3) | |

Same 512-token prompt (first 2,140 characters of wikitext-2 test, `prompt_512.txt`), 128 tokens decoded (`-c 1024 --ignore-eos --temp 0`). Decode here runs at context 512–640; llama-bench's tg128 runs at 0–128. Generated text: `timings/run*.out`.

## Per-operator time breakdown (perf)

Not run: perf not found on rdcn25 (PERF_BIN unset). llama.cpp's own timings above still split prefill from decode.

## Roofline check (decode, 1B Q4_K_M)

Bytes read per decoded token = 799,862,912 B of weights (GGUF tensor sizes) + the KV cache at tg128's mean context of 64 tokens (2,097,152 B × 0.531 for q8_0 = 1.1 MB) = 0.8010 GB.

| Bandwidth used for the prediction | GB/s | Predicted tg128 (tok/s) | Measured tg128 (tok/s) | Measured / predicted |
| --- | --- | --- | --- | --- |
| Read-only probe, 4 threads (this job) | 49.6 | 61.9 | 39.65 ± 0.06 | 0.64 |
| STREAM Copy, 4 threads | 40.6 | 50.6 | 39.65 ± 0.06 | 0.78 |
| STREAM Triad, 4 threads | 49.2 | 61.5 | 39.65 ± 0.06 | 0.64 |
| Read-only probe, 1 thread × 4 (if bandwidth grew linearly with cores) | 50.1 | 62.6 | 39.65 ± 0.06 | 0.63 |
| Theoretical, one socket | 140.8 | 175.8 | 39.65 ± 0.06 | 0.23 |

Decode achieves 31.8 GB/s (measured tok/s × bytes per token): 64% of what 4 cores read in the probe and 23% of the socket's theoretical bandwidth.

Time per token: 25.2 ms measured, of which 16.1 ms is reading 0.801 GB at the probe's 49.6 GB/s; the other 9.1 ms is the gap to explain (compute, thread synchronisation, small operators). The per-operator profile is what splits it.

## Files

`manifest.txt` (software, build, model checksum, allocation), `node/` (lscpu, caches, NUMA, memory), `bw/` (STREAM, readbw), `reference/`, `timings/`, `perf/`, `commands.txt` (every command with its exit code and time), `limits/`.
