# PARAM Rudra: shared reference run and 4-core study

Project 1, Option 1 (dense transformer baseline), PARAM Rudra section, 20 Nov 2026.

**Result.** On 4 cores of a PARAM Rudra node, Llama-3.2-1B-Instruct Q4_K_M runs the shared reference test at **pp512 209.68 ± 0.41 tok/s and tg128 42.13 ± 0.02 tok/s**, with peak RSS 1.32 GB (job 187670, node rdcn20).

Decode keeps scaling with every core, 3.10× from 1 to 4 cores, because each Xeon core can read only about 12.5 GB/s. It reaches 64% of what the 4 cores can read. The rest of the time is a nearly fixed ~9 ms per token that does not shrink with more cores. No per-operator profile could be taken, because perf is not installed on Rudra.

## 1. What we were allowed to run

The QOS sets the ceiling, not the partition: every job gets at most 4 cores of one shared node for 5 minutes. These limits were recorded on login02 by `50_limits_login.sh` (`sacctmgr`, `sinfo`, `scontrol`, `sacct`).

| Limit | Value |
| --- | --- |
| User / account / QOS | bapic_iiitd / nsmext / aicte_qip |
| Max resources per user (QOS MaxTRESPU) | cpu=4, node=1 |
| Max wall time per job (QOS MaxWall) | 5 min |
| Max running jobs per user (MaxJobsPU) | 2; no limit on submitted jobs |
| Partition `cpu` | 36 nodes (rdcn01–36) × 48 cores, 188 GB each; max time 4 days; default 3,800 MB per CPU |
| Most any of our jobs got (`sacct`, 18–20 Nov) | 4 CPUs, 8 GB, 2 min 11 s |

Every job therefore asks for `--cpus-per-task=4 --mem=8G --time=00:04:30`. The node is shared with other users' jobs, which can take some of its memory bandwidth.

## 2. Hardware record

Recorded inside job 187685 (node rdcn25) with `lscpu`, `/sys` and `numactl`. Nodes rdcn20 and rdcn25 have the same model.

| Item | Value |
| --- | --- |
| CPU | Intel Xeon Gold 6240R (Cascade Lake), 2.4 GHz base, 4.0 GHz max turbo |
| Core types and count | 2 sockets × 24 cores, one core type (no hybrid cores), 1 hardware thread per core |
| Caches | L1d 32 KB and L2 1 MB per core; L3 35.75 MB per socket |
| SIMD | SSE4.2, AVX, AVX2, FMA, AVX-512 (F, BW, CD, DQ, VL), AVX-512 VNNI; no AVX-512 FP16 or BF16 |
| Memory | 188 GB DDR4-2933, 6 channels per socket = 140.8 GB/s theoretical per socket; 2 NUMA nodes (one per socket) |
| Our allocation | CPUs 0–3 (4 physical cores of socket 0), 8 GB |
| OS and settings | AlmaLinux 8.9; `performance` governor; transparent huge pages `always` |
| Software | llama.cpp `827466b` (unmodified), CPU backend, `-DGGML_NATIVE=ON` (= `cascadelake`), OpenMP, gcc 12.3; model sha256 `6f85a640…` matches |

## 3. Memory bandwidth

The 4 cores reach about 50 GB/s, only 35% of the socket's theoretical bandwidth. Bandwidth grows 3.97× from 1 to 4 cores, so each core's own reads are the limit, not DRAM. These are means ± std of 3 runs in job 187685, with memory bound to socket 0.

| Threads | STREAM Copy | STREAM Scale | STREAM Add | STREAM Triad | Read-only probe |
| --- | --- | --- | --- | --- | --- |
| 1 | 10.5 ± 0.0 | 12.6 ± 0.0 | 13.3 ± 0.0 | 13.3 ± 0.0 | 12.5 ± 0.1 |
| 4 | 40.6 ± 0.0 | 46.5 ± 0.1 | 49.3 ± 0.1 | 49.2 ± 0.0 | 49.6 ± 0.1 |

All values are GB/s.

- **STREAM:** the standard `stream.c` with 80M-element arrays (1.9 GB, 54× the L3), best of 10 iterations per run.
- **Read-only probe:** `readbw.c`, AVX-512 loads over 2 GiB, mean of passes 2–10. Decode only reads weights, so this probe is the roofline's bandwidth.
- **Theoretical:** for comparison, one Zen 4 core on the workstation reads 28 GB/s. The likely cause of the difference is the server chip's higher memory latency, which limits how many reads one core keeps in flight.

## 4. Shared reference run

The test is fixed by the course: `llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -t 4 -p 512 -n 128 -r 3 -ctk q8_0 -ctv q8_0 -fa 1`. Threads run one per core on socket 0 with memory bound to socket 0. The KV cache is Q8_0 because of the rule "No F16 anywhere".

| Job | Node | Prefill pp512 (tok/s) | Decode tg128 (tok/s) | Peak RSS (GB) |
| --- | --- | --- | --- | --- |
| **187670 (headline)** | rdcn20 | **209.68 ± 0.41** (209.31–210.12) | **42.13 ± 0.02** (42.11–42.14) | 1.32 |
| 187667 | rdcn20 | 211.18 ± 0.44 (210.91–211.69) | 42.75 ± 0.01 (42.74–42.76) | not recorded |
| 187685 | rdcn25 | 209.09 ± 0.08 (209.01–209.17) | 39.65 ± 0.06 (39.59–39.70) | 1.32 |

- **Same node (rdcn20):** repeats agree within 0.4%, and the two jobs within 1.5%.
- **Another node (rdcn25):** prefill is the same, but decode is 6% slower. Decode is the bandwidth-sensitive phase and these are shared nodes, so we take ±6% between nodes as normal.
- **Memory:** peak RSS is 1.32 GB, far under the 4.5 GB budget.

## 5. Thread scaling, Rudra against the workstation

On Rudra, decode gains from every core (3.10× at 4). On the Ryzen 9 7950X3D, decode stops at 3 cores, where it saturates DRAM. Prefill scales almost linearly on both, and the Xeon is 28–32% slower per core.

| Cores | Rudra pp512 (tok/s) | Rudra tg128 (tok/s) | Rudra decode reads (GB/s) | Ryzen pp512 (tok/s) | Ryzen tg128 (tok/s) | Ryzen decode reads (GB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 57.37 ± 0.01 | 13.58 ± 0.01 | 10.9 | 80.07 ± 0.03 | 34.97 ± 0.01 | 28.0 |
| 2 | 111.28 ± 0.45 | 24.26 ± 0.04 | 19.4 | 148.62 ± 0.10 | 61.34 ± 0.23 | 49.1 |
| 3 | 156.44 ± 0.06 | 33.35 ± 0.01 | 26.7 | 229.20 ± 0.08 | 69.32 ± 0.07 | 55.4 |
| 4 | 209.68 ± 0.41 | 42.13 ± 0.02 | 33.7 | 308.32 ± 0.30 | 70.41 ± 0.04 | 56.3 |
| 4 vs 1 | 3.65× | 3.10× | | 3.85× | 2.01× | |

- **Rudra runs:** jobs 187669 (1–3 cores) and 187670 (4 cores), on the same node and CPUs.
- **Ryzen runs:** cores 0–3 (CCD0), with the same commit, model, KV cache and command.
- **Decode reads:** tg128 tok/s × 0.80 GB of weights per token.

## 6. llama.cpp's own timings

llama.cpp's own counters (`llama_perf`) agree with llama-bench on prefill within 0.5%. Decode is slower here only because it runs at a longer context. The runs use `llama-completion` on a fixed 512-token prompt (the first 2,140 characters of wikitext-2 test) with 128 tokens decoded (`-c 1024 --ignore-eos --temp 0`). Each value is the mean ± std of 3 runs in job 187685.

| Measure | Value |
| --- | --- |
| Model load | 536 ± 1 ms |
| Prompt eval (prefill, 512 tokens) | 4.76 ± 0.01 ms/token = 210.06 ± 0.19 tok/s |
| Eval (decode, 127 tokens at context 512–640) | 28.76 ± 0.05 ms/token = 34.77 ± 0.06 tok/s |
| Peak RSS | 1.34 GB |

Decode at context 512–640 costs 3.5 ms more per token than llama-bench's tg128 at context 0–128 on the same node (28.8 against 25.2 ms). Reading the extra 8.9 MB of Q8_0 KV cache takes only 0.2 ms at 49.6 GB/s. So attention costs compute, not bandwidth, as on the workstation.

## 7. Roofline check

**Prediction.** Decoding one token reads every weight once plus the KV cache, so the predicted speed is bandwidth ÷ bytes per token.

- **Weights:** 799,862,912 B, the sum of the GGUF file's stored tensor sizes.
- **KV cache:** 1.1 MB at tg128's mean context of 64 tokens (64 × 32,768 B × 34/64 for Q8_0).
- **Total:** 0.801 GB per token.

| Bandwidth used (job 187685) | GB/s | Predicted tg128 (tok/s) | Measured / predicted |
| --- | --- | --- | --- |
| Read-only probe, 4 threads | 49.6 | 61.9 | 0.64 |
| STREAM Copy, 4 threads | 40.6 | 50.6 | 0.78 |
| STREAM Triad, 4 threads | 49.2 | 61.5 | 0.64 |
| Theoretical, one socket | 140.8 | 175.8 | 0.23 |

The ratios use tg128 = 39.65 ± 0.06 tok/s, measured in the same job. Decode therefore moves 31.8 GB/s: 64% of the probe and 23% of the socket's theoretical bandwidth. STREAM Copy looks closer only because it under-measures read bandwidth: it writes as much as it reads.

**The gap is a fixed time per token.** We split each token's time into "reading 0.801 GB at the probe's bandwidth" and the rest. The probe was measured at 1 and 4 threads; at 2 and 3 we assume 12.5 GB/s per core, because it scales 3.97× by 4 cores.

| Cores | Measured (ms/token) | Reading the bytes (ms) | Extra (ms) |
| --- | --- | --- | --- |
| 1 | 73.6 | 64.1 | 9.6 |
| 2 | 41.2 | 32.0 | 9.2 |
| 3 | 30.0 | 21.4 | 8.6 |
| 4 | 23.7 | 16.1 | 7.6 |
| 4 (job 187685, rdcn25) | 25.2 | 16.1 | 9.1 |

The measured times are from jobs 187669 and 187670 on rdcn20; the bandwidth is from job 187685 on rdcn25.

**What the fixed extra time rules out:**

- **Work that splits across cores:** per-core arithmetic, such as Q4_K dequantization and the dot products, would shrink 4× from 1 to 4 cores, from 9.6 to 2.4 ms. The extra time instead stays at 8–10 ms.
- **Reading below the probe's rate:** a weaker access pattern would also shrink as 1/cores, because the reading time does.

What stays roughly constant is work that does not split across threads, or that grows with them:

- the barrier that synchronises all threads after each of the roughly 300 operations in one token's compute graph;
- small operations such as RMSNorm, RoPE, the residual adds and quantizing the activations for the next matmul.

This is an inference from the timings. Without a profile it is not confirmed.

## 8. Profiling: not possible on Rudra yet

`perf` is installed on neither the login node nor the compute nodes, so job 187685 skipped the per-operator breakdown and wrote the reason to `perf/NOT_RUN.txt`. The login node does allow profiling one's own processes (`perf_event_paranoid=0`). It also offers Intel VTune (`compiler/oneapi-2024/vtune/2024.1`), whose user-mode hotspots collection needs no kernel driver. A VTune hotspots run of decode would split the ~9 ms per token and test the barrier explanation. On the workstation, perf shows short-context decode spending 86–90% of samples in two matrix-vector kernels.

## 9. Limitations

- **4 of 48 cores, on a shared node:** this shows how the Xeon scales per core, not what a whole socket or node can do. Long-context prefill and decode (6–8 min on 4 cores) do not fit the 5-minute limit.
- **Node-to-node variation:** decode differs by 6% between rdcn20 and rdcn25. The bandwidth numbers come only from rdcn25, so the per-token split above mixes two nodes; the 4-core row from rdcn25 alone gives the same answer, 9.1 ms.
- **No profile:** the explanation of the gap is inferred, not measured.
- **Only the reference model:** Llama-3.2-1B Q4_K_M only. The workstation covers the Q8_0 1B and the 3B.

## 10. Reproduce

Scripts are in [`scripts/`](scripts/README.md) and raw data in [`results/`](results/README.md). On a Rudra login node, inside a copy of `scripts/`:

```bash
./00_setup_login.sh            # llama.cpp 827466b, model (sha256 checked), build, STREAM + readbw
./50_limits_login.sh           # Slurm limits -> results/limits_<date>/
./submit.sh reference report   # two 4-core jobs, each under 3 min
```

| Section | Source folder |
| --- | --- |
| 1 Limits | `results/limits_20261120_181502/` |
| 2–3, 6–8 Hardware, bandwidth, timings, roofline, profile | `results/rudra_187685/` |
| 4–5 Reference run and thread scaling | `results/ref_187670/`, `ref_187669/`, `ref_187667/` |
| Workstation comparison | the main project report |

Every result folder holds `manifest.txt` (software, build, checksum, allocation), `node/` (hardware) and `commands.txt` (each exact command with its exit code and time).
