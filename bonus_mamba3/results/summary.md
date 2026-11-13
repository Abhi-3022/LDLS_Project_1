# Mamba-3 decode step on CPU: results

Layer shape: d_model 2048, d_inner 4096, 64 heads x head dim 64, d_state 128, 1 B/C group, RoPE on half the state (32 angles). State per layer = 2 MiB fp32 (same for Mamba-2).

## Correctness (1000 decode steps vs official fp64 reference, tolerance 1e-5 relative)

| case       | isa    |   rel_err |   state_rel_err |   torch_fp32_rel_err |   torch_fp32_state_rel_err | result   |
|:-----------|:-------|----------:|----------------:|---------------------:|---------------------------:|:---------|
| siso       | scalar |  3.02e-07 |        2.80e-07 |             1.76e-07 |                   6.03e-06 | PASS     |
| siso       | avx2   |  1.81e-07 |        2.80e-07 |             1.76e-07 |                   6.03e-06 | PASS     |
| siso       | avx512 |  1.86e-07 |        2.80e-07 |             1.76e-07 |                   6.03e-06 | PASS     |
| mimo1      | scalar |  2.88e-07 |        2.59e-07 |             1.41e-07 |                   1.72e-05 | PASS     |
| mimo1      | avx2   |  1.89e-07 |        2.59e-07 |             1.41e-07 |                   1.72e-05 | PASS     |
| mimo1      | avx512 |  1.89e-07 |        2.59e-07 |             1.41e-07 |                   1.72e-05 | PASS     |
| mimo2      | scalar |  2.42e-07 |        3.38e-07 |             1.92e-07 |                   1.58e-05 | PASS     |
| mimo2      | avx2   |  1.70e-07 |        3.78e-07 |             1.92e-07 |                   1.58e-05 | PASS     |
| mimo2      | avx512 |  1.70e-07 |        3.78e-07 |             1.92e-07 |                   1.58e-05 | PASS     |
| mimo4      | scalar |  3.48e-07 |        2.57e-07 |             1.91e-07 |                   3.15e-05 | PASS     |
| mimo4      | avx2   |  1.19e-07 |        3.09e-07 |             1.91e-07 |                   3.15e-05 | PASS     |
| mimo4      | avx512 |  1.03e-07 |        3.09e-07 |             1.91e-07 |                   3.15e-05 | PASS     |
| mimo8      | scalar |  1.55e-07 |        6.72e-07 |             1.80e-07 |                   1.70e-05 | PASS     |
| mimo8      | avx2   |  1.48e-07 |        6.41e-07 |             1.80e-07 |                   1.70e-05 | PASS     |
| mimo8      | avx512 |  1.48e-07 |        6.41e-07 |             1.80e-07 |                   1.70e-05 | PASS     |
| mamba2     | scalar |  2.43e-07 |        1.45e-07 |             1.83e-07 |                   1.40e-07 | PASS     |
| mamba2     | avx2   |  1.42e-07 |        1.18e-07 |             1.83e-07 |                   1.40e-07 | PASS     |
| mamba2     | avx512 |  1.46e-07 |        1.18e-07 |             1.83e-07 |                   1.40e-07 | PASS     |
| real_siso  | scalar |  3.93e-07 |        7.11e-07 |             6.73e-06 |                   7.03e-04 | PASS     |
| real_siso  | avx2   |  1.36e-07 |        6.81e-07 |             6.73e-06 |                   7.03e-04 | PASS     |
| real_siso  | avx512 |  1.36e-07 |        6.81e-07 |             6.73e-06 |                   7.03e-04 | PASS     |
| real_mimo4 | scalar |  2.43e-07 |        6.05e-07 |             7.91e-06 |                   5.77e-04 | PASS     |
| real_mimo4 | avx2   |  1.09e-07 |        6.05e-07 |             7.91e-06 |                   5.77e-04 | PASS     |
| real_mimo4 | avx512 |  1.09e-07 |        6.05e-07 |             7.91e-06 |                   5.77e-04 | PASS     |

All 48 runs (1 and 8 threads): 48 PASS, 0 FAIL.

## Time per decode step (µs, mean of 3 reps), AVX-512

| state   |   threads |   Mamba-2 |   Mamba-3 SISO |   MIMO R=2 |   MIMO R=4 |   MIMO R=8 |   MIMO4 / SISO |
|:--------|----------:|----------:|---------------:|-----------:|-----------:|-----------:|---------------:|
| cold    |         1 |     192.1 |          196.3 |      218.5 |      256.6 |      420.4 |           1.31 |
| cold    |         2 |      97.2 |          101.2 |      112.5 |      132.8 |      212.2 |           1.31 |
| cold    |         4 |      55   |           57.1 |       62.8 |       73.4 |      108.9 |           1.29 |
| cold    |         8 |      39   |           40.9 |       44.2 |       48.4 |       90.4 |           1.18 |
| cold    |        16 |      35.3 |           36.5 |       37.6 |       40.8 |       54.2 |           1.12 |
| cold    |        32 |      33.9 |           37.9 |       37.4 |       39.1 |       43.4 |           1.03 |
| hot     |         1 |      66.8 |           78.1 |       98.3 |      162.5 |      366.7 |           2.08 |
| hot     |         2 |      28.7 |           34.9 |       44.8 |       82.3 |      180.7 |           2.36 |
| hot     |         4 |      15.7 |           19.7 |       24.8 |       43.1 |       92.8 |           2.19 |
| hot     |         8 |       9.4 |           13.3 |       14.8 |       24.1 |       50.2 |           1.8  |
| hot     |        16 |       8.9 |           11.6 |       12.9 |       22.4 |       45.1 |           1.93 |
| hot     |        32 |       8.4 |           10.7 |       12.1 |       15.6 |       26.7 |           1.45 |

## FLOPs, bytes and arithmetic intensity per step

| variant      |    flops |   bytes |    ai |
|:-------------|---------:|--------:|------:|
| Mamba-2      |  2621440 | 4194304 | 0.625 |
| MIMO R=2     |  6815744 | 4390912 | 1.55  |
| MIMO R=4     | 13107200 | 4521984 | 2.9   |
| MIMO R=8     | 25690112 | 4784128 | 5.37  |
| Mamba-3 SISO |  3670016 | 4325376 | 0.848 |

## Same comparison per ISA, DRAM-resident state, 1 thread (µs)

| isa    |   Mamba-2 |   Mamba-3 SISO |   MIMO R=2 |   MIMO R=4 |   MIMO R=8 |   MIMO4 / SISO |
|:-------|----------:|---------------:|-----------:|-----------:|-----------:|---------------:|
| scalar |     394.7 |          472.8 |      807.9 |     1435.7 |     3405.8 |           3.04 |
| avx2   |     195.6 |          209   |      227.4 |      329.9 |      688.6 |           1.58 |
| avx512 |     192.1 |          196.3 |      218.5 |      256.6 |      420.4 |           1.31 |

## Full layer estimate (core measured + in/out projections at measured read bandwidth, Q4_K_M weights)

|   threads | variant      |   proj MB |   proj µs |   core µs |   layer µs |
|----------:|:-------------|----------:|----------:|----------:|-----------:|
|         1 | Mamba-3 SISO |     14.71 |    1082.9 |     196.3 |     1279.2 |
|         1 | MIMO R=4     |     15.59 |    1148   |     256.6 |     1404.6 |
|         8 | Mamba-3 SISO |     14.71 |     158.6 |      40.9 |      199.5 |
|         8 | MIMO R=4     |     15.59 |     168.1 |      48.4 |      216.5 |
|        32 | Mamba-3 SISO |     14.71 |     108.4 |      37.9 |      146.3 |
|        32 | MIMO R=4     |     15.59 |     114.9 |      39.1 |      154   |

## ggml integration

| part   | case   | impl               |   threads |   steps |   us_per_step |   rel_err_vs_ref |   max_abs_diff_vs_ours |   direct_us_per_step |
|:-------|:-------|:-------------------|----------:|--------:|--------------:|-----------------:|-----------------------:|---------------------:|
| A      | mamba2 | ggml_ssm_scan      |         1 |    1000 |        272    |         6.09e-07 |             nan        |                nan   |
| A      | mamba2 | ours_avx512        |         1 |    1000 |         88    |         1.38e-07 |               1.91e-05 |                nan   |
| A      | mamba2 | ggml_ssm_scan      |         8 |    1000 |         45.1  |         6.09e-07 |             nan        |                nan   |
| A      | mamba2 | ours_avx512        |         8 |    1000 |         17.6  |         1.38e-07 |               1.91e-05 |                nan   |
| A      | mamba2 | ggml_ssm_scan      |        32 |    1000 |         48.3  |         6.09e-07 |             nan        |                nan   |
| A      | mamba2 | ours_avx512        |        32 |    1000 |          9.51 |         1.38e-07 |               1.91e-05 |                nan   |
| B      | siso   | ggml_custom_avx512 |         1 |    1000 |         97    |         1.86e-07 |               0        |                 93.7 |
| B      | siso   | ggml_custom_avx512 |         8 |    1000 |         29.6  |         1.86e-07 |               0        |                 22.3 |
| B      | siso   | ggml_custom_avx512 |        32 |    1000 |         25.1  |         1.86e-07 |               0        |                 12.1 |
| B      | mimo4  | ggml_custom_avx512 |         1 |    1000 |        170    |         1.03e-07 |               0        |                168   |
| B      | mimo4  | ggml_custom_avx512 |         8 |    1000 |         41.8  |         1.03e-07 |               0        |                 33.9 |
| B      | mimo4  | ggml_custom_avx512 |        32 |    1000 |         40.8  |         1.03e-07 |               0        |                 17.6 |

## Answer: does MIMO's extra compute come at little cost on this CPU?

Partly. MIMO at rank 4 does 3.6x the FLOPs of SISO per step (25 vs 7 FLOP per state element) on the same 2 MiB state, so its arithmetic intensity rises from 0.85 to 2.90 FLOP/B. When the state must stream from DRAM, as it does for a deep model, the extra compute is mostly hidden behind the memory traffic: one AVX-512 core takes 196 µs (SISO) vs 257 µs (MIMO-4), 1.31x, and 8 cores take 41 vs 48 µs (1.18x), far below the 3.6x FLOP ratio. Once enough cores stream the state at the socket's bandwidth (32 threads: 38 vs 39 µs, 114 GB/s of state traffic) MIMO-4 is 1.03x, i.e. effectively free, which is the GPU argument holding on a CPU. With AVX2 only, as on most laptops, the ratio grows to 1.58x on one core (209 vs 330 µs), because half-width vectors halve the compute headroom. When the state is cache-resident (8 threads, 256 KiB of state per core in L2), the step becomes compute-bound and MIMO-4 costs 1.80x (13 vs 24 µs). For reference, Mamba-2 with the same state takes 192 µs on one core (DRAM-resident). For a full layer, the Q4_K_M in_proj/out_proj reads (~14.7 MB for SISO, ~15.6 MB for MIMO-4, whose B/C projections are 4x wider) dominate: per layer, SISO and MIMO-4 come to about 1279 vs 1405 µs on 1 thread (+10%) and 200 vs 216 µs on 8 threads (+9%). So on this CPU MIMO's extra compute is not free inside the SSM kernel, but at the level of a decode step it costs on the order of 10%, and less when the kernel is memory-bound.
