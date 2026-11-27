# Mamba-3 decode step on an AMD Ryzen 9 5950X: results

Layer: d_model 2048, 64 heads x head dim 64, d_state 128, 1 B/C group, RoPE on half the state. State per layer 2 MiB fp32 (same for Mamba-2). AVX2 + FMA (Zen 3 has no AVX-512).

## Memory bandwidth (GB/s, mean of 3)

|   threads | cores                                 |   copy_GBps |   scale_GBps |   add_GBps |   triad_GBps |   read_GBps |
|----------:|:--------------------------------------|------------:|-------------:|-----------:|-------------:|------------:|
|         1 | 0                                     |        38.2 |         23.9 |       26.4 |         26.4 |        33.1 |
|         2 | 0,1                                   |        40.9 |         26.3 |       28.8 |         28.8 |        43.7 |
|         4 | 0,1,2,3                               |        38.9 |         24.9 |       27.4 |         27.5 |        41.4 |
|         8 | 0,1,2,3,4,5,6,7                       |        37.4 |         23.4 |       25.6 |         25.7 |        38.3 |
|         8 | 0,1,2,3,8,9,10,11                     |        40.3 |         25.7 |       27.9 |         27.9 |        44.6 |
|        16 | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |        37.9 |         23.9 |       26.2 |         26.2 |        42.6 |

## Correctness: 1000 decode steps vs the official fp64 reference (tolerance 1e-5 relative)

| case       | isa    |   rel_err |   state_rel_err |   torch_fp32_rel_err |   torch_fp32_state_rel_err | result   |
|:-----------|:-------|----------:|----------------:|---------------------:|---------------------------:|:---------|
| siso       | scalar |  3.71e-07 |        2.78e-07 |             1.52e-07 |                   7.20e-06 | PASS     |
| siso       | avx2   |  2.17e-07 |        2.78e-07 |             1.52e-07 |                   7.20e-06 | PASS     |
| mimo1      | scalar |  3.33e-07 |        2.56e-07 |             1.54e-07 |                   1.55e-05 | PASS     |
| mimo1      | avx2   |  1.85e-07 |        2.31e-07 |             1.54e-07 |                   1.55e-05 | PASS     |
| mimo2      | scalar |  2.26e-07 |        3.59e-07 |             2.22e-07 |                   1.26e-05 | PASS     |
| mimo2      | avx2   |  1.53e-07 |        3.59e-07 |             2.22e-07 |                   1.26e-05 | PASS     |
| mimo4      | scalar |  2.19e-07 |        2.68e-07 |             2.35e-07 |                   2.88e-05 | PASS     |
| mimo4      | avx2   |  1.29e-07 |        2.57e-07 |             2.35e-07 |                   2.88e-05 | PASS     |
| mimo8      | scalar |  1.99e-07 |        6.00e-07 |             1.64e-07 |                   1.36e-05 | PASS     |
| mimo8      | avx2   |  1.46e-07 |        6.15e-07 |             1.64e-07 |                   1.36e-05 | PASS     |
| mamba2     | scalar |  2.41e-07 |        1.38e-07 |             1.77e-07 |                   1.20e-07 | PASS     |
| mamba2     | avx2   |  1.92e-07 |        1.38e-07 |             1.77e-07 |                   1.20e-07 | PASS     |
| real_siso  | scalar |  4.17e-07 |        6.59e-07 |             8.11e-06 |                   7.57e-04 | PASS     |
| real_siso  | avx2   |  1.61e-07 |        6.59e-07 |             8.11e-06 |                   7.57e-04 | PASS     |
| real_mimo4 | scalar |  2.10e-07 |        6.01e-07 |             7.94e-06 |                   5.67e-04 | PASS     |
| real_mimo4 | avx2   |  1.65e-07 |        6.01e-07 |             7.94e-06 |                   5.67e-04 | PASS     |

All 48 runs (threads [np.int64(1), np.int64(8), np.int64(16)]): 48 PASS, 0 FAIL. Worst output error 4.17e-07, worst state error 6.59e-07.

## Time per decode step, avx2 (µs, mean ± s.d. of 3)

| state                                      |   threads | Mamba-2     | Mamba-3 SISO   | MIMO R=2    | MIMO R=4    | MIMO R=8     |   MIMO4/SISO |
|:-------------------------------------------|----------:|:------------|:---------------|:------------|:------------|:-------------|-------------:|
| 1 layer (2 MiB, in L2/L3)                  |         1 | 51.8 ± 0.1  | 64.5 ± 0.0     | 98.4 ± 1.5  | 185.9 ± 0.0 | 507.9 ± 0.3  |         2.88 |
| 1 layer (2 MiB, in L2/L3)                  |         2 | 28.6 ± 0.0  | 35.1 ± 0.5     | 51.4 ± 0.1  | 97.4 ± 0.1  | 259.8 ± 0.9  |         2.77 |
| 1 layer (2 MiB, in L2/L3)                  |         4 | 14.7 ± 0.0  | 19.0 ± 0.1     | 27.2 ± 0.0  | 48.8 ± 0.7  | 132.5 ± 0.2  |         2.56 |
| 1 layer (2 MiB, in L2/L3)                  |         8 | 7.9 ± 0.0   | 10.8 ± 0.2     | 14.0 ± 0.1  | 25.8 ± 0.1  | 67.3 ± 0.2   |         2.39 |
| 1 layer (2 MiB, in L2/L3)                  |        16 | 7.0 ± 0.1   | 9.2 ± 0.0      | 10.9 ± 0.1  | 16.8 ± 0.3  | 37.4 ± 1.1   |         1.84 |
| 24 layers (48 MiB, real 1.5B model)        |         1 | 93.5 ± 1.2  | 93.8 ± 2.3     | 118.0 ± 0.3 | 210.7 ± 0.6 | 529.7 ± 1.3  |         2.25 |
| 24 layers (48 MiB, real 1.5B model)        |         2 | 62.6 ± 0.4  | 64.2 ± 0.1     | 73.3 ± 0.6  | 116.7 ± 1.4 | 276.6 ± 0.4  |         1.82 |
| 24 layers (48 MiB, real 1.5B model)        |         4 | 55.9 ± 0.4  | 57.1 ± 0.6     | 61.8 ± 0.2  | 77.5 ± 0.2  | 145.6 ± 0.5  |         1.36 |
| 24 layers (48 MiB, real 1.5B model)        |         8 | 56.9 ± 1.0  | 59.3 ± 0.3     | 62.4 ± 0.4  | 68.8 ± 0.3  | 88.1 ± 1.0   |         1.16 |
| 24 layers (48 MiB, real 1.5B model)        |        16 | 7.7 ± 0.0   | 9.8 ± 0.0      | 11.8 ± 0.1  | 17.5 ± 0.3  | 38.5 ± 1.1   |         1.78 |
| 24 layers + their Q4 weights (full decode) |         1 | 614.4 ± 2.0 | 608.8 ± 1.6    | 627.3 ± 6.0 | 744.4 ± 4.3 | 1101.1 ± 2.1 |         1.22 |
| 24 layers + their Q4 weights (full decode) |         2 | 457.1 ± 2.8 | 458.2 ± 1.5    | 477.7 ± 3.4 | 526.7 ± 2.6 | 715.9 ± 5.4  |         1.15 |
| 24 layers + their Q4 weights (full decode) |         4 | 473.1 ± 1.2 | 478.1 ± 2.4    | 491.9 ± 2.5 | 522.6 ± 5.9 | 617.7 ± 0.6  |         1.09 |
| 24 layers + their Q4 weights (full decode) |         8 | 529.6 ± 0.1 | 539.9 ± 2.6    | 548.4 ± 0.1 | 569.9 ± 0.6 | 622.5 ± 5.5  |         1.06 |
| 24 layers + their Q4 weights (full decode) |        16 | 441.5 ± 2.3 | 444.8 ± 1.2    | 453.6 ± 2.8 | 477.3 ± 0.6 | 527.8 ± 11.1 |         1.07 |
| 128 layers (256 MiB, DRAM)                 |         1 | 138.9 ± 0.3 | 135.3 ± 1.9    | 144.2 ± 0.4 | 232.8 ± 0.5 | 569.3 ± 2.3  |         1.72 |
| 128 layers (256 MiB, DRAM)                 |         2 | 115.0 ± 0.6 | 113.4 ± 1.0    | 117.3 ± 0.4 | 142.8 ± 0.6 | 296.8 ± 0.8  |         1.26 |
| 128 layers (256 MiB, DRAM)                 |         4 | 111.2 ± 0.6 | 113.7 ± 0.8    | 115.5 ± 1.1 | 124.6 ± 0.8 | 163.4 ± 1.0  |         1.1  |
| 128 layers (256 MiB, DRAM)                 |         8 | 115.9 ± 0.8 | 119.3 ± 0.5    | 121.7 ± 0.5 | 125.3 ± 0.4 | 135.3 ± 0.9  |         1.05 |
| 128 layers (256 MiB, DRAM)                 |        16 | 105.0 ± 0.6 | 110.1 ± 1.2    | 111.8 ± 1.4 | 116.9 ± 0.9 | 124.9 ± 0.8  |         1.06 |
## Time per decode step, scalar (µs, mean ± s.d. of 3)

| state                                      |   threads | Mamba-2      | Mamba-3 SISO   | MIMO R=2     | MIMO R=4      | MIMO R=8     |   MIMO4/SISO |
|:-------------------------------------------|----------:|:-------------|:---------------|:-------------|:--------------|:-------------|-------------:|
| 1 layer (2 MiB, in L2/L3)                  |         1 | 418.0 ± 16.1 | 460.4 ± 6.8    | 752.5 ± 0.6  | 1713.8 ± 0.7  | 4163.1 ± 6.3 |         3.72 |
| 1 layer (2 MiB, in L2/L3)                  |         2 | 205.7 ± 0.1  | 232.7 ± 2.5    | 386.8 ± 0.2  | 876.6 ± 5.1   | 2083.9 ± 2.5 |         3.77 |
| 1 layer (2 MiB, in L2/L3)                  |         4 | 103.8 ± 0.1  | 120.4 ± 0.5    | 195.4 ± 0.4  | 438.4 ± 0.4   | 1047.6 ± 2.4 |         3.64 |
| 1 layer (2 MiB, in L2/L3)                  |         8 | 53.7 ± 0.2   | 61.9 ± 0.2     | 99.8 ± 0.3   | 221.2 ± 0.1   | 526.3 ± 0.8  |         3.58 |
| 1 layer (2 MiB, in L2/L3)                  |        16 | 29.6 ± 0.1   | 36.2 ± 3.5     | 55.2 ± 1.1   | 113.8 ± 0.2   | 268.7 ± 2.3  |         3.14 |
| 24 layers (48 MiB, real 1.5B model)        |         1 | 416.5 ± 0.8  | 468.9 ± 4.7    | 764.2 ± 0.3  | 1735.2 ± 1.5  | 4182.9 ± 1.1 |         3.7  |
| 24 layers (48 MiB, real 1.5B model)        |         2 | 214.1 ± 1.4  | 236.8 ± 2.5    | 393.8 ± 1.2  | 911.4 ± 21.8  | 2095.7 ± 3.1 |         3.85 |
| 24 layers (48 MiB, real 1.5B model)        |         4 | 108.8 ± 0.1  | 120.9 ± 1.3    | 200.2 ± 0.1  | 448.9 ± 1.4   | 1053.4 ± 3.0 |         3.71 |
| 24 layers (48 MiB, real 1.5B model)        |         8 | 60.0 ± 0.3   | 68.8 ± 0.2     | 104.2 ± 0.5  | 228.6 ± 0.6   | 532.9 ± 1.4  |         3.32 |
| 24 layers (48 MiB, real 1.5B model)        |        16 | 30.3 ± 0.4   | 34.5 ± 0.4     | 54.9 ± 1.6   | 115.8 ± 0.5   | 269.4 ± 3.8  |         3.36 |
| 24 layers + their Q4 weights (full decode) |         1 | 913.1 ± 4.6  | 948.8 ± 9.3    | 1266.7 ± 1.9 | 2268.7 ± 2.7  | 4672.5 ± 2.7 |         2.39 |
| 24 layers + their Q4 weights (full decode) |         2 | 607.1 ± 4.0  | 620.4 ± 2.1    | 784.2 ± 1.4  | 1305.0 ± 2.8  | 2543.8 ± 8.3 |         2.1  |
| 24 layers + their Q4 weights (full decode) |         4 | 527.8 ± 1.7  | 542.8 ± 1.9    | 628.6 ± 0.3  | 894.8 ± 1.9   | 1539.3 ± 7.7 |         1.65 |
| 24 layers + their Q4 weights (full decode) |         8 | 533.0 ± 1.0  | 543.2 ± 1.8    | 588.0 ± 4.0  | 727.8 ± 2.3   | 1069.1 ± 2.0 |         1.34 |
| 24 layers + their Q4 weights (full decode) |        16 | 440.7 ± 1.3  | 450.5 ± 7.1    | 460.6 ± 2.8  | 549.8 ± 15.9  | 727.5 ± 6.9  |         1.22 |
| 128 layers (256 MiB, DRAM)                 |         1 | 417.6 ± 0.4  | 461.8 ± 1.9    | 756.7 ± 0.7  | 1740.6 ± 19.3 | 4139.7 ± 4.9 |         3.77 |
| 128 layers (256 MiB, DRAM)                 |         2 | 215.3 ± 1.9  | 238.0 ± 0.2    | 397.9 ± 7.1  | 880.7 ± 10.4  | 2082.4 ± 4.2 |         3.7  |
| 128 layers (256 MiB, DRAM)                 |         4 | 117.1 ± 0.5  | 128.8 ± 0.3    | 202.6 ± 1.2  | 446.5 ± 1.9   | 1049.0 ± 2.4 |         3.47 |
| 128 layers (256 MiB, DRAM)                 |         8 | 121.8 ± 0.2  | 119.8 ± 0.4    | 126.3 ± 0.9  | 232.8 ± 2.4   | 537.5 ± 1.7  |         1.94 |
| 128 layers (256 MiB, DRAM)                 |        16 | 104.4 ± 0.7  | 109.8 ± 1.3    | 112.5 ± 1.3  | 128.8 ± 3.8   | 286.4 ± 8.8  |         1.17 |

## FLOPs, bytes, arithmetic intensity per step

| variant      |    flops |   bytes |    ai |   gflops |   gbps |
|:-------------|---------:|--------:|------:|---------:|-------:|
| Mamba-2      |  2621440 | 4194304 | 0.625 |     18.9 |   30.2 |
| MIMO R=2     |  6815744 | 4390912 | 1.55  |     47.3 |   30.4 |
| MIMO R=4     | 13107200 | 4521984 | 2.9   |     56.3 |   19.4 |
| MIMO R=8     | 25690112 | 4784128 | 5.37  |     45.1 |    8.4 |
| Mamba-3 SISO |  3670016 | 4325376 | 0.848 |     27.1 |   32   |

## Own experiment: CCD placement, 8 threads, AVX2 (µs)

| placement   | state                                      |   Mamba-2 |   SISO |   MIMO-4 |   MIMO4/SISO |
|:------------|:-------------------------------------------|----------:|-------:|---------:|-------------:|
| one_ccd     | 1 layer (2 MiB, in L2/L3)                  |       8.5 |   10.5 |     25.5 |         2.43 |
| one_ccd     | 12 layers (24 MiB)                         |       9.6 |   12.4 |     27.3 |         2.2  |
| one_ccd     | 24 layers (48 MiB, real 1.5B model)        |      56.8 |   59.3 |     69.7 |         1.18 |
| one_ccd     | 24 layers + their Q4 weights (full decode) |     537   |  540.2 |    570.6 |         1.06 |
| one_ccd     | 128 layers (256 MiB, DRAM)                 |     116.1 |  119.3 |    125.5 |         1.05 |
| two_ccd     | 1 layer (2 MiB, in L2/L3)                  |      10.7 |   12.9 |     27   |         2.09 |
| two_ccd     | 12 layers (24 MiB)                         |       9.8 |   12.3 |     27.1 |         2.21 |
| two_ccd     | 24 layers (48 MiB, real 1.5B model)        |      10.6 |   13.1 |     29.3 |         2.24 |
| two_ccd     | 24 layers + their Q4 weights (full decode) |     421.5 |  426.2 |    458.9 |         1.08 |
| two_ccd     | 128 layers (256 MiB, DRAM)                 |      98.5 |  102.6 |    108.5 |         1.06 |

## Three CPUs, MIMO-4 / SISO time ratio

| CPU / ISA                         |   1 thread, DRAM |   all cores, DRAM |   multi-core, cache |
|:----------------------------------|-----------------:|------------------:|--------------------:|
| Xeon 8480+ AVX-512 (team)         |             1.31 |              1.03 |                1.80 |
| Xeon 8480+ AVX2 (team)            |             1.58 |            nan    |              nan    |
| Core Ultra 7 155H AVX2 (Rupinder) |             1.39 |              1.33 |                2.03 |
| Ryzen 9 5950X AVX2 (this work)    |             1.72 |              1.06 |                2.39 |

Xeon numbers: team run (`LDLS_Project_1/bonus_mamba3`); Core Ultra numbers: Rupinder's laptop run (his README).


## Full-layer estimate (measured SSM core + Q4_K_M in/out projections at measured read bandwidth)

|   threads | state    | variant      |   proj MB |   proj µs |   core µs |   layer µs |
|----------:|:---------|:-------------|----------:|----------:|----------:|-----------:|
|         1 | layers24 | Mamba-3 SISO |     14.71 |     444.6 |      93.8 |      538.5 |
|         1 | cold     | Mamba-3 SISO |     14.71 |     444.6 |     135.3 |      579.9 |
|         1 | layers24 | MIMO R=4     |     15.59 |     471.4 |     210.7 |      682.1 |
|         1 | cold     | MIMO R=4     |     15.59 |     471.4 |     232.8 |      704.2 |
|         4 | layers24 | Mamba-3 SISO |     14.71 |     355.7 |      57.1 |      412.7 |
|         4 | cold     | Mamba-3 SISO |     14.71 |     355.7 |     113.7 |      469.3 |
|         4 | layers24 | MIMO R=4     |     15.59 |     377   |      77.5 |      454.6 |
|         4 | cold     | MIMO R=4     |     15.59 |     377   |     124.6 |      501.6 |
|        16 | layers24 | Mamba-3 SISO |     14.71 |     345   |       9.8 |      354.8 |
|        16 | cold     | Mamba-3 SISO |     14.71 |     345   |     110.1 |      455   |
|        16 | layers24 | MIMO R=4     |     15.59 |     365.7 |      17.5 |      383.2 |
|        16 | cold     | MIMO R=4     |     15.59 |     365.7 |     116.9 |      482.6 |

## ggml integration (ggml SSM_SCAN baseline + Mamba-3 as a ggml custom op)

| part   | case   | impl             |   threads |   steps |   us_per_step |   rel_err_vs_ref |   max_abs_diff_vs_ours |   direct_us_per_step |
|:-------|:-------|:-----------------|----------:|--------:|--------------:|-----------------:|-----------------------:|---------------------:|
| A      | mamba2 | ggml_ssm_scan    |         1 |    1000 |         92.2  |         5.83e-07 |              nan       |                nan   |
| A      | mamba2 | ours_avx2        |         1 |    1000 |         69.1  |         1.53e-07 |                2.1e-05 |                nan   |
| A      | mamba2 | ggml_ssm_scan    |         8 |    1000 |         16.1  |         5.83e-07 |              nan       |                nan   |
| A      | mamba2 | ours_avx2        |         8 |    1000 |         11    |         1.53e-07 |                2.1e-05 |                nan   |
| A      | mamba2 | ggml_ssm_scan    |        16 |    1000 |        102    |         5.83e-07 |              nan       |                nan   |
| A      | mamba2 | ours_avx2        |        16 |    1000 |          7.63 |         1.53e-07 |                2.1e-05 |                nan   |
| B      | siso   | ggml_custom_avx2 |         1 |    1000 |         65.7  |         2.17e-07 |                0       |                 64.6 |
| B      | siso   | ggml_custom_avx2 |         8 |    1000 |         12.6  |         2.17e-07 |                0       |                 10.6 |
| B      | siso   | ggml_custom_avx2 |        16 |    1000 |         13.4  |         2.17e-07 |                0       |                  9.2 |
| B      | mimo4  | ggml_custom_avx2 |         1 |    1000 |        182    |         1.29e-07 |                0       |                186   |
| B      | mimo4  | ggml_custom_avx2 |         8 |    1000 |         28    |         1.29e-07 |                0       |                 26.1 |
| B      | mimo4  | ggml_custom_avx2 |        16 |    1000 |         20.3  |         1.29e-07 |                0       |                 16.8 |

## Peak memory (RSS, MB, 16 threads, mean of 3 runs; budget 0.5 GB)

| run     | regime    |   mean |   std |
|:--------|:----------|-------:|------:|
| bench   | cold      |  298.1 |   0.1 |
| bench   | hot       |    8.7 |   0.1 |
| bench   | layers24  |   60.9 |   0.1 |
| bench   | model24   |  444.7 |   0.1 |
| ggml_m3 | all_cases |   76.2 |   0.1 |
| verify  | all_cases |  100.3 |   0.2 |
