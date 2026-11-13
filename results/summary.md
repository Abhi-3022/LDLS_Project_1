# Results summary

All runs: socket 0, physical cores only (core 1 excluded), memory bound to node 0, main thread count T_MAIN = 32. Values are mean ± std over 3 separate processes.

**Hardware:** Intel(R) Xeon(R) Platinum 8480+, 2 sockets x 56 cores (SMT 2), caches {'L1d': '48K', 'L1i': '32K', 'L2': '2048K', 'L3': '107520K'}, RAM 503.5 GB, SIMD: avx2 avx512_bf16 avx512_bitalg avx512_vbmi2 avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma.

## Memory bandwidth (GB/s)

|   threads | read_GBps   | copy_GBps   | triad_GBps   |
|----------:|:------------|:------------|:-------------|
|         1 | 13.6 ± 0.1  | 21.6 ± 0.1  | 14.4 ± 0.0   |
|         2 | 27.0 ± 0.0  | 41.3 ± 0.1  | 28.1 ± 0.2   |
|         4 | 50.7 ± 1.2  | 69.9 ± 2.0  | 49.8 ± 0.2   |
|         8 | 92.7 ± 3.6  | 101.5 ± 2.4 | 78.4 ± 1.8   |
|        16 | 133.7 ± 0.6 | 116.1 ± 0.1 | 94.7 ± 0.2   |
|        24 | 135.0 ± 0.3 | 117.2 ± 0.5 | 98.8 ± 0.1   |
|        32 | 135.7 ± 0.2 | 117.1 ± 0.1 | 108.5 ± 0.1  |
|        48 | 135.7 ± 0.2 | 116.4 ± 0.1 | 122.9 ± 0.0  |
|        55 | 133.9 ± 0.6 | 114.6 ± 0.2 | 121.6 ± 0.6  |

## Bytes read per decoded token (roofline input)

| model        |   params (B) |   active params (B) |   file GB |   bytes/token active (GB) |   bytes/token all experts (GB) |   LM head GB |   KV KiB per ctx token |   experts |   used |
|:-------------|-------------:|--------------------:|----------:|--------------------------:|-------------------------------:|-------------:|-----------------------:|----------:|-------:|
| llama1b-q4   |        1.236 |               1.236 |     0.8   |                     0.8   |                          0.8   |        0.215 |                     32 |         0 |      0 |
| llama1b-q8   |        1.236 |               1.236 |     1.313 |                     1.313 |                          1.313 |        0.279 |                     32 |         0 |      0 |
| llama3b-q4   |        3.213 |               3.213 |     2.012 |                     2.012 |                          2.012 |        0.323 |                    112 |         0 |      0 |
| granite1b-q4 |        1.335 |               0.429 |     0.82  |                     0.272 |                          0.82  |        0.041 |                     48 |        32 |      8 |
| granite1b-q8 |        1.335 |               0.429 |     1.421 |                     0.458 |                          1.421 |        0.053 |                     48 |        32 |      8 |
| granite3b-q4 |        3.299 |               0.883 |     2.015 |                     0.552 |                          2.015 |        0.062 |                     64 |        40 |      8 |

## Correctness: WikiText-2 perplexity (first 100k chars of test, ctx 512)

| model                             |     ppl |   ppl_err |   n_chunks |   maxrss_gb | role           |
|:----------------------------------|--------:|----------:|-----------:|------------:|:---------------|
| llama1b-q4                        | 14.5948 |   0.38621 |         45 |       1.195 | benchmarked    |
| llama1b-q8                        | 14.1191 |   0.37458 |         45 |       1.687 | benchmarked    |
| llama3b-q4                        | 10.6285 |   0.26727 |         45 |       2.461 | benchmarked    |
| granite1b-q4                      |  8.8156 |   0.1904  |         56 |       1.011 | benchmarked    |
| granite1b-q8                      |  8.3646 |   0.17967 |         56 |       1.591 | benchmarked    |
| granite3b-q4                      |  7.715  |   0.17726 |         56 |       2.218 | benchmarked    |
| Llama-3.2-1B-Instruct-f16         | 14.1084 |   0.37419 |         45 |       2.849 | reference_only |
| Llama-3.2-3B-Instruct-f16         | 10.308  |   0.25934 |         45 |       6.859 | reference_only |
| granite-3.1-1b-a400m-instruct-f16 |  8.3294 |   0.17863 |         56 |       2.846 | reference_only |
| granite-3.1-3b-a800m-instruct-f16 |  7.1981 |   0.16217 |         56 |       6.792 | reference_only |

## Thread scaling (tok/s; pp512 prefill, tg128 decode)

|                        | 1           | 2            | 4            | 8             | 16            | 24            | 32            | 48            | 55            |
|:-----------------------|:------------|:-------------|:-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| ('granite1b-q4', 'pp') | 105.0 ± 1.0 | 200.4 ± 12.3 | 349.6 ± 85.2 | 602.5 ± 154.9 | 826.8 ± 10.6  | 1019.2 ± 25.4 | 1276.1 ± 9.4  | 1335.5 ± 1.3  | 1334.1 ± 11.7 |
| ('granite1b-q4', 'tg') | 28.4 ± 0.1  | 56.1 ± 0.1   | 101.2 ± 1.6  | 171.8 ± 2.6   | 223.9 ± 4.8   | 228.8 ± 13.1  | 221.9 ± 4.0   | 183.0 ± 2.9   | 167.2 ± 0.7   |
| ('granite1b-q8', 'pp') | 69.7 ± 0.6  | 136.8 ± 0.3  | 255.0 ± 24.8 | 477.2 ± 49.5  | 826.8 ± 40.2  | 1044.6 ± 1.9  | 1270.7 ± 36.0 | 1549.9 ± 43.8 | 1767.7 ± 48.6 |
| ('granite1b-q8', 'tg') | 17.8 ± 0.0  | 35.3 ± 0.1   | 65.9 ± 0.1   | 111.9 ± 2.8   | 163.2 ± 1.5   | 170.5 ± 7.1   | 169.3 ± 4.9   | 143.5 ± 2.5   | 137.6 ± 0.4   |
| ('granite3b-q4', 'pp') | 50.7 ± 0.1  | 99.1 ± 1.8   | 190.6 ± 1.8  | 338.1 ± 40.4  | 520.5 ± 126.2 | 598.2 ± 62.8  | 676.9 ± 8.8   | 733.6 ± 9.3   | 729.1 ± 15.1  |
| ('granite3b-q4', 'tg') | 13.8 ± 0.0  | 26.9 ± 0.2   | 49.9 ± 0.3   | 86.1 ± 1.1    | 120.1 ± 3.8   | 129.7 ± 1.9   | 130.0 ± 2.9   | 111.7 ± 1.4   | 104.1 ± 1.6   |
| ('llama1b-q4', 'pp')   | 84.0 ± 0.2  | 165.5 ± 5.5  | 300.4 ± 46.8 | 547.3 ± 83.7  | 915.5 ± 22.4  | 1166.9 ± 3.7  | 1458.7 ± 24.4 | 1679.0 ± 4.0  | 1894.9 ± 14.6 |
| ('llama1b-q4', 'tg')   | 9.9 ± 0.0   | 20.2 ± 0.0   | 39.6 ± 0.1   | 71.0 ± 1.0    | 111.4 ± 3.5   | 129.5 ± 0.9   | 134.0 ± 0.9   | 133.0 ± 0.4   | 130.2 ± 2.8   |
| ('llama1b-q8', 'pp')   | 48.4 ± 0.4  | 97.1 ± 0.0   | 187.6 ± 10.0 | 273.5 ± 78.9  | 587.9 ± 104.2 | 682.1 ± 190.4 | 764.4 ± 60.3  | 1067.3 ± 15.5 | 1131.5 ± 8.7  |
| ('llama1b-q8', 'tg')   | 6.4 ± 0.0   | 13.1 ± 0.0   | 25.6 ± 0.0   | 45.6 ± 0.8    | 74.6 ± 0.3    | 87.1 ± 0.7    | 86.5 ± 0.1    | 84.8 ± 0.4    | 83.1 ± 0.8    |
| ('llama3b-q4', 'pp')   | 29.2 ± 0.2  | 58.4 ± 0.4   | 115.6 ± 0.8  | 217.0 ± 8.6   | 356.6 ± 35.8  | 478.8 ± 37.5  | 514.6 ± 27.2  | 679.0 ± 20.7  | 770.1 ± 1.2   |
| ('llama3b-q4', 'tg')   | 4.0 ± 0.0   | 7.9 ± 0.0    | 15.4 ± 0.0   | 27.6 ± 0.7    | 43.4 ± 1.0    | 51.0 ± 0.4    | 54.1 ± 0.6    | 54.0 ± 0.2    | 52.6 ± 1.2    |

Best thread count per model/test:

| model        | test   |   threads | value         |
|:-------------|:-------|----------:|:--------------|
| granite1b-q4 | pp     |        48 | 1335.5 ± 1.3  |
| granite1b-q4 | tg     |        24 | 228.8 ± 13.1  |
| granite1b-q8 | pp     |        55 | 1767.7 ± 48.6 |
| granite1b-q8 | tg     |        24 | 170.5 ± 7.1   |
| granite3b-q4 | pp     |        48 | 733.6 ± 9.3   |
| granite3b-q4 | tg     |        32 | 130.0 ± 2.9   |
| llama1b-q4   | pp     |        55 | 1894.9 ± 14.6 |
| llama1b-q4   | tg     |        32 | 134.0 ± 0.9   |
| llama1b-q8   | pp     |        55 | 1131.5 ± 8.7  |
| llama1b-q8   | tg     |        24 | 87.1 ± 0.7    |
| llama3b-q4   | pp     |        55 | 770.1 ± 1.2   |
| llama3b-q4   | tg     |        32 | 54.1 ± 0.6    |

## Prefill vs prompt length and decode vs context (32 threads)

| model        | test   |   context | value         |   maxrss_gb | over budget   |
|:-------------|:-------|----------:|:--------------|------------:|:--------------|
| granite1b-q4 | pp     |       512 | 1579.8 ± 35.4 |        0.92 |               |
| granite1b-q4 | pp     |      4096 | 1266.1 ± 9.4  |        1.1  |               |
| granite1b-q4 | pp     |     16384 | 713.7 ± 3.2   |        1.72 |               |
| granite1b-q4 | tg     |      1024 | 183.3 ± 0.7   |        1    |               |
| granite1b-q4 | tg     |      8192 | 100.7 ± 0.5   |        1.71 |               |
| granite1b-q4 | tg     |     16384 | 52.0 ± 0.8    |        2.52 |               |
| granite1b-q8 | pp     |       512 | 1375.6 ± 23.3 |        1.51 |               |
| granite1b-q8 | pp     |      4096 | 1149.9 ± 9.5  |        1.69 |               |
| granite1b-q8 | pp     |     16384 | 667.2 ± 4.0   |        2.31 |               |
| granite1b-q8 | tg     |      1024 | 154.8 ± 1.0   |        1.58 |               |
| granite1b-q8 | tg     |      8192 | 88.2 ± 0.5    |        2.29 |               |
| granite1b-q8 | tg     |     16384 | 51.8 ± 1.5    |        3.11 |               |
| granite3b-q4 | pp     |       512 | 847.5 ± 4.5   |        2.16 |               |
| granite3b-q4 | pp     |      4096 | 618.0 ± 5.6   |        2.4  |               |
| granite3b-q4 | pp     |      8192 | 481.9 ± 4.1   |        2.67 |               |
| granite3b-q4 | tg     |      1024 | 112.2 ± 2.7   |        2.25 |               |
| granite3b-q4 | tg     |      8192 | 61.0 ± 1.5    |        3.2  |               |
| llama1b-q4   | pp     |       512 | 1520.2 ± 24.6 |        0.97 |               |
| llama1b-q4   | pp     |      4096 | 1261.5 ± 30.0 |        1.09 |               |
| llama1b-q4   | pp     |     16384 | 611.9 ± 6.9   |        1.51 |               |
| llama1b-q4   | tg     |      1024 | 122.4 ± 0.6   |        1.02 |               |
| llama1b-q4   | tg     |      8192 | 79.8 ± 1.3    |        1.49 |               |
| llama1b-q4   | tg     |     16384 | 55.1 ± 2.3    |        2.04 |               |
| llama1b-q8   | pp     |       512 | 1050.4 ± 19.8 |        1.47 |               |
| llama1b-q8   | pp     |      4096 | 912.3 ± 15.3  |        1.59 |               |
| llama1b-q8   | pp     |     16384 | 479.8 ± 14.7  |        2.01 |               |
| llama1b-q8   | tg     |      1024 | 80.5 ± 0.3    |        1.52 |               |
| llama1b-q8   | tg     |      8192 | 59.6 ± 0.8    |        1.99 |               |
| llama1b-q8   | tg     |     16384 | 45.0 ± 1.1    |        2.54 |               |
| llama3b-q4   | pp     |       512 | 565.6 ± 7.5   |        2.23 |               |
| llama3b-q4   | pp     |      4096 | 436.9 ± 6.9   |        2.65 |               |
| llama3b-q4   | pp     |      8192 | 338.2 ± 3.1   |        3.12 |               |
| llama3b-q4   | tg     |      1024 | 49.2 ± 0.2    |        2.39 |               |
| llama3b-q4   | tg     |      8192 | 29.6 ± 0.0    |        4.05 |               |

Runs with swap activity: 1; runs over the 4.5 GB budget: 0.

## Roofline: predicted vs measured decode

| model        | suite   |   threads |   context |   predicted |   predicted_all_experts |   measured |   efficiency |
|:-------------|:--------|----------:|----------:|------------:|------------------------:|-----------:|-------------:|
| granite1b-q4 | threads |        32 |        64 |       494   |                   164.8 |      221.9 |           45 |
| granite1b-q8 | threads |        32 |        64 |       294.4 |                    95.3 |      169.3 |           58 |
| granite3b-q4 | threads |        32 |        64 |       243.8 |                    67.2 |      130   |           53 |
| llama1b-q4   | threads |        32 |        64 |       169.2 |                   169.2 |      134   |           79 |
| llama1b-q8   | threads |        32 |        64 |       103.2 |                   103.2 |       86.5 |           84 |
| llama3b-q4   | threads |        32 |        64 |        67.2 |                    67.2 |       54.1 |           80 |
| granite1b-q4 | lengths |        32 |       960 |       425.8 |                   156.5 |      183.3 |           43 |
| granite1b-q4 | lengths |        32 |      8128 |       202.2 |                   111.3 |      100.7 |           50 |
| granite1b-q4 | lengths |        32 |     16320 |       126.4 |                    83.7 |       52   |           41 |
| granite1b-q8 | lengths |        32 |       960 |       268.7 |                    92.5 |      154.8 |           58 |
| granite1b-q8 | lengths |        32 |      8128 |       158.3 |                    74.6 |       88.2 |           56 |
| granite1b-q8 | lengths |        32 |     16320 |       107.7 |                    61.1 |       51.8 |           48 |
| granite3b-q4 | lengths |        32 |       960 |       220.6 |                    65.3 |      112.2 |           51 |
| granite3b-q4 | lengths |        32 |      8128 |       125.1 |                    53.3 |       61   |           49 |
| llama1b-q4   | lengths |        32 |       960 |       163.2 |                   163.2 |      122.4 |           75 |
| llama1b-q4   | lengths |        32 |      8128 |       127.3 |                   127.3 |       79.8 |           63 |
| llama1b-q4   | lengths |        32 |     16320 |       101.7 |                   101.7 |       55.1 |           54 |
| llama1b-q8   | lengths |        32 |       960 |       100.9 |                   100.9 |       80.5 |           80 |
| llama1b-q8   | lengths |        32 |      8128 |        85.9 |                    85.9 |       59.6 |           69 |
| llama1b-q8   | lengths |        32 |     16320 |        73.4 |                    73.4 |       45   |           61 |
| llama3b-q4   | lengths |        32 |       960 |        64   |                    64   |       49.2 |           77 |
| llama3b-q4   | lengths |        32 |      8128 |        46.1 |                    46.1 |       29.6 |           64 |

Mean measured/predicted by thread count: 1: 0.59, 2: 0.59, 4: 0.6, 8: 0.57, 16: 0.6, 24: 0.66, 32: 0.61, 48: 0.62, 55: 0.61

## Operator groups (ms per token; sync % = share of the op's wall time spent waiting at its barrier)

| model        | phase   |   context |   threads | group                                  |   ms_per_token |   share % |   sync % |
|:-------------|:--------|----------:|----------:|:---------------------------------------|---------------:|----------:|---------:|
| granite1b-q4 | decode  |      1024 |         1 | KV cache write                         |          0.013 |       0   |        7 |
| granite1b-q4 | decode  |      1024 |         1 | LM head (tied embedding)               |          5.035 |      10.6 |        0 |
| granite1b-q4 | decode  |      1024 |         1 | MoE expert matmuls                     |         22.387 |      47.1 |        0 |
| granite1b-q4 | decode  |      1024 |         1 | MoE routing + combine                  |          0.645 |       1.4 |        1 |
| granite1b-q4 | decode  |      1024 |         1 | attention (QK/softmax/V over KV cache) |         14.141 |      29.7 |        0 |
| granite1b-q4 | decode  |      1024 |         1 | attention projections (QKV, O)         |          4.946 |      10.4 |        0 |
| granite1b-q4 | decode  |      1024 |         1 | norms                                  |          0.061 |       0.1 |        2 |
| granite1b-q4 | decode  |      1024 |         1 | rope / elementwise / other             |          0.131 |       0.3 |        3 |
| granite1b-q4 | decode  |      1024 |        32 | KV cache write                         |          0.093 |       1.8 |       83 |
| granite1b-q4 | decode  |      1024 |        32 | LM head (tied embedding)               |          0.236 |       4.5 |        2 |
| granite1b-q4 | decode  |      1024 |        32 | MoE expert matmuls                     |          1.724 |      32.9 |       14 |
| granite1b-q4 | decode  |      1024 |        32 | MoE routing + combine                  |          0.801 |      15.3 |       60 |
| granite1b-q4 | decode  |      1024 |        32 | attention (QK/softmax/V over KV cache) |          0.8   |      15.3 |        4 |
| granite1b-q4 | decode  |      1024 |        32 | attention projections (QKV, O)         |          0.964 |      18.4 |       16 |
| granite1b-q4 | decode  |      1024 |        32 | norms                                  |          0.094 |       1.8 |       35 |
| granite1b-q4 | decode  |      1024 |        32 | rope / elementwise / other             |          0.355 |       6.8 |       73 |
| granite1b-q4 | decode  |      4096 |        32 | KV cache write                         |          0.101 |       1.4 |       85 |
| granite1b-q4 | decode  |      4096 |        32 | LM head (tied embedding)               |          0.276 |       3.8 |        2 |
| granite1b-q4 | decode  |      4096 |        32 | MoE expert matmuls                     |          1.754 |      24.2 |       12 |
| granite1b-q4 | decode  |      4096 |        32 | MoE routing + combine                  |          0.868 |      12   |       62 |
| granite1b-q4 | decode  |      4096 |        32 | attention (QK/softmax/V over KV cache) |          2.645 |      36.5 |        1 |
| granite1b-q4 | decode  |      4096 |        32 | attention projections (QKV, O)         |          0.938 |      12.9 |       19 |
| granite1b-q4 | decode  |      4096 |        32 | norms                                  |          0.103 |       1.4 |       39 |
| granite1b-q4 | decode  |      4096 |        32 | rope / elementwise / other             |          0.384 |       5.3 |       75 |
| granite1b-q4 | decode  |      8192 |        32 | KV cache write                         |          0.095 |       0.9 |       83 |
| granite1b-q4 | decode  |      8192 |        32 | LM head (tied embedding)               |          0.313 |       3.1 |        2 |
| granite1b-q4 | decode  |      8192 |        32 | MoE expert matmuls                     |          1.823 |      18.2 |       12 |
| granite1b-q4 | decode  |      8192 |        32 | MoE routing + combine                  |          0.83  |       8.3 |       60 |
| granite1b-q4 | decode  |      8192 |        32 | attention (QK/softmax/V over KV cache) |          5.318 |      53.2 |        1 |
| granite1b-q4 | decode  |      8192 |        32 | attention projections (QKV, O)         |          0.963 |       9.6 |       17 |
| granite1b-q4 | decode  |      8192 |        32 | norms                                  |          0.099 |       1   |       34 |
| granite1b-q4 | decode  |      8192 |        32 | rope / elementwise / other             |          0.367 |       3.7 |       73 |
| granite1b-q4 | decode  |     16384 |        32 | KV cache write                         |          0.108 |       0.6 |       83 |
| granite1b-q4 | decode  |     16384 |        32 | LM head (tied embedding)               |          0.323 |       1.7 |        1 |
| granite1b-q4 | decode  |     16384 |        32 | MoE expert matmuls                     |          1.894 |       9.9 |       14 |
| granite1b-q4 | decode  |     16384 |        32 | MoE routing + combine                  |          0.927 |       4.9 |       61 |
| granite1b-q4 | decode  |     16384 |        32 | attention (QK/softmax/V over KV cache) |         14.064 |      73.8 |        0 |
| granite1b-q4 | decode  |     16384 |        32 | attention projections (QKV, O)         |          1.022 |       5.4 |       18 |
| granite1b-q4 | decode  |     16384 |        32 | norms                                  |          0.115 |       0.6 |       37 |
| granite1b-q4 | decode  |     16384 |        32 | rope / elementwise / other             |          0.409 |       2.1 |       74 |
| granite1b-q4 | prefill |       512 |        32 | KV cache write                         |          0.001 |       0.1 |       35 |
| granite1b-q4 | prefill |       512 |        32 | LM head (tied embedding)               |          0.001 |       0.1 |        1 |
| granite1b-q4 | prefill |       512 |        32 | MoE expert matmuls                     |          0.503 |      72.1 |       10 |
| granite1b-q4 | prefill |       512 |        32 | MoE routing + combine                  |          0.028 |       4   |       19 |
| granite1b-q4 | prefill |       512 |        32 | attention (QK/softmax/V over KV cache) |          0.049 |       7.1 |       14 |
| granite1b-q4 | prefill |       512 |        32 | attention projections (QKV, O)         |          0.096 |      13.7 |        4 |
| granite1b-q4 | prefill |       512 |        32 | norms                                  |          0.005 |       0.8 |       13 |
| granite1b-q4 | prefill |       512 |        32 | rope / elementwise / other             |          0.015 |       2.1 |       34 |
| granite1b-q4 | prefill |      4096 |        32 | KV cache write                         |          0.001 |       0.1 |       28 |
| granite1b-q4 | prefill |      4096 |        32 | LM head (tied embedding)               |          0     |       0   |        3 |
| granite1b-q4 | prefill |      4096 |        32 | MoE expert matmuls                     |          0.471 |      57.3 |       12 |
| granite1b-q4 | prefill |      4096 |        32 | MoE routing + combine                  |          0.026 |       3.1 |       24 |
| granite1b-q4 | prefill |      4096 |        32 | attention (QK/softmax/V over KV cache) |          0.228 |      27.7 |       14 |
| granite1b-q4 | prefill |      4096 |        32 | attention projections (QKV, O)         |          0.079 |       9.6 |        3 |
| granite1b-q4 | prefill |      4096 |        32 | norms                                  |          0.005 |       0.6 |       15 |
| granite1b-q4 | prefill |      4096 |        32 | rope / elementwise / other             |          0.012 |       1.5 |       31 |
| granite3b-q4 | decode  |      1024 |        32 | KV cache write                         |          0.146 |       1.6 |       85 |
| granite3b-q4 | decode  |      1024 |        32 | LM head (tied embedding)               |          0.432 |       4.7 |        2 |
| granite3b-q4 | decode  |      1024 |        32 | MoE expert matmuls                     |          3.385 |      36.6 |       14 |
| granite3b-q4 | decode  |      1024 |        32 | MoE routing + combine                  |          1.287 |      13.9 |       56 |
| granite3b-q4 | decode  |      1024 |        32 | attention (QK/softmax/V over KV cache) |          1.482 |      16   |        4 |
| granite3b-q4 | decode  |      1024 |        32 | attention projections (QKV, O)         |          1.557 |      16.8 |       18 |
| granite3b-q4 | decode  |      1024 |        32 | norms                                  |          0.145 |       1.6 |       17 |
| granite3b-q4 | decode  |      1024 |        32 | rope / elementwise / other             |          0.558 |       6   |       74 |
| granite3b-q4 | decode  |      4096 |        32 | KV cache write                         |          0.143 |       1.2 |       85 |
| granite3b-q4 | decode  |      4096 |        32 | LM head (tied embedding)               |          0.422 |       3.4 |        2 |
| granite3b-q4 | decode  |      4096 |        32 | MoE expert matmuls                     |          3.381 |      27.6 |       11 |
| granite3b-q4 | decode  |      4096 |        32 | MoE routing + combine                  |          1.263 |      10.3 |       55 |
| granite3b-q4 | decode  |      4096 |        32 | attention (QK/softmax/V over KV cache) |          4.513 |      36.9 |        1 |
| granite3b-q4 | decode  |      4096 |        32 | attention projections (QKV, O)         |          1.592 |      13   |       17 |
| granite3b-q4 | decode  |      4096 |        32 | norms                                  |          0.142 |       1.2 |       16 |
| granite3b-q4 | decode  |      4096 |        32 | rope / elementwise / other             |          0.548 |       4.5 |       73 |
| granite3b-q4 | decode  |      8192 |        32 | KV cache write                         |          0.14  |       0.8 |       84 |
| granite3b-q4 | decode  |      8192 |        32 | LM head (tied embedding)               |          0.474 |       2.9 |        1 |
| granite3b-q4 | decode  |      8192 |        32 | MoE expert matmuls                     |          3.464 |      20.8 |       11 |
| granite3b-q4 | decode  |      8192 |        32 | MoE routing + combine                  |          1.252 |       7.5 |       54 |
| granite3b-q4 | decode  |      8192 |        32 | attention (QK/softmax/V over KV cache) |          8.739 |      52.5 |        1 |
| granite3b-q4 | decode  |      8192 |        32 | attention projections (QKV, O)         |          1.635 |       9.8 |       16 |
| granite3b-q4 | decode  |      8192 |        32 | norms                                  |          0.148 |       0.9 |       18 |
| granite3b-q4 | decode  |      8192 |        32 | rope / elementwise / other             |          0.54  |       3.2 |       71 |
| granite3b-q4 | prefill |       512 |        32 | KV cache write                         |          0.001 |       0.1 |       34 |
| granite3b-q4 | prefill |       512 |        32 | LM head (tied embedding)               |          0.001 |       0.1 |        1 |
| granite3b-q4 | prefill |       512 |        32 | MoE expert matmuls                     |          1.182 |      72.3 |       32 |
| granite3b-q4 | prefill |       512 |        32 | MoE routing + combine                  |          0.067 |       4.1 |       33 |
| granite3b-q4 | prefill |       512 |        32 | attention (QK/softmax/V over KV cache) |          0.121 |       7.4 |       18 |
| granite3b-q4 | prefill |       512 |        32 | attention projections (QKV, O)         |          0.227 |      13.9 |       14 |
| granite3b-q4 | prefill |       512 |        32 | norms                                  |          0.011 |       0.7 |       25 |
| granite3b-q4 | prefill |       512 |        32 | rope / elementwise / other             |          0.022 |       1.3 |       44 |
| granite3b-q4 | prefill |      4096 |        32 | KV cache write                         |          0.001 |       0.1 |       29 |
| granite3b-q4 | prefill |      4096 |        32 | LM head (tied embedding)               |          0     |       0   |        2 |
| granite3b-q4 | prefill |      4096 |        32 | MoE expert matmuls                     |          0.909 |      53.5 |       16 |
| granite3b-q4 | prefill |      4096 |        32 | MoE routing + combine                  |          0.052 |       3.1 |       20 |
| granite3b-q4 | prefill |      4096 |        32 | attention (QK/softmax/V over KV cache) |          0.531 |      31.4 |       10 |
| granite3b-q4 | prefill |      4096 |        32 | attention projections (QKV, O)         |          0.176 |      10.4 |       10 |
| granite3b-q4 | prefill |      4096 |        32 | norms                                  |          0.01  |       0.6 |       12 |
| granite3b-q4 | prefill |      4096 |        32 | rope / elementwise / other             |          0.015 |       0.9 |       30 |
| llama1b-q4   | decode  |      1024 |         1 | KV cache write                         |          0.01  |       0   |        6 |
| llama1b-q4   | decode  |      1024 |         1 | LM head (tied embedding)               |         26.265 |      22.6 |        0 |
| llama1b-q4   | decode  |      1024 |         1 | attention (QK/softmax/V over KV cache) |         17.225 |      14.8 |        0 |
| llama1b-q4   | decode  |      1024 |         1 | attention projections (QKV, O)         |         12.25  |      10.5 |        0 |
| llama1b-q4   | decode  |      1024 |         1 | dense FFN matmuls                      |         60.193 |      51.8 |        0 |
| llama1b-q4   | decode  |      1024 |         1 | norms                                  |          0.074 |       0.1 |        1 |
| llama1b-q4   | decode  |      1024 |         1 | rope / elementwise / other             |          0.13  |       0.1 |        1 |
| llama1b-q4   | decode  |      1024 |        32 | KV cache write                         |          0.065 |       0.8 |       83 |
| llama1b-q4   | decode  |      1024 |        32 | LM head (tied embedding)               |          1.275 |      16   |        1 |
| llama1b-q4   | decode  |      1024 |        32 | attention (QK/softmax/V over KV cache) |          0.927 |      11.7 |        4 |
| llama1b-q4   | decode  |      1024 |        32 | attention projections (QKV, O)         |          1.129 |      14.2 |       14 |
| llama1b-q4   | decode  |      1024 |        32 | dense FFN matmuls                      |          4.169 |      52.5 |       11 |
| llama1b-q4   | decode  |      1024 |        32 | norms                                  |          0.083 |       1   |       12 |
| llama1b-q4   | decode  |      1024 |        32 | rope / elementwise / other             |          0.242 |       3   |       34 |
| llama1b-q4   | decode  |      4096 |        32 | KV cache write                         |          0.062 |       0.6 |       83 |
| llama1b-q4   | decode  |      4096 |        32 | LM head (tied embedding)               |          1.386 |      13.9 |        1 |
| llama1b-q4   | decode  |      4096 |        32 | attention (QK/softmax/V over KV cache) |          2.824 |      28.4 |        1 |
| llama1b-q4   | decode  |      4096 |        32 | attention projections (QKV, O)         |          1.118 |      11.2 |       14 |
| llama1b-q4   | decode  |      4096 |        32 | dense FFN matmuls                      |          4.181 |      42   |       11 |
| llama1b-q4   | decode  |      4096 |        32 | norms                                  |          0.083 |       0.8 |       11 |
| llama1b-q4   | decode  |      4096 |        32 | rope / elementwise / other             |          0.237 |       2.4 |       32 |
| llama1b-q4   | decode  |      8192 |        32 | KV cache write                         |          0.066 |       0.5 |       83 |
| llama1b-q4   | decode  |      8192 |        32 | LM head (tied embedding)               |          1.557 |      12.5 |        1 |
| llama1b-q4   | decode  |      8192 |        32 | attention (QK/softmax/V over KV cache) |          5.056 |      40.5 |        1 |
| llama1b-q4   | decode  |      8192 |        32 | attention projections (QKV, O)         |          1.145 |       9.2 |       13 |
| llama1b-q4   | decode  |      8192 |        32 | dense FFN matmuls                      |          4.279 |      34.2 |       10 |
| llama1b-q4   | decode  |      8192 |        32 | norms                                  |          0.084 |       0.7 |       12 |
| llama1b-q4   | decode  |      8192 |        32 | rope / elementwise / other             |          0.247 |       2   |       34 |
| llama1b-q4   | decode  |     16384 |        32 | KV cache write                         |          0.065 |       0.4 |       81 |
| llama1b-q4   | decode  |     16384 |        32 | LM head (tied embedding)               |          1.652 |       8.9 |        1 |
| llama1b-q4   | decode  |     16384 |        32 | attention (QK/softmax/V over KV cache) |         10.977 |      59.2 |        0 |
| llama1b-q4   | decode  |     16384 |        32 | attention projections (QKV, O)         |          1.144 |       6.2 |       13 |
| llama1b-q4   | decode  |     16384 |        32 | dense FFN matmuls                      |          4.314 |      23.3 |        9 |
| llama1b-q4   | decode  |     16384 |        32 | norms                                  |          0.087 |       0.5 |       11 |
| llama1b-q4   | decode  |     16384 |        32 | rope / elementwise / other             |          0.249 |       1.3 |       32 |
| llama1b-q4   | prefill |       512 |        32 | KV cache write                         |          0.001 |       0.1 |       33 |
| llama1b-q4   | prefill |       512 |        32 | LM head (tied embedding)               |          0.003 |       0.5 |        1 |
| llama1b-q4   | prefill |       512 |        32 | attention (QK/softmax/V over KV cache) |          0.06  |       7.9 |       17 |
| llama1b-q4   | prefill |       512 |        32 | attention projections (QKV, O)         |          0.153 |      20   |       15 |
| llama1b-q4   | prefill |       512 |        32 | dense FFN matmuls                      |          0.522 |      68.5 |       15 |
| llama1b-q4   | prefill |       512 |        32 | norms                                  |          0.007 |       1   |       25 |
| llama1b-q4   | prefill |       512 |        32 | rope / elementwise / other             |          0.016 |       2.1 |       33 |
| llama1b-q4   | prefill |      4096 |        32 | KV cache write                         |          0.001 |       0.1 |       33 |
| llama1b-q4   | prefill |      4096 |        32 | LM head (tied embedding)               |          0.001 |       0.1 |        1 |
| llama1b-q4   | prefill |      4096 |        32 | attention (QK/softmax/V over KV cache) |          0.292 |      34   |       11 |
| llama1b-q4   | prefill |      4096 |        32 | attention projections (QKV, O)         |          0.122 |      14.1 |        6 |
| llama1b-q4   | prefill |      4096 |        32 | dense FFN matmuls                      |          0.427 |      49.7 |        8 |
| llama1b-q4   | prefill |      4096 |        32 | norms                                  |          0.006 |       0.7 |       14 |
| llama1b-q4   | prefill |      4096 |        32 | rope / elementwise / other             |          0.01  |       1.2 |       27 |
| llama3b-q4   | decode  |      1024 |        32 | KV cache write                         |          0.104 |       0.5 |       69 |
| llama3b-q4   | decode  |      1024 |        32 | LM head (tied embedding)               |          2.372 |      11.8 |        1 |
| llama3b-q4   | decode  |      1024 |        32 | attention (QK/softmax/V over KV cache) |          1.957 |       9.8 |        3 |
| llama3b-q4   | decode  |      1024 |        32 | attention projections (QKV, O)         |          3.962 |      19.8 |       10 |
| llama3b-q4   | decode  |      1024 |        32 | dense FFN matmuls                      |         10.891 |      54.4 |        9 |
| llama3b-q4   | decode  |      1024 |        32 | norms                                  |          0.198 |       1   |        9 |
| llama3b-q4   | decode  |      1024 |        32 | rope / elementwise / other             |          0.436 |       2.2 |       22 |
| llama3b-q4   | decode  |      4096 |        32 | KV cache write                         |          0.114 |       0.5 |       71 |
| llama3b-q4   | decode  |      4096 |        32 | LM head (tied embedding)               |          2.467 |      10   |        1 |
| llama3b-q4   | decode  |      4096 |        32 | attention (QK/softmax/V over KV cache) |          6.435 |      26   |        1 |
| llama3b-q4   | decode  |      4096 |        32 | attention projections (QKV, O)         |          4.021 |      16.3 |        9 |
| llama3b-q4   | decode  |      4096 |        32 | dense FFN matmuls                      |         10.905 |      44.1 |        7 |
| llama3b-q4   | decode  |      4096 |        32 | norms                                  |          0.202 |       0.8 |        9 |
| llama3b-q4   | decode  |      4096 |        32 | rope / elementwise / other             |          0.453 |       1.8 |       25 |
| llama3b-q4   | decode  |      8192 |        32 | KV cache write                         |          0.125 |       0.4 |       73 |
| llama3b-q4   | decode  |      8192 |        32 | LM head (tied embedding)               |          2.477 |       7.3 |        0 |
| llama3b-q4   | decode  |      8192 |        32 | attention (QK/softmax/V over KV cache) |         15.138 |      44.8 |        0 |
| llama3b-q4   | decode  |      8192 |        32 | attention projections (QKV, O)         |          4.1   |      12.1 |       10 |
| llama3b-q4   | decode  |      8192 |        32 | dense FFN matmuls                      |         11.165 |      33   |        8 |
| llama3b-q4   | decode  |      8192 |        32 | norms                                  |          0.209 |       0.6 |        9 |
| llama3b-q4   | decode  |      8192 |        32 | rope / elementwise / other             |          0.471 |       1.4 |       27 |
| llama3b-q4   | prefill |       512 |        32 | KV cache write                         |          0.002 |       0.1 |       17 |
| llama3b-q4   | prefill |       512 |        32 | LM head (tied embedding)               |          0.005 |       0.2 |        0 |
| llama3b-q4   | prefill |       512 |        32 | attention (QK/softmax/V over KV cache) |          0.153 |       7   |       17 |
| llama3b-q4   | prefill |       512 |        32 | attention projections (QKV, O)         |          0.605 |      27.6 |       15 |
| llama3b-q4   | prefill |       512 |        32 | dense FFN matmuls                      |          1.373 |      62.7 |       14 |
| llama3b-q4   | prefill |       512 |        32 | norms                                  |          0.02  |       0.9 |       24 |
| llama3b-q4   | prefill |       512 |        32 | rope / elementwise / other             |          0.031 |       1.4 |       31 |
| llama3b-q4   | prefill |      4096 |        32 | KV cache write                         |          0.002 |       0.1 |       19 |
| llama3b-q4   | prefill |      4096 |        32 | LM head (tied embedding)               |          0.001 |       0.1 |        1 |
| llama3b-q4   | prefill |      4096 |        32 | attention (QK/softmax/V over KV cache) |          0.715 |      29.5 |       12 |
| llama3b-q4   | prefill |      4096 |        32 | attention projections (QKV, O)         |          0.502 |      20.7 |        5 |
| llama3b-q4   | prefill |      4096 |        32 | dense FFN matmuls                      |          1.159 |      47.8 |        7 |
| llama3b-q4   | prefill |      4096 |        32 | norms                                  |          0.018 |       0.7 |        6 |
| llama3b-q4   | prefill |      4096 |        32 | rope / elementwise / other             |          0.026 |       1.1 |       18 |

### Expert routing cost

| model        | phase   |   context |   threads |   ms_per_token |   share % |
|:-------------|:--------|----------:|----------:|---------------:|----------:|
| granite1b-q4 | decode  |      1024 |         1 |          0.645 |       1.4 |
| granite1b-q4 | decode  |      1024 |        32 |          0.801 |      15.3 |
| granite1b-q4 | decode  |      4096 |        32 |          0.868 |      12   |
| granite1b-q4 | decode  |      8192 |        32 |          0.83  |       8.3 |
| granite1b-q4 | decode  |     16384 |        32 |          0.927 |       4.9 |
| granite1b-q4 | prefill |       512 |        32 |          0.028 |       4   |
| granite1b-q4 | prefill |      4096 |        32 |          0.026 |       3.1 |
| granite3b-q4 | decode  |      1024 |        32 |          1.287 |      13.9 |
| granite3b-q4 | decode  |      4096 |        32 |          1.263 |      10.3 |
| granite3b-q4 | decode  |      8192 |        32 |          1.252 |       7.5 |
| granite3b-q4 | prefill |       512 |        32 |          0.067 |       4.1 |
| granite3b-q4 | prefill |      4096 |        32 |          0.052 |       3.1 |

## Top-3 operators per phase

| model        | phase   |   context | op             | name          |   ms_per_token |   share % |
|:-------------|:--------|----------:|:---------------|:--------------|---------------:|----------:|
| granite1b-q4 | decode  |      1024 | FLASH_ATTN_EXT | node          |          0.8   |      15.3 |
| granite1b-q4 | decode  |      1024 | MUL_MAT_ID     | ffn_moe_down  |          0.654 |      12.5 |
| granite1b-q4 | decode  |      1024 | MUL_MAT_ID     | ffn_moe_gate  |          0.538 |      10.3 |
| granite1b-q4 | decode  |      4096 | FLASH_ATTN_EXT | node          |          2.645 |      36.5 |
| granite1b-q4 | decode  |      4096 | MUL_MAT_ID     | ffn_moe_down  |          0.677 |       9.4 |
| granite1b-q4 | decode  |      4096 | MUL_MAT_ID     | ffn_moe_up    |          0.541 |       7.5 |
| granite1b-q4 | decode  |      8192 | FLASH_ATTN_EXT | node          |          5.318 |      53.2 |
| granite1b-q4 | decode  |      8192 | MUL_MAT_ID     | ffn_moe_down  |          0.701 |       7   |
| granite1b-q4 | decode  |      8192 | MUL_MAT_ID     | ffn_moe_gate  |          0.564 |       5.6 |
| granite1b-q4 | decode  |     16384 | FLASH_ATTN_EXT | node          |         14.064 |      73.8 |
| granite1b-q4 | decode  |     16384 | MUL_MAT_ID     | ffn_moe_down  |          0.712 |       3.7 |
| granite1b-q4 | decode  |     16384 | MUL_MAT_ID     | ffn_moe_gate  |          0.613 |       3.2 |
| granite1b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_down  |          0.172 |      24.6 |
| granite1b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_gate  |          0.169 |      24.3 |
| granite1b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_up    |          0.162 |      23.2 |
| granite1b-q4 | prefill |      4096 | FLASH_ATTN_EXT | node          |          0.228 |      27.7 |
| granite1b-q4 | prefill |      4096 | MUL_MAT_ID     | ffn_moe_down  |          0.17  |      20.7 |
| granite1b-q4 | prefill |      4096 | MUL_MAT_ID     | ffn_moe_gate  |          0.156 |      19   |
| granite3b-q4 | decode  |      1024 | FLASH_ATTN_EXT | node          |          1.482 |      16   |
| granite3b-q4 | decode  |      1024 | MUL_MAT_ID     | ffn_moe_down  |          1.277 |      13.8 |
| granite3b-q4 | decode  |      1024 | MUL_MAT_ID     | ffn_moe_gate  |          1.091 |      11.8 |
| granite3b-q4 | decode  |      4096 | FLASH_ATTN_EXT | node          |          4.513 |      36.9 |
| granite3b-q4 | decode  |      4096 | MUL_MAT_ID     | ffn_moe_down  |          1.292 |      10.6 |
| granite3b-q4 | decode  |      4096 | MUL_MAT_ID     | ffn_moe_gate  |          1.059 |       8.7 |
| granite3b-q4 | decode  |      8192 | FLASH_ATTN_EXT | node          |          8.739 |      52.5 |
| granite3b-q4 | decode  |      8192 | MUL_MAT_ID     | ffn_moe_down  |          1.326 |       8   |
| granite3b-q4 | decode  |      8192 | MUL_MAT_ID     | ffn_moe_gate  |          1.087 |       6.5 |
| granite3b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_gate  |          0.404 |      24.7 |
| granite3b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_up    |          0.402 |      24.6 |
| granite3b-q4 | prefill |       512 | MUL_MAT_ID     | ffn_moe_down  |          0.376 |      23   |
| granite3b-q4 | prefill |      4096 | FLASH_ATTN_EXT | node          |          0.531 |      31.4 |
| granite3b-q4 | prefill |      4096 | MUL_MAT_ID     | ffn_moe_gate  |          0.322 |      19   |
| granite3b-q4 | prefill |      4096 | MUL_MAT_ID     | ffn_moe_up    |          0.305 |      18   |
| llama1b-q4   | decode  |      1024 | MUL_MAT        | ffn_out       |          1.643 |      20.7 |
| llama1b-q4   | decode  |      1024 | MUL_MAT        | result_output |          1.275 |      16   |
| llama1b-q4   | decode  |      1024 | MUL_MAT        | ffn_gate      |          1.274 |      16   |
| llama1b-q4   | decode  |      4096 | FLASH_ATTN_EXT | node          |          2.824 |      28.4 |
| llama1b-q4   | decode  |      4096 | MUL_MAT        | ffn_out       |          1.658 |      16.7 |
| llama1b-q4   | decode  |      4096 | MUL_MAT        | result_output |          1.386 |      13.9 |
| llama1b-q4   | decode  |      8192 | FLASH_ATTN_EXT | node          |          5.056 |      40.5 |
| llama1b-q4   | decode  |      8192 | MUL_MAT        | ffn_out       |          1.705 |      13.6 |
| llama1b-q4   | decode  |      8192 | MUL_MAT        | result_output |          1.557 |      12.5 |
| llama1b-q4   | decode  |     16384 | FLASH_ATTN_EXT | node          |         10.977 |      59.2 |
| llama1b-q4   | decode  |     16384 | MUL_MAT        | ffn_out       |          1.727 |       9.3 |
| llama1b-q4   | decode  |     16384 | MUL_MAT        | result_output |          1.652 |       8.9 |
| llama1b-q4   | prefill |       512 | MUL_MAT        | ffn_gate      |          0.189 |      24.8 |
| llama1b-q4   | prefill |       512 | MUL_MAT        | ffn_up        |          0.18  |      23.7 |
| llama1b-q4   | prefill |       512 | MUL_MAT        | ffn_out       |          0.153 |      20   |
| llama1b-q4   | prefill |      4096 | FLASH_ATTN_EXT | node          |          0.292 |      34   |
| llama1b-q4   | prefill |      4096 | MUL_MAT        | ffn_gate      |          0.15  |      17.5 |
| llama1b-q4   | prefill |      4096 | MUL_MAT        | ffn_up        |          0.139 |      16.2 |
| llama3b-q4   | decode  |      1024 | MUL_MAT        | ffn_out       |          4.273 |      21.3 |
| llama3b-q4   | decode  |      1024 | MUL_MAT        | ffn_gate      |          3.321 |      16.6 |
| llama3b-q4   | decode  |      1024 | MUL_MAT        | ffn_up        |          3.297 |      16.5 |
| llama3b-q4   | decode  |      4096 | FLASH_ATTN_EXT | node          |          6.435 |      26   |
| llama3b-q4   | decode  |      4096 | MUL_MAT        | ffn_out       |          4.283 |      17.3 |
| llama3b-q4   | decode  |      4096 | MUL_MAT        | ffn_gate      |          3.323 |      13.4 |
| llama3b-q4   | decode  |      8192 | FLASH_ATTN_EXT | node          |         15.138 |      44.8 |
| llama3b-q4   | decode  |      8192 | MUL_MAT        | ffn_out       |          4.41  |      13   |
| llama3b-q4   | decode  |      8192 | MUL_MAT        | ffn_gate      |          3.388 |      10   |
| llama3b-q4   | prefill |       512 | MUL_MAT        | ffn_gate      |          0.466 |      21.3 |
| llama3b-q4   | prefill |       512 | MUL_MAT        | ffn_up        |          0.455 |      20.8 |
| llama3b-q4   | prefill |       512 | MUL_MAT        | ffn_out       |          0.452 |      20.6 |
| llama3b-q4   | prefill |      4096 | FLASH_ATTN_EXT | node          |          0.715 |      29.5 |
| llama3b-q4   | prefill |      4096 | MUL_MAT        | ffn_gate      |          0.405 |      16.7 |
| llama3b-q4   | prefill |      4096 | MUL_MAT        | ffn_out       |          0.382 |      15.7 |

## Own experiment: experts per token k (tok/s)

| model            | ('pp', 0)     | ('tg', 0)    | ('tg', 896)   |
|:-----------------|:--------------|:-------------|:--------------|
| granite1b-q4-k1  | 2632.3 ± 16.6 | 278.9 ± 8.3  | 234.4 ± 13.4  |
| granite1b-q4-k16 | 743.4 ± 27.6  | 150.2 ± 4.4  | 135.2 ± 4.4   |
| granite1b-q4-k2  | 2556.4 ± 12.5 | 301.5 ± 10.7 | 241.5 ± 8.5   |
| granite1b-q4-k32 | 397.0 ± 6.0   | 91.8 ± 2.7   | 82.7 ± 2.4    |
| granite1b-q4-k4  | 1889.2 ± 4.3  | 259.8 ± 5.1  | 211.4 ± 7.4   |
| granite1b-q4-k8  | 1239.9 ± 4.3  | 201.1 ± 2.6  | 175.3 ± 1.2   |
| granite3b-q4-k1  | 1521.4 ± 11.3 | 212.2 ± 4.1  | 167.3 ± 2.8   |
| granite3b-q4-k16 | 401.5 ± 0.3   | 87.0 ± 2.6   | 80.1 ± 0.2    |
| granite3b-q4-k2  | 1384.3 ± 15.2 | 195.4 ± 4.5  | 154.7 ± 6.4   |
| granite3b-q4-k32 | 215.1 ± 1.6   | 52.3 ± 1.3   | 50.2 ± 1.4    |
| granite3b-q4-k4  | 1050.0 ± 5.9  | 173.5 ± 5.3  | 138.8 ± 3.8   |
| granite3b-q4-k40 | 225.3 ± 1.4   | 44.8 ± 1.0   | 42.0 ± 0.7    |
| granite3b-q4-k8  | 684.9 ± 11.1  | 126.3 ± 1.2  | 110.4 ± 3.3   |

| model            |   k |         ppl |    ppl_err |
|:-----------------|----:|------------:|-----------:|
| granite1b-q4-k1  |   1 | 144323      | 4269.03    |
| granite1b-q4-k2  |   2 |   3956.14   |  123.774   |
| granite1b-q4-k4  |   4 |     22.3404 |    0.53592 |
| granite1b-q4-k8  |   8 |      8.8156 |    0.1904  |
| granite1b-q4-k16 |  16 |     11.1434 |    0.24614 |
| granite1b-q4-k32 |  32 |     25.7696 |    0.6019  |
| granite3b-q4-k1  |   1 |  82937.2    | 2176.92    |
| granite3b-q4-k2  |   2 |   3046.2    |   93.066   |
| granite3b-q4-k4  |   4 |     15.8027 |    0.37673 |
| granite3b-q4-k8  |   8 |      7.715  |    0.17726 |
| granite3b-q4-k16 |  16 |      9.6517 |    0.21933 |
| granite3b-q4-k32 |  32 |     34.0486 |    0.87885 |
| granite3b-q4-k40 |  40 |     45.2063 |    1.17784 |

## llama.cpp vs ik_llama.cpp (tok/s)

|                        | ik             | ik (rtr)       | llama         |
|:-----------------------|:---------------|:---------------|:--------------|
| ('granite1b-q4', 'pp') | 3614.2 ± 65.0  | 4004.3 ± 56.4  | 1594.5 ± 25.1 |
| ('granite1b-q4', 'tg') | 293.7 ± 10.4   | 281.9 ± 12.4   | 210.8 ± 14.0  |
| ('granite3b-q4', 'pp') | 2110.9 ± 23.3  | 2329.3 ± 130.2 | 686.0 ± 0.8   |
| ('granite3b-q4', 'tg') | 160.2 ± 1.4    | 155.0 ± 2.4    | 130.9 ± 3.1   |
| ('llama1b-q4', 'pp')   | 2199.9 ± 104.5 | 2959.9 ± 479.6 | 1449.9 ± 1.7  |
| ('llama1b-q4', 'tg')   | 133.1 ± 4.3    | 139.1 ± 1.6    | 135.3 ± 0.8   |
| ('llama3b-q4', 'pp')   | 843.5 ± 16.7   | 1563.4 ± 13.3  | 543.0 ± 34.5  |
| ('llama3b-q4', 'tg')   | 52.2 ± 0.5     | 55.3 ± 0.7     | 53.4 ± 0.1    |

## Shared reference test (reference_precision.csv)

| host      | test   |   threads | value         |
|:----------|:-------|----------:|:--------------|
| precision | pp     |        32 | 1784.3 ± 66.3 |
| precision | pp     |        55 | 1872.5 ± 2.5  |
| precision | tg     |        32 | 129.0 ± 0.8   |
| precision | tg     |        55 | 123.7 ± 0.3   |

## Bonus
See bonus_mamba3/results/summary.md.
