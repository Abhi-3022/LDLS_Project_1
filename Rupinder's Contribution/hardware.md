# Hardware

| Item | Value |
|---|---|
| Laptop | MSI, Windows 11 Home (build 26200) |
| CPU | Intel Core Ultra 7 155H (Meteor Lake) |
| Cores | 6 P-cores (2 threads each), 8 E-cores, 2 low-power E-cores = 22 logical processors |
| SIMD | AVX2, FMA, F16C (no AVX-512) |
| Caches | L2 18 MB total, L3 24 MB shared |
| RAM | 16 GB (project budget: 8 GB-class laptop, so the 4.5 GB rule from the brief applies) |
| Memory bandwidth | STREAM Triad, 6 threads on the 6 P-cores: **52,103 MB/s** (mean of 51.7, 52.3, 52.3 GB/s) |
| Inference engine | llama.cpp build b12431 (commit a946eb541), CPU backend only, MSVC + OpenMP |

## Which logical processor is which core

From Sysinternals `Coreinfo64.exe -c` (screenshot in the report appendix, A2):

| Logical processors | Core type |
|---|---|
| 0-1, 10-19 | P-cores (two logical processors per core) |
| 2-9 | E-cores |
| 20-21 | Low-power E-cores |

A per-processor test (1B Q4_K_M, prefill 128 tokens, 1 thread pinned to each logical processor in turn)
gave about 73 tok/s on P-cores, 29 on E-cores and 19 on LP E-cores.

## Thread masks used (one thread per physical P-core)

| Threads | Logical processors | Mask |
|---|---|---|
| 1 | 0 | `0x1` |
| 2 | 0, 10 | `0x401` |
| 3 | 0, 10, 12 | `0x1401` |
| 4 | 0, 10, 12, 14 | `0x5401` |
| 5 | 0, 10, 12, 14, 16 | `0x15401` |
| 6 | 0, 10, 12, 14, 16, 18 | `0x55401` |

**On another machine these masks must be changed** (in `scripts/settings.ps1`): run Coreinfo, pick one
logical processor per P-core, and set bit *n* for logical processor *n*.
