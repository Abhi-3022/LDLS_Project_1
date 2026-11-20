# Bonus: Mamba-3 decode step on CPU

One Mamba-3 decode step (one token, one layer) in C++ for x86 CPUs: SISO and MIMO (rank 4), each in
plain C++ and AVX2, plus a Mamba-2 step at the same state size. Every kernel is checked against the
official PyTorch reference over 1,000 decode steps.

## What one decode step computes

Per head h (P = head dim, N = state size, R = MIMO rank; SISO is R = 1):

```
angle    += tanh(theta) * dt * pi                          # data-dependent RoPE = the complex-valued state
B_r, C_r  = rotate(B_r + Bbias, angle), rotate(C_r + Cbias, angle)
alpha = exp(A dt),  beta = (1 - trap) dt alpha,  gamma = trap dt        # exponential-trapezoidal rule
S[p,n] = alpha S[p,n] + sum_r gamma xp_r[p] x[p] B_r[n]          # current input
                      + sum_r beta  xp_r[p] x_prev[p] B_prev_r[n] # previous input (the trapezoid term)
y[p]   = sum_r op_r[p] silu(zp_r[p] z[p]) (sum_n S[p,n] C_r[n] + D xp_r[p] x[p])
```

Mamba-2 (exponential-Euler): `S = exp(A dt) S + dt x B^T`, `y = (S C + D x) silu(z)`.
The kernels start from the outputs of `in_proj` (and its activations); `in_proj`, `out_proj` and the
B/C RMSNorm are ordinary matmuls and are not part of the step.

## Files

| File | Purpose |
|---|---|
| `ref/mamba3_reference.py` | Loads the official reference functions from `state-spaces/mamba` (commit `e9594ce`), runs 1,000 SISO, MIMO and Mamba-2 steps in float64 on CPU, writes `data/case.bin` |
| `src/m3.h` | Data layout, RoPE, FLOP/byte model |
| `src/plain.cpp` | Plain C++ kernels (built with auto-vectorisation off) |
| `src/avx2.cpp` | AVX2 + FMA kernels |
| `src/verify.cpp` | Runs every kernel over all steps and compares each step to the float64 reference |
| `src/bench.cpp` | Time per step, FLOPs, bytes, arithmetic intensity |
| `plot_bench.py` | Plots benchmark CSVs |

Official functions used (loaded from source, not copied): `apply_rotary_qk_inference_reference`
(`ops/triton/mamba3/mamba3_mimo_rotary_step.py`), `selective_state_update_fused_ref_v2`
(`ops/cute/mamba3/mamba3_step_fn.py`), `selective_state_update_ref` (`ops/triton/selective_state_update.py`).
The module's own `Mamba3.step()` calls CUDA-only kernels, so it cannot run on a CPU.

## Correctness criterion

Per step: `max_p |y - y_ref| / max_p |y_ref| <= 1e-4`, and the same for the final state after 1,000 steps.
The kernels compute in float32; the reference is float64. Random weights are used (correctness and
speed do not depend on trained weights).

## Running on Windows (PowerShell)

```powershell
# one-time
pip install torch einops pandas matplotlib
git clone https://github.com/state-spaces/mamba $HOME\mamba-official
git -C $HOME\mamba-official checkout e9594ce1c732d97440f0332fdc43170a2294dbfa

cd $HOME\bonus_mamba3
mkdir data, build, results -Force | Out-Null

# 1. reference (about 10 s)
python ref\mamba3_reference.py --repo $HOME\mamba-official --out data\case.bin

# 2. build with the 64-bit WinLibs gcc (put it ahead of the old C:\MinGW in PATH)
$env:PATH = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin;$env:PATH"
g++ -O2 -std=c++17 -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize -c src/plain.cpp -o build/plain.o
g++ -O3 -std=c++17 -fopenmp -mavx2 -mfma -c src/avx2.cpp -o build/avx2.o
g++ -O2 -std=c++17 -fopenmp -static src/verify.cpp build/plain.o build/avx2.o -o build/verify.exe
g++ -O2 -std=c++17 -fopenmp -static src/bench.cpp build/plain.o build/avx2.o -o build/bench.exe

# 3. verify all 6 kernels over 1,000 steps
.\build\verify.exe data\case.bin 1e-4

# 4. benchmark, pinned to P-cores: 1 thread (processor 10) and 6 threads (one per P-core)
function RunPinned($threads, $mask, $out) {
  $env:OMP_NUM_THREADS = $threads
  $p = Start-Process .\build\bench.exe -NoNewWindow -PassThru -RedirectStandardOutput $out
  $p.ProcessorAffinity = [IntPtr]$mask
  $p.WaitForExit()
}
RunPinned 1 0x400   results\bench_t1.csv
RunPinned 6 0x55401 results\bench_t6.csv

# 5. plot
python plot_bench.py results\bench_t1.csv results\bench_t6.csv --out results\mamba3_bench.png
```

Benchmark settings: H = 64 heads, P = 64, N = 128, the layer shape of `state-spaces/mamba3-siso-1.5b`
and `mamba3-mimo-1.5b` (2 MiB of state per layer). Two regimes:

* `layers = 1`: the same layer's state every step, so it stays in cache.
* `layers = 24`: the steps cycle through 24 layers (48 MiB, twice the 24 MB L3), so each step streams its
  state from DRAM, as in real decoding of a 24-layer model.

Each configuration is timed 3 times for at least 1 s each; the CSV has the mean, min and max.
