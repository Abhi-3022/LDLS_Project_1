# Bonus: Mamba-3 decode step on CPU

One Mamba-3 decode step (one token, one layer) in C++ for x86 CPUs: SISO and MIMO (rank 4), each in
plain C++ and AVX2, plus a Mamba-2 step at the same state size. Every kernel is checked against the
official PyTorch reference over 1,000 decode steps. As the stretch goal, the same kernels also run as
a ggml operator inside a small ggml graph.

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
| `src/ggml_m3op.cpp` | Stretch goal: the kernels wrapped as a ggml operator and tested in a small ggml graph |
| `plot_bench.py` | Plots benchmark CSVs |
| `run_bonus.ps1` | Runs every step below in order and saves each output to `results/` |

Official functions used (loaded from source, not copied): `apply_rotary_qk_inference_reference`
(`ops/triton/mamba3/mamba3_mimo_rotary_step.py`), `selective_state_update_fused_ref_v2`
(`ops/cute/mamba3/mamba3_step_fn.py`), `selective_state_update_ref` (`ops/triton/selective_state_update.py`).
The module's own `Mamba3.step()` calls CUDA-only kernels, so it cannot run on a CPU.

## Correctness criterion

Per step: `max_p |y - y_ref| / max_p |y_ref| <= 1e-4`, and the same for the final state after 1,000 steps.
The kernels compute in float32; the reference is float64. Two cases are checked: `data/case.bin`
(H = 16 heads) and `data/case_h64.bin` (H = 64, the benchmark shape, made with `--heads 64`); the logs are
`results/verify_output.txt` and `results/verify_output_h64.txt`. The case files are not byte-identical
across machines (PyTorch's CPU random generator differs between versions and CPUs), but each file holds
its own inputs and reference outputs, so the check is the same.

Pretrained checkpoints are public: `state-spaces/mamba3-siso-1.5b` and `state-spaces/mamba3-mimo-1.5b`
on Hugging Face (BF16; 24 layers, 64 heads of dim 64, d_state 128, 1 B/C group). They are used here only
for the layer shape. All tests use random weights, which the brief allows for correctness and speed;
perplexity, which would need the trained weights and a full model, is not measured.

## Answer: does MIMO's extra compute come at little cost on this CPU?

Partly. MIMO (rank 4) does 3.58x the arithmetic of SISO per decode step (13.53M vs 3.78M FLOPs,
counted with the FLOP model `step_flops` in `src/m3.h`, not hardware counters; same 2 MiB state),
but with the AVX2 kernels its measured time is only 1.33x-2.22x SISO's (the plain C++ kernels,
compute-bound throughout, show 3.2x-4.4x), never reaching the "free" 1.0x line a GPU gets close to. The ratio is highest when the step is single-threaded and
the state stays in cache (2.22x at 1 thread, 2.03x at 6 threads) and lowest when the state streams
from DRAM across 24 layers, as in real decoding (1.39x at 1 thread, 1.33x at 6 threads). So the same
mechanism that makes MIMO nearly free on a GPU -- decode being limited by memory traffic rather than
arithmetic -- is partly at work on this CPU too: in the memory-bound regime, 3.58x the FLOPs costs
only about 1.3x the time. But a CPU core has far less compute per byte of memory bandwidth than a
GPU, so the extra math is never fully hidden; budget roughly 1.3x-2.2x SISO's decode time for MIMO,
depending on thread count and whether the state fits in cache. See `results/mamba3_bench.png`
(right panel: MIMO/SISO time ratio vs the FLOP ratio and the free line) and
`results/verify_output.txt` and `results/verify_output_h64.txt` for the correctness runs this answer
relies on.

**Hardware and roofline.** Intel Core Ultra 7 155H: 6 P-cores with AVX2 + FMA (two 256-bit FMA units,
32 fp32 FLOP per cycle per core, up to 4.8 GHz), 24 MB L3, measured DRAM bandwidth 52.1 GB/s (STREAM
Triad, 6 threads on the 6 P-cores). Its fp32 peak is at most 6 x 32 x 4.8 GHz = 922 GFLOP/s, so the
textbook ridge point is about 922 / 52.1 = 18 FLOP/byte. SISO (0.85 FLOP/byte) and MIMO (2.70) are both
far below it, so by the peak roofline MIMO would be purely memory-bound and free. It is not, because
these kernels run far below peak, for two roughly equal reasons. Per core, one thread in cache
reaches 50 GFLOP/s for MIMO, about a third of one P-core's 154 GFLOP/s peak. Across cores, six threads
reach only 108 GFLOP/s, 2.15x one thread instead of 6x (SISO 1.97x, Mamba-2 1.24x). For the scaling,
the data cannot say why; lower clocks with six cores busy, per-step OpenMP start-up and shared-cache contention are candidates,
and a 1-6 thread sweep with frequency logging would separate them. Taking MIMO's own in-cache 108
GFLOP/s as an empirical ceiling for this code puts its ridge at about 108 / 52.1 = 2.1 FLOP/byte,
between SISO's 0.85 and MIMO's 2.70. SISO stays memory-bound (it streams 36 GB/s from DRAM at 6
threads, 69% of STREAM); MIMO streams only 31 GB/s (59%) and reaches 83 GFLOP/s, below both this
ceiling and the DRAM roof at its intensity, which is consistent with only part of its extra arithmetic
being hidden behind the memory traffic. The 1.33x-2.22x ratios above come straight from measured times
and do not depend on this analysis.

**Caveat: the in-cache ratio includes a shared fixed cost.** For the RoPE angles, every head computes
32 `tanh`, `sin` and `cos` values per step with scalar library calls. This costs SISO and MIMO the same
and is not in the FLOP count, so it dilutes the MIMO/SISO ratio, most of all in cache, where the rest
of the step is fast. The 2.0x-2.2x in-cache ratios therefore likely reflect this shared cost rather than
MIMO's extra arithmetic being hidden. An exploratory run with the RoPE work switched off pointed the same
way, but its timings drifted too much between sessions to report. The DRAM regime (1.33x-1.39x) is less
affected and is the better evidence for the answer. Computing `tanh` once per step (it does not depend
on the head) and a vectorised `sincos` would cut this cost; both would need a new benchmark run.

## Stretch goal: the kernels as a ggml operator

`src/ggml_m3op.cpp` adds `ggml_m3_step()`, a ggml operator built with `ggml_custom_4d`. The graph is
small: eight input tensors (`x`, `z` `[P, H]`; `B`, `C` `[N, R]`; `dt`, `A`, `trap` `[H]`; the angle
projection `[S]`) feed one `GGML_OP_CUSTOM` node whose output is `y` `[P, H]`. The node calls the
unmodified kernels from `src/m3.h`. The layer's fixed parameters and its decode cache (SSM state,
RoPE angles, previous B and x) are kept outside the graph and passed to the op as `userdata`, the way
llama.cpp keeps a recurrent model's state in a cache next to the compute graph. The graph is built
once, and each decode step copies the token's inputs into the input tensors and recomputes it.

The test runs all 6 kernels through the graph for the 1,000 steps of `data/case.bin`. A kernel passes
if (1) every step's output is within the same tolerance of the float64 reference as `verify.cpp`
uses, (2) the final state is within that tolerance, and (3) the graph's output and state are
bit-identical to a direct call of the same kernel (`graph_vs_direct = 0`). The table also gives the
time per step through the graph and by direct call. The run is pinned like the 1-thread benchmark:
one P-core (logical processor 10, mask `0x400`), `OMP_NUM_THREADS=1` and one ggml thread, so both
times are single-thread P-core times. They are for the 16-head case, so they are not directly
comparable with the 64-head benchmark (a 64-head step does 4x the work; the direct-call times are
0.83x-1.19x a quarter of the 1-thread in-cache benchmark). The result here is the difference between
them, the graph overhead: at most about 1.5 us per step for the AVX2 kernels, and within run-to-run
noise for the plain kernels (between -6.9 and +5.3 us; a negative value can only be noise).
ggml comes from the same llama.cpp checkout as the main project (commit `a946eb541`, build b12431),
compiled a second time with MinGW gcc because the MSVC build cannot be linked with g++ objects. Output: `results/ggml_m3op_output.txt`.

## Running on Windows (PowerShell)

`.\run_bonus.ps1` runs steps 1-6 in order (`-SkipBench` and `-SkipGgml` leave out the slow steps).
The same commands, one by one:

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

# 3. verify all 6 kernels over 1,000 steps, at 16 heads and at the benchmark shape (64 heads)
.\build\verify.exe data\case.bin 1e-4 | Out-File -Encoding ascii results\verify_output.txt; Get-Content results\verify_output.txt
python ref\mamba3_reference.py --repo $HOME\mamba-official --heads 64 --out data\case_h64.bin
.\build\verify.exe data\case_h64.bin 1e-4 | Out-File -Encoding ascii results\verify_output_h64.txt; Get-Content results\verify_output_h64.txt

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

# 6. stretch goal: ggml operator (a separate MinGW build of the ggml in $HOME\llama.cpp at a946eb541)
cmake -S $HOME\llama.cpp -B $HOME\llama.cpp\build-mingw -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build $HOME\llama.cpp\build-mingw --target ggml -j 8
$L = "$HOME\llama.cpp\build-mingw\ggml\src"
g++ -O2 -std=c++17 -fopenmp -static "-I$HOME\llama.cpp\ggml\include" src/ggml_m3op.cpp build/plain.o build/avx2.o "$L\ggml.a" "$L\ggml-cpu.a" "$L\ggml-base.a" -o build/ggml_m3op.exe
$env:OMP_NUM_THREADS = 1    # pinned like the 1-thread benchmark: one P-core, one thread
$p = Start-Process .\build\ggml_m3op.exe -ArgumentList "data\case.bin","1e-4","1" -NoNewWindow -PassThru -RedirectStandardOutput results\ggml_m3op_output.txt
$null = $p.Handle; $p.ProcessorAffinity = [IntPtr]0x400; $p.WaitForExit(); Get-Content results\ggml_m3op_output.txt
```

Benchmark settings: H = 64 heads, P = 64, N = 128, the layer shape of `state-spaces/mamba3-siso-1.5b`
and `mamba3-mimo-1.5b` (2 MiB of state per layer). Two regimes:

* `layers = 1`: the same layer's state every step, so it stays in cache.
* `layers = 24`: the steps cycle through 24 layers (48 MiB, twice the 24 MB L3), so each step streams its
  state from DRAM, as in real decoding of a 24-layer model.

Each configuration is timed 3 times for at least 1 s each; the CSV has the mean, min and max.
The FLOPs and bytes per step in the CSVs are modelled (`step_flops` and `step_bytes` in `src/m3.h`,
counting a multiply and an add separately), not read from hardware counters; arithmetic intensity is
their ratio.
