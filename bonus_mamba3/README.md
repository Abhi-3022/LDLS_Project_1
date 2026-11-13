# Bonus: one Mamba-3 decode step on CPU (SISO and MIMO)

**Question:** MIMO adds decode compute without growing the state. On GPUs that compute is
almost free because decode is memory-bound. Is it also almost free on a CPU?

`bash run_bonus.sh` runs everything below. The numbers and the answer paragraph are written
to `results/summary.md`; the plots go to `../plots/m3_*.png`.

## 1. Reference: the official code, on CPU

`state-spaces/mamba` (pinned at `e9594ce`) implements the Mamba-3 decode step with GPU-only
kernels: Triton rotary + CuTe-DSL `mamba3_step_fn`. The package cannot be imported on a CPU-only
machine. The repository also contains plain-PyTorch reference implementations of the
recurrence, and `ref/official_ref.py` extracts these **verbatim** with `ast` and runs them on CPU:

| Function | Source file | Use |
|---|---|---|
| `mamba3_siso_step_ref` | `tests/ops/triton/test_mamba3_siso.py` | SISO ground truth |
| `mamba3_MIMO_step_ref` | `tests/ops/tilelang/test_mamba3_mimo.py` | MIMO ground truth |
| `mamba3_siso_fwd_ref` | same as SISO | cross-check: step recurrence = chunked training forward |
| `selective_state_update_ref` | `mamba_ssm/ops/triton/selective_state_update.py` | Mamba-2 ground truth |
| `heavy_tail_activation` | `mamba_ssm/modules/mamba3.py` | data-dependent A |

`ref/make_reference.py` builds the per-token kernel inputs exactly as `Mamba3.step()` does (in_proj
split, `softplus(dd_dt + dt_bias)`, heavy-tail A, RMS-normed B/C) and runs 1,000 decode steps
through the references in **float64** (ground truth) and float32 (to show PyTorch's own fp32 error).
Sanity checks it prints (stored in `data/*/meta.json`):

* SISO step ref vs chunked forward ref: 6e-8 (the forward ref computes angles in fp32);
* SISO ref vs MIMO ref at rank 1 with unit projections: 3e-16, so the two references agree.

Two input sets are used:

* **Random weights**, with the parameterisation and initialisation of `Mamba3.__init__`, at
  ranks 1/2/4/8, plus Mamba-2 at the same state size. Shape: d_model 2048, 64 heads x head dim 64,
  d_state 128, one B/C group, RoPE on half of the state.
* **Released checkpoints.** `state-spaces/mamba3-siso-1.5b` and `mamba3-mimo-1.5b` are public and
  have exactly this layer shape. Layer 0 is run on the real Llama-3 token embeddings of the
  WikiText-2 sample (`--real`), with the checkpoint's own in_proj, dt_bias, B/C norms and biases,
  D and MIMO projections. A full-model port and perplexity are out of scope, as the brief allows.

## 2. Recurrence implemented (per head, per token)

```
angle  += tanh(angle_proj) * pi * dt            (fp64, kept mod 2*pi)
q_r, k_r = RoPE(C_r + C_bias, angle), RoPE(B_r + B_bias, angle)        r = 0..R-1
alpha = exp(A dt),  beta = (1 - sigmoid(trap)) dt alpha,  gamma = sigmoid(trap) dt
v_r = x * mimo_x_r                               (SISO: v = x)
S  <- alpha S + sum_r beta v_r(t-1) k_r(t-1)^T + gamma v_r k_r^T       (exponential-trapezoidal)
y_r = S q_r + D v_r
out = sum_r y_r * silu(z * mimo_z_r) * mimo_o_r  (SISO: y * silu(z))
```

The state is `[H][P][N]` fp32 (2 MiB per layer), the same layout as the reference and as ggml's
`ssm_scan`. Besides S, Mamba-3 keeps the previous token's rotated B (`[H][R][N]`), x (`[H][P]`)
and the angles (`[H][S]`, fp64). The kernel's scope equals that of the GPU step kernels: in_proj and
out_proj are ordinary GEMVs left to the engine.

## 3. Kernels (`src/`)

`m3_kernel.inc` holds one implementation over a small vector abstraction. It is compiled four
times, and `m3_dispatch.cpp` picks the variant at run time:

| Build | Flags | Notes |
|---|---|---|
| `scalar` | baseline x86-64, `-fno-tree-vectorize` | plain C++ baseline; libm `expf/sinf/cosf` |
| `avx2` | `-mavx2 -mfma` | the laptop path (16 x 256-bit registers) |
| `avx512` | `-mavx512f -mavx512dq` | 32 x 512-bit registers |
| `neon` | aarch64 | **compiles only on ARM; not tested here (no ARM hardware)** |

Optimisations, each measured with the verifier rerun after it:

1. **Register blocking over state rows.** The B, previous-B and C vectors loaded for one chunk of n
   are reused across PB rows (PB = 4/2/1 for R <= 2 / 4 / 8 on AVX-512; less on AVX2).
2. **Vectorised transcendental functions.** The SiLU gates (R x 64 x 64 `exp` per step) and the
   RoPE `sin/cos` were scalar libm calls and dominated the step: MIMO-4 went from 250 to 155 µs on
   one cache-hot core. They now use Cephes-style polynomial `exp` and `sincos` (~1 ulp) in a
   vectorised epilogue.
3. **Software prefetch** of the state rows two blocks ahead: -10 to -20 % with 8 threads when the
   state comes from DRAM.

Mamba-2 (`mamba2`) gets the same blocking, prefetch and vectorised gate, so the comparison is fair.

## 4. Correctness (`build/verify`)

Over 1,000 decode steps, each kernel is compared with the fp64 official reference. The stated
tolerance is `max |y - y_ref| <= 1e-5 * max |y_ref|`, and the same for the final state.

* All 48 runs pass: scalar/AVX2/AVX-512 x 1 and 8 threads x {SISO, MIMO R=1/2/4/8, Mamba-2,
  real SISO layer, real MIMO-4 layer}.
* Output error is 1e-7 to 3.5e-7, at or below PyTorch's own fp32 run of the reference (1.4e-7 to
  7.9e-6).
* Final-state error is 1e-7 to 7e-7, against PyTorch fp32's 6e-6 to 7e-4.
* Why the RoPE angle state is fp64: the angle is a running sum over the whole sequence. Kept in fp32
  (as the GPU kernel does), it drifted 4.4e-5 rad over 1,000 steps on the real SISO layer. That put
  the final state at 2.5e-5 and failed the tolerance. In fp64 it costs 16 KiB per layer (<1 % of the
  2 MiB state) and S scalar ops per head; `sincos` stays vectorised in fp32.
* Full table: `results/verify.csv`.

## 5. Timing (`build/bench`)

* **Measured per step, one layer:** scalar / AVX2 / AVX-512, 1-32 threads (heads split across
  OpenMP threads), mean of 3 repetitions.
* **Two memory regimes:**
  * *cold*: steps cycle through 256 layer states (512 MiB, > 4x the 105 MiB L3), so the state
    streams from DRAM. This is what a 24-48-layer model sees on a laptop.
  * *hot*: one layer reused, so the state stays in L2/L3.
* **FLOPs per state element:** Mamba-2: 5; Mamba-3 rank R: 1 + 6R (SISO 7, MIMO-4 25).
* **Bytes per step:** the 2 MiB state is read and written (8 B/element), plus the small B/x/angle
  states.
* **Arithmetic intensity:** SISO ~0.85, MIMO-4 ~3 FLOP/B.

## 6. ggml integration (`build/ggml_m3`)

* **Mamba-2 baseline:** ggml's own `GGML_OP_SSM_SCAN`, the operator llama.cpp uses for Mamba-2,
  timed for one decode token at the same shape. It is checked against the reference and against
  our Mamba-2 kernel.
* **Stretch goal:** our Mamba-3 SISO/MIMO step wrapped as a ggml custom op (`ggml_custom_4d`,
  heads split by ggml's `ith/nth`) in a small ggml graph. Over 1,000 steps the graph output is
  **bit-identical** to the direct kernel call. The table also reports the per-step graph-dispatch
  overhead.

## Files

```
ref/official_ref.py     extract + run the upstream reference functions on CPU
ref/make_reference.py   inputs and fp64/fp32 reference outputs  -> data/<case>/
src/m3_kernel.inc       the kernels (one source, per-ISA builds m3_{scalar,avx2,avx512,neon}.cpp)
src/verify.cpp          1000-step correctness vs reference      -> results/verify.csv
src/bench.cpp           timing sweep                            -> results/bench.csv
src/ggml_m3.cpp         ggml SSM_SCAN baseline + custom op      -> results/ggml_m3.csv
analysis.py             tables, plots, answer                   -> results/summary.md, ../plots/m3_*.png
```
