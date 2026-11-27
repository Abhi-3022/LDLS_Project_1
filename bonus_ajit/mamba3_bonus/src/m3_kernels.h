// Mamba-3 (SISO / MIMO) and Mamba-2 single-token decode steps for one layer, CPU.
//
// Scope = the part of the layer that the GPU step kernels implement
// (rotary_qk_inference + mamba3_step_fn in state-spaces/mamba): angle update, data-
// dependent RoPE on B/C, exponential-trapezoidal state update, readout, D skip,
// SiLU gate and the MIMO x/z up- and out down-projections. in_proj/out_proj are
// plain GEMVs and are left to the GEMV kernels of the inference engine.
//
// Layouts (row-major, last index fastest):
//   ssm state [H][P][N] fp32, the same layout as ggml's ssm_scan and the reference.
#pragma once

struct M3Dims {
    int H;  // heads
    int P;  // head dim (x/v)
    int N;  // d_state
    int R;  // MIMO rank (1 = SISO)
    int S;  // number of RoPE angles (rotates the first 2*S state dims pairwise)
};

struct M3Params {            // fixed per layer
    const float * D;         // [H]
    const float * qb;        // [H][R][N]  C bias
    const float * kb;        // [H][R][N]  B bias
    const float * mx;        // [H][R][P]  MIMO x up-projection   (unused for SISO)
    const float * mz;        // [H][R][P]  MIMO z up-projection   (unused for SISO)
    const float * mo;        // [H][R][P]  MIMO out down-projection (unused for SISO)
};

struct M3State {             // per layer, updated in place
    double * ang;            // [H][S]     cumulative RoPE angle (fp64: a 1000-step fp32 sum drifts ~4e-5 rad)
    float * ssm;             // [H][P][N]
    float * kst;             // [H][R][N]  rotated B of the previous token
    float * xst;             // [H][P]     x of the previous token
};

struct M3Step {              // per token (after in_proj and preprocessing)
    const float * q;         // [R][N]  RMS-normed C (one B/C group)
    const float * k;         // [R][N]  RMS-normed B
    const float * x;         // [H][P]
    const float * z;         // [H][P]
    const float * adt;       // [H]  A*dt
    const float * dt;        // [H]  softplus(dd_dt + dt_bias)
    const float * trap;      // [H]  trapezoid logit (sigmoid applied in the kernel)
    const float * ang;       // [S]  raw angle projection (tanh * pi applied in the kernel)
};

struct M2Params { const float * A; const float * D; const float * dt_bias; };  // [H] each
struct M2Step   { const float * x; const float * z; const float * B; const float * C; const float * dt; };
                  // x,z [H][P]; B,C [N] (one group); dt [H] raw (softplus(dt + dt_bias) in kernel)

// One decode step for heads [h0, h1). y is [H][P].
typedef void (*m3_step_fn)(const M3Dims &, const M3Params &, M3State &, const M3Step &, float * y, int h0, int h1);
// dz = false gives the bare ssm_scan semantics of ggml (no D skip, no gate).
typedef void (*m2_step_fn)(const M3Dims &, const M2Params &, float * ssm, const M2Step &, float * y, int h0, int h1, bool dz);

#define M3_DECLARE(ns)                                                                                   \
    namespace ns {                                                                                       \
    void mamba3_siso(const M3Dims &, const M3Params &, M3State &, const M3Step &, float *, int, int);    \
    void mamba3_mimo(const M3Dims &, const M3Params &, M3State &, const M3Step &, float *, int, int);    \
    void mamba2(const M3Dims &, const M2Params &, float *, const M2Step &, float *, int, int, bool);     \
    }

M3_DECLARE(m3_scalar)
M3_DECLARE(m3_avx2)
M3_DECLARE(m3_avx512)
M3_DECLARE(m3_neon)

struct M3Impl {
    const char * name;
    m3_step_fn   siso;
    m3_step_fn   mimo;
    m2_step_fn   m2;
    bool         supported;   // by the running CPU
};

// all implementations compiled into this binary, filtered by CPU support
int m3_impls(M3Impl * out, int max);
