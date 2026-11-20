// Mamba-3 (SISO, MIMO) and Mamba-2 single-token decode steps for CPU.
//
// One decode step for one layer, per head h (P = headdim, N = d_state, R = MIMO rank):
//   angle_h  += tanh(theta) * dt_h * pi                     (data-dependent RoPE, num_rope_angles = N/4)
//   B_r, C_r  = rotate(B_r + Bbias_hr, angle_h), rotate(C_r + Cbias_hr, angle_h)
//   alpha = exp(A_h dt_h),  beta = (1 - trap_h) dt_h alpha,  gamma = trap_h dt_h
//   S_h[p,n] = alpha S_h[p,n] + sum_r gamma xp_r[p] x[p] B_r[n] + sum_r beta xp_r[p] x_prev[p] B_prev_r[n]
//   y[p]     = sum_r op_r[p] * silu(zp_r[p] z[p]) * (sum_n S_h[p,n] C_r[n] + D_h xp_r[p] x[p])
//   B_prev, x_prev <- B_r (rotated), x
// SISO is R = 1 with xp = zp = op = 1 and RoPE on interleaved pairs; MIMO uses RoPE on pairs (i, i + N/2).
// This matches selective_state_update_fused_ref_v2 + apply_rotary_qk_inference_reference in
// state-spaces/mamba. in_proj / out_proj / B,C RMSNorm are outside the kernel.
//
// Mamba-2 (selective_state_update_ref):  S = exp(A dt) S + dt x B^T,  y = (S C + D x) * silu(z).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

constexpr int kMaxN = 256;      // largest d_state supported by the stack scratch buffers
constexpr int kMaxRank = 4;
constexpr float kPi = 3.14159265358979323846f;

struct Params {                 // one layer
    int H = 0, P = 0, N = 0, R = 0, S = 0;   // heads, headdim, d_state, stored MIMO rank, rope angles
    std::vector<float> D;       // [H]
    std::vector<float> Bb, Cb;  // [H][R][N]  B/C biases
    std::vector<float> xp, zp, op;  // [R][H][P] MIMO x / z / output projections
};

struct State {                  // per-layer decode cache
    std::vector<float> s;       // [H][P][N] SSM state
    std::vector<float> ang;     // [H][S]    accumulated RoPE angles
    std::vector<float> k;       // [H][R][N] previous rotated B
    std::vector<float> v;       // [H][P]    previous x
    void init(const Params& p, int R) {
        s.assign((size_t)p.H * p.P * p.N, 0.f);
        ang.assign((size_t)p.H * p.S, 0.f);
        k.assign((size_t)p.H * R * p.N, 0.f);
        v.assign((size_t)p.H * p.P, 0.f);
    }
};

struct Input {                  // one token, outputs of in_proj after activations
    const float *x, *z;         // [H][P]
    const float *B, *C;         // [R][N] (ngroups = 1, shared by all heads)
    const float *dt, *A, *trap; // [H]
    const float *ang;           // [S]
};

enum class Rope { Pairwise, Halved };

// RoPE on B and C for one head (O(R*N) work, shared by the plain and SIMD kernels).
template <int R>
inline void rope_head(const Params& p, int h, Rope mode, const Input& in, float* angle,
                      float* Brot, float* Crot) {
    const int N = p.N, S = p.S;
    float cs[kMaxN / 2], sn[kMaxN / 2];
    const float dt = in.dt[h];
    for (int i = 0; i < S; ++i) {
        angle[i] += std::tanh(in.ang[i]) * dt * kPi;
        cs[i] = std::cos(angle[i]);
        sn[i] = std::sin(angle[i]);
    }
    for (int r = 0; r < R; ++r) {
        const float* bb = &p.Bb[((size_t)h * p.R + r) * N];
        const float* cb = &p.Cb[((size_t)h * p.R + r) * N];
        float* bo = Brot + r * N;
        float* co = Crot + r * N;
        for (int n = 0; n < N; ++n) {
            bo[n] = in.B[r * N + n] + bb[n];
            co[n] = in.C[r * N + n] + cb[n];
        }
        for (int i = 0; i < S; ++i) {
            const int a = mode == Rope::Pairwise ? 2 * i : i;
            const int b = mode == Rope::Pairwise ? 2 * i + 1 : i + N / 2;
            const float b0 = bo[a], b1 = bo[b], c0 = co[a], c1 = co[b];
            bo[a] = b0 * cs[i] - b1 * sn[i];
            bo[b] = b0 * sn[i] + b1 * cs[i];
            co[a] = c0 * cs[i] - c1 * sn[i];
            co[b] = c0 * sn[i] + c1 * cs[i];
        }
    }
}

inline float silu(float v) { return v / (1.f + std::exp(-v)); }

// ---- kernels (plain.cpp: portable C++, avx2.cpp: AVX2 + FMA intrinsics) ----
void siso_plain(const Params&, State&, const Input&, float* y);
void mimo_plain(const Params&, State&, const Input&, float* y);
void mamba2_plain(const Params&, State&, const Input&, float* y);
void siso_avx2(const Params&, State&, const Input&, float* y);
void mimo_avx2(const Params&, State&, const Input&, float* y);
void mamba2_avx2(const Params&, State&, const Input&, float* y);

struct Kernel {
    const char* name;
    const char* impl;
    int rank;                   // 1 = SISO, 4 = MIMO, 0 = Mamba-2
    void (*fn)(const Params&, State&, const Input&, float*);
};

inline const Kernel kKernels[] = {
    {"siso", "plain", 1, siso_plain},     {"siso", "avx2", 1, siso_avx2},
    {"mimo", "plain", 4, mimo_plain},     {"mimo", "avx2", 4, mimo_avx2},
    {"mamba2", "plain", 0, mamba2_plain}, {"mamba2", "avx2", 0, mamba2_avx2},
};

// Work model per layer-step (state traffic dominates; O(P*R) and O(R*N) per-head terms included).
// FLOPs count a multiply and an add separately (an FMA = 2).
inline double step_flops(const Params& p, int rank) {
    const double PN = (double)p.P * p.N;
    if (rank == 0) return p.H * PN * 5.0;                      // decay 1, update 2, output 2
    return p.H * (PN * (1.0 + 6.0 * rank)                     // decay 1, 2R update terms x2, R outputs x2
                  + (double)rank * p.N * 4 * 2                 // RoPE + bias on B and C
                  + (double)rank * p.P * 10);                  // projections, D skip, gate
}
inline double step_bytes(const Params& p, int rank) {
    const double f = 4.0, PN = (double)p.P * p.N;
    if (rank == 0) return p.H * (2 * PN + 3.0 * p.P) * f + 2.0 * p.N * f;
    const int proj = rank > 1 ? 3 : 0;
    return p.H * (2 * PN                                       // state read + write
                  + 5.0 * p.P                                  // x, z, y, x_prev read + write
                  + 2.0 * rank * p.N                           // B_prev read + write
                  + 2.0 * rank * p.N                           // B/C biases
                  + (double)proj * rank * p.P                  // MIMO projections
                  + 2.0 * p.S) * f                             // angle state read + write
           + 2.0 * rank * p.N * f;                             // B, C (shared by heads)
}
