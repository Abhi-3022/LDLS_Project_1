// Kernel body, compiled once per ISA. The including file defines one of
// M3_ISA_AVX512 / M3_ISA_AVX2 / M3_ISA_NEON / M3_ISA_SCALAR and M3_NS.
//
// Per head h and per token:
//   angle  += tanh(ang_raw) * pi * dt                         (mod 2*pi)
//   q_r, k_r = RoPE(C_r + qb, angle), RoPE(B_r + kb, angle)   r = 0..R-1
//   alpha = exp(A*dt), beta = (1 - sigmoid(trap)) * dt * alpha, gamma = sigmoid(trap) * dt
//   v_r = x * mx_r          (SISO: v = x)
//   S[p][n] = alpha*S[p][n] + sum_r beta*vprev_r[p]*kprev_r[n] + gamma*v_r[p]*k_r[n]
//   y_r[p]  = sum_n S[p][n] q_r[n] + D v_r[p]
//   out[p]  = sum_r y_r[p] * silu(z[p] * mz_r[p]) * mo_r[p]   (SISO: y * silu(z))
// matching mamba3_siso_step_ref / mamba3_MIMO_step_ref of state-spaces/mamba.
#include "m3_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(M3_ISA_AVX512)
#include <immintrin.h>
typedef __m512 vf;
enum { W = 16 };
static inline vf    vload(const float * p)        { return _mm512_loadu_ps(p); }
static inline void  vstore(float * p, vf v)        { _mm512_storeu_ps(p, v); }
static inline vf    vset1(float x)                 { return _mm512_set1_ps(x); }
static inline vf    vzero()                        { return _mm512_setzero_ps(); }
static inline vf    vmul(vf a, vf b)               { return _mm512_mul_ps(a, b); }
static inline vf    vadd(vf a, vf b)               { return _mm512_add_ps(a, b); }
static inline vf    vfma(vf a, vf b, vf c)         { return _mm512_fmadd_ps(a, b, c); }  // a*b + c
static inline float vsum(vf v)                     { return _mm512_reduce_add_ps(v); }
static inline vf    vsub(vf a, vf b)               { return _mm512_sub_ps(a, b); }
static inline vf    vdiv(vf a, vf b)               { return _mm512_div_ps(a, b); }
static inline vf    vmin(vf a, vf b)               { return _mm512_min_ps(a, b); }
static inline vf    vmax(vf a, vf b)               { return _mm512_max_ps(a, b); }
static inline vf    vabs(vf a)                     { return _mm512_abs_ps(a); }
static inline vf    vfloor(vf a)                   { return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
static inline vf    vpow2n(vf y, vf n)             { return _mm512_scalef_ps(y, n); }  // y * 2^n
#elif defined(M3_ISA_AVX2)
#include <immintrin.h>
typedef __m256 vf;
enum { W = 8 };
static inline vf    vload(const float * p)        { return _mm256_loadu_ps(p); }
static inline void  vstore(float * p, vf v)        { _mm256_storeu_ps(p, v); }
static inline vf    vset1(float x)                 { return _mm256_set1_ps(x); }
static inline vf    vzero()                        { return _mm256_setzero_ps(); }
static inline vf    vmul(vf a, vf b)               { return _mm256_mul_ps(a, b); }
static inline vf    vadd(vf a, vf b)               { return _mm256_add_ps(a, b); }
static inline vf    vfma(vf a, vf b, vf c)         { return _mm256_fmadd_ps(a, b, c); }
static inline float vsum(vf v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
static inline vf    vsub(vf a, vf b)               { return _mm256_sub_ps(a, b); }
static inline vf    vdiv(vf a, vf b)               { return _mm256_div_ps(a, b); }
static inline vf    vmin(vf a, vf b)               { return _mm256_min_ps(a, b); }
static inline vf    vmax(vf a, vf b)               { return _mm256_max_ps(a, b); }
static inline vf    vabs(vf a)                     { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
static inline vf    vfloor(vf a)                   { return _mm256_floor_ps(a); }
static inline vf    vpow2n(vf y, vf n) {
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
#elif defined(M3_ISA_NEON)
#include <arm_neon.h>
typedef float32x4_t vf;
enum { W = 4 };
static inline vf    vload(const float * p)        { return vld1q_f32(p); }
static inline void  vstore(float * p, vf v)        { vst1q_f32(p, v); }
static inline vf    vset1(float x)                 { return vdupq_n_f32(x); }
static inline vf    vzero()                        { return vdupq_n_f32(0.0f); }
static inline vf    vmul(vf a, vf b)               { return vmulq_f32(a, b); }
static inline vf    vadd(vf a, vf b)               { return vaddq_f32(a, b); }
static inline vf    vfma(vf a, vf b, vf c)         { return vfmaq_f32(c, a, b); }
static inline float vsum(vf v)                     { return vaddvq_f32(v); }
static inline vf    vsub(vf a, vf b)               { return vsubq_f32(a, b); }
static inline vf    vdiv(vf a, vf b)               { return vdivq_f32(a, b); }
static inline vf    vmin(vf a, vf b)               { return vminq_f32(a, b); }
static inline vf    vmax(vf a, vf b)               { return vmaxq_f32(a, b); }
static inline vf    vabs(vf a)                     { return vabsq_f32(a); }
static inline vf    vfloor(vf a)                   { return vrndmq_f32(a); }
static inline vf    vpow2n(vf y, vf n) {
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}
#else  // M3_ISA_SCALAR: plain C++, built with auto-vectorisation disabled
typedef float vf;
enum { W = 1 };
static inline vf    vload(const float * p)        { return *p; }
static inline void  vstore(float * p, vf v)        { *p = v; }
static inline vf    vset1(float x)                 { return x; }
static inline vf    vzero()                        { return 0.0f; }
static inline vf    vmul(vf a, vf b)               { return a * b; }
static inline vf    vadd(vf a, vf b)               { return a + b; }
static inline vf    vfma(vf a, vf b, vf c)         { return a * b + c; }
static inline float vsum(vf v)                     { return v; }
static inline vf    vsub(vf a, vf b)               { return a - b; }
static inline vf    vdiv(vf a, vf b)               { return a / b; }
static inline vf    vmin(vf a, vf b)               { return a < b ? a : b; }
static inline vf    vmax(vf a, vf b)               { return a > b ? a : b; }
static inline vf    vabs(vf a)                     { return fabsf(a); }
static inline vf    vfloor(vf a)                   { return floorf(a); }
#endif

#if !defined(M3_ISA_SCALAR)
// exp(x): Cody-Waite reduction x = n*ln2 + r, Cephes expf polynomial on |r| <= ln2/2 (~1 ulp)
static inline vf vexp(vf x) {
    x = vmin(vmax(x, vset1(-87.3f)), vset1(88.7f));
    const vf n = vfloor(vfma(x, vset1(1.44269504088896341f), vset1(0.5f)));
    vf r = vfma(n, vset1(-0.693359375f), x);
    r = vfma(n, vset1(2.12194440e-4f), r);
    vf p = vset1(1.9875691500e-4f);
    p = vfma(p, r, vset1(1.3981999507e-3f));
    p = vfma(p, r, vset1(8.3334519073e-3f));
    p = vfma(p, r, vset1(4.1665795894e-2f));
    p = vfma(p, r, vset1(1.6666665459e-1f));
    p = vfma(p, r, vset1(5.0000001201e-1f));
    const vf y = vadd(vfma(vmul(p, r), r, r), vset1(1.0f));
    return vpow2n(y, n);
}

// sin and cos: quadrant q = round(2x/pi), 3-part Cody-Waite reduction to |r| <= pi/4,
// Cephes sinf/cosf polynomials, then rotate by q*pi/2 (no branches, no integer lanes)
static inline void vsincos(vf x, vf * sn, vf * cs) {
    const vf q = vfloor(vfma(x, vset1(0.636619772367581343f), vset1(0.5f)));
    vf r = vfma(q, vset1(-1.5703125f), x);
    r = vfma(q, vset1(-4.837512969970703125e-4f), r);
    r = vfma(q, vset1(-7.54978995489188216e-8f), r);
    const vf z = vmul(r, r);
    vf ps = vfma(z, vset1(-1.9515295891e-4f), vset1(8.3321608736e-3f));
    ps = vfma(ps, z, vset1(-1.6666654611e-1f));
    const vf s = vfma(vmul(ps, z), r, r);
    vf pc = vfma(z, vset1(2.443315711809948e-5f), vset1(-1.388731625493765e-3f));
    pc = vfma(pc, z, vset1(4.166664568298827e-2f));
    const vf c = vadd(vfma(vmul(pc, z), z, vmul(z, vset1(-0.5f))), vset1(1.0f));
    const vf q4 = vsub(q, vmul(vset1(4.0f), vfloor(vmul(q, vset1(0.25f)))));     // q mod 4 in {0,1,2,3}
    const vf cq = vsub(vset1(1.0f), vmin(q4, vsub(vset1(4.0f), q4)));            // cos(q*pi/2)
    const vf sq = vsub(vset1(1.0f), vabs(vsub(q4, vset1(1.0f))));                // sin(q*pi/2)
    *sn = vfma(s, cq, vmul(c, sq));
    *cs = vsub(vmul(c, cq), vmul(s, sq));
}
#else
// the scalar baseline uses libm
static inline vf vexp(vf x) { return expf(x); }
static inline void vsincos(vf x, vf * sn, vf * cs) { *sn = sinf(x); *cs = cosf(x); }
#endif

// silu(x) = x * sigmoid(x)
static inline vf vsilu(vf x) { return vdiv(x, vadd(vset1(1.0f), vexp(vsub(vzero(), x)))); }

namespace M3_NS {

static const int RMAX = 16;
static const int NMAX = 256;
static const int SMAX = 128;
static const int PMAX = 256;

#ifndef M3_PREFETCH
#define M3_PREFETCH 2   // state-row prefetch distance in row blocks (0 = off); 2 was best on SPR
#endif

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x)    { return x * sigmoidf(x); }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// RoPE on (v + bias) for the first 2*S dims (interleaved pairs), bias-add on the rest
static inline void rope_bias(const float * v, const float * b, const float * cs, const float * sn,
                             float * out, int N, int S) {
    for (int i = 0; i < S; i++) {
        const float a0 = v[2*i] + b[2*i];
        const float a1 = v[2*i + 1] + b[2*i + 1];
        out[2*i]     = a0 * cs[i] - a1 * sn[i];
        out[2*i + 1] = a0 * sn[i] + a1 * cs[i];
    }
    for (int n = 2*S; n < N; n++) {
        out[n] = v[n] + b[n];
    }
}

// Rows per register block. The B/C/previous-B vectors loaded for one chunk of n are
// reused across PB state rows, so each load feeds PB FMAs (GEMM-style register
// blocking). PB is sized to the vector register file: 32 regs (AVX-512, NEON) or 16
// (AVX2). The scalar build stays unblocked as the plain C++ baseline.
#if defined(M3_ISA_AVX512) || defined(M3_ISA_NEON)
template <int R> struct RowBlock { enum { v = R <= 2 ? 4 : R <= 4 ? 2 : 1 }; };
enum { PB_M2 = 4 };
#elif defined(M3_ISA_AVX2)
template <int R> struct RowBlock { enum { v = R <= 2 ? 2 : 1 }; };
enum { PB_M2 = 4 };
#else
template <int R> struct RowBlock { enum { v = 1 }; };
enum { PB_M2 = 1 };
#endif

struct HeadCtx {             // per-head values shared by all row blocks
    float alpha, beta, gamma, Dh;
    const float * kp;        // [R][N] previous rotated B
    const float * kr;        // [R][N] current rotated B
    const float * qr;        // [R][N] current rotated C
    const float * xp;        // [P] previous x
    const float * xc;        // [P] current x
    const float * zc;        // [P]
    const float * mx, * mz, * mo;  // [R][P] (MIMO only)
    float * S0;              // [P][N]
    float * yb;              // [R][P] raw readouts sum_n S q_r
};

// update + read out PB consecutive state rows starting at p0; R = RT (RT > 0) or rt
template <int RT, bool MIMO, int PB>
static inline void m3_rows(const HeadCtx & c, int p0, int rt, int P, int N) {
    const int R = RT > 0 ? RT : rt;
    const int RA = RT > 0 ? RT : RMAX;
    vf ub[PB][RA], ug[PB][RA], acc[PB][RA];
    for (int b = 0; b < PB; b++) {
        const int p = p0 + b;
        for (int r = 0; r < R; r++) {
            const float m = MIMO ? c.mx[r * P + p] : 1.0f;
            ub[b][r]  = vset1(c.beta * c.xp[p] * m);
            ug[b][r]  = vset1(c.gamma * c.xc[p] * m);
            acc[b][r] = vzero();
        }
    }
    const vf va = vset1(c.alpha);
#if M3_PREFETCH > 0 && !defined(M3_ISA_SCALAR) && !defined(M3_ISA_NEON)
    // pull the state rows M3_PREFETCH blocks ahead into L2 while this block computes
    if (p0 + PB * (M3_PREFETCH + 1) <= P) {
        const char * pf = (const char *) (c.S0 + (size_t) (p0 + PB * M3_PREFETCH) * N);
        for (int off = 0; off < PB * N * (int) sizeof(float); off += 64) _mm_prefetch(pf + off, _MM_HINT_T1);
    }
#endif
    for (int n = 0; n < N; n += W) {
        vf s[PB];
        for (int b = 0; b < PB; b++) s[b] = vmul(va, vload(c.S0 + (size_t) (p0 + b) * N + n));
        for (int r = 0; r < R; r++) {
            const vf kv = vload(c.kp + r * N + n);
            for (int b = 0; b < PB; b++) s[b] = vfma(ub[b][r], kv, s[b]);
            const vf kc = vload(c.kr + r * N + n);
            for (int b = 0; b < PB; b++) s[b] = vfma(ug[b][r], kc, s[b]);
        }
        for (int b = 0; b < PB; b++) vstore(c.S0 + (size_t) (p0 + b) * N + n, s[b]);
        for (int r = 0; r < R; r++) {
            const vf qv = vload(c.qr + r * N + n);
            for (int b = 0; b < PB; b++) acc[b][r] = vfma(s[b], qv, acc[b][r]);
        }
    }
    for (int b = 0; b < PB; b++) {
        for (int r = 0; r < R; r++) {
            c.yb[r * P + p0 + b] = vsum(acc[b][r]);
        }
    }
}

// y[p] = sum_r (yb_r[p] + D v_r[p]) * silu(z[p] mz_r[p]) * mo_r[p],  v_r = x mx_r   (vectorised over p)
template <bool MIMO>
static inline void m3_epilogue(const HeadCtx & c, int R, int P, float * y) {
    const vf vD = vset1(c.Dh);
    for (int p = 0; p < P; p += W) {
        const vf x = vload(c.xc + p), z = vload(c.zc + p);
        vf out = vzero();
        for (int r = 0; r < R; r++) {
            const vf v  = MIMO ? vmul(x, vload(c.mx + r * P + p)) : x;
            const vf yr = vfma(vD, v, vload(c.yb + r * P + p));
            const vf g  = MIMO ? vmul(vsilu(vmul(z, vload(c.mz + r * P + p))), vload(c.mo + r * P + p)) : vsilu(z);
            out = vfma(yr, g, out);
        }
        vstore(y + p, out);
    }
}

// RT > 0: compile-time rank. MIMO = false: SISO (no x/z/out projections).
template <int RT, bool MIMO>
static void mamba3_heads(const M3Dims & d, const M3Params & pr, M3State & st, const M3Step & in,
                         float * y, int h0, int h1) {
    const int R = RT > 0 ? RT : d.R;
    const int P = d.P, N = d.N, S = d.S;
    const int PB = RT > 0 ? (int) RowBlock<RT>::v : 1;

    // angle increments and the angle state are accumulated in double: they are summed over
    // the whole sequence, so fp32 rounding would drift (cost: S scalar ops per head)
    double th[SMAX];
    alignas(64) float af[SMAX], cs[SMAX], sn[SMAX];
    for (int s = 0; s < S; s++) {
        th[s] = tanh((double) in.ang[s]) * 3.14159265358979323846;
    }
    alignas(64) float qr[RMAX * NMAX];
    alignas(64) float kr[RMAX * NMAX];
    alignas(64) float yb[RMAX * PMAX];
    if (P % W || N % W || S % W || P > PMAX || N > NMAX || S > SMAX || R > RMAX) {
        fprintf(stderr, "m3: unsupported shape P=%d N=%d S=%d R=%d (need multiples of %d)\n", P, N, S, R, (int) W);
        abort();
    }

    for (int h = h0; h < h1; h++) {
        const float dt = in.dt[h];
        const float tr = sigmoidf(in.trap[h]);
        HeadCtx c;
        c.alpha = expf(in.adt[h]);
        c.beta  = (1.0f - tr) * dt * c.alpha;
        c.gamma = tr * dt;
        c.Dh    = pr.D[h];

        double * ang = st.ang + (size_t) h * S;
        for (int s = 0; s < S; s++) {
            double a = ang[s] + th[s] * (double) dt;
            a -= 6.28318530717958647692 * floor(a * 0.15915494309189533577);   // wrap to [0, 2*pi)
            ang[s] = a;
            af[s] = (float) a;
        }
        for (int s = 0; s < S; s += W) {
            vf vs, vc;
            vsincos(vload(af + s), &vs, &vc);
            vstore(sn + s, vs);
            vstore(cs + s, vc);
        }
        for (int r = 0; r < R; r++) {
            rope_bias(in.q + (size_t) r * N, pr.qb + ((size_t) h * R + r) * N, cs, sn, qr + r * N, N, S);
            rope_bias(in.k + (size_t) r * N, pr.kb + ((size_t) h * R + r) * N, cs, sn, kr + r * N, N, S);
        }
        c.kp = st.kst + (size_t) h * R * N;
        c.kr = kr;
        c.qr = qr;
        c.xp = st.xst + (size_t) h * P;
        c.xc = in.x + (size_t) h * P;
        c.zc = in.z + (size_t) h * P;
        c.mx = MIMO ? pr.mx + (size_t) h * R * P : nullptr;
        c.mz = MIMO ? pr.mz + (size_t) h * R * P : nullptr;
        c.mo = MIMO ? pr.mo + (size_t) h * R * P : nullptr;
        c.S0 = st.ssm + (size_t) h * P * N;
        c.yb = yb;

        int p = 0;
        if (PB > 1) {
            for (; p + PB <= P; p += PB) m3_rows<RT, MIMO, (RT > 0 ? (int) RowBlock<(RT > 0 ? RT : 1)>::v : 1)>(c, p, R, P, N);
        }
        for (; p < P; p++) m3_rows<RT, MIMO, 1>(c, p, R, P, N);
        m3_epilogue<MIMO>(c, R, P, y + (size_t) h * P);

        memcpy(st.kst + (size_t) h * R * N, kr, sizeof(float) * R * N);
        memcpy(st.xst + (size_t) h * P, c.xc, sizeof(float) * P);
    }
}

void mamba3_siso(const M3Dims & d, const M3Params & pr, M3State & st, const M3Step & in, float * y, int h0, int h1) {
    mamba3_heads<1, false>(d, pr, st, in, y, h0, h1);
}

void mamba3_mimo(const M3Dims & d, const M3Params & pr, M3State & st, const M3Step & in, float * y, int h0, int h1) {
    switch (d.R) {
        case 1:  mamba3_heads<1, true>(d, pr, st, in, y, h0, h1); break;
        case 2:  mamba3_heads<2, true>(d, pr, st, in, y, h0, h1); break;
        case 4:  mamba3_heads<4, true>(d, pr, st, in, y, h0, h1); break;
        case 8:  mamba3_heads<8, true>(d, pr, st, in, y, h0, h1); break;
        default: mamba3_heads<0, true>(d, pr, st, in, y, h0, h1); break;
    }
}

// Mamba-2: S = exp(dt*A) S + dt x B^T ; y = S C (+ D x, * silu(z)); PB rows per block
template <int PB>
static inline void m2_rows(float * S0, const float * B, const float * C, const float * x, const float * z,
                           float dt, vf va, float Dh, float * y, int p0, int P, int N, bool dz) {
    vf vx[PB], acc[PB];
    for (int b = 0; b < PB; b++) { vx[b] = vset1(x[p0 + b] * dt); acc[b] = vzero(); }
#if M3_PREFETCH > 0 && !defined(M3_ISA_SCALAR) && !defined(M3_ISA_NEON)
    if (p0 + PB * (M3_PREFETCH + 1) <= P) {  // same prefetch as the Mamba-3 rows
        const char * pf = (const char *) (S0 + (size_t) (p0 + PB * M3_PREFETCH) * N);
        for (int off = 0; off < PB * N * (int) sizeof(float); off += 64) _mm_prefetch(pf + off, _MM_HINT_T1);
    }
#endif
    for (int n = 0; n < N; n += W) {
        const vf bv = vload(B + n), cv = vload(C + n);
        for (int b = 0; b < PB; b++) {
            float * Sp = S0 + (size_t) (p0 + b) * N + n;
            const vf s = vfma(vx[b], bv, vmul(va, vload(Sp)));
            vstore(Sp, s);
            acc[b] = vfma(s, cv, acc[b]);
        }
    }
    for (int b = 0; b < PB; b++) {
        y[p0 + b] = vsum(acc[b]);
    }
    (void) z; (void) Dh; (void) dz; (void) P;
}

void mamba2(const M3Dims & d, const M2Params & pr, float * ssm, const M2Step & in, float * y, int h0, int h1, bool dz) {
    const int P = d.P, N = d.N;
    for (int h = h0; h < h1; h++) {
        const float dt = softplusf(in.dt[h] + pr.dt_bias[h]);
        const vf va = vset1(expf(dt * pr.A[h]));
        float * S0 = ssm + (size_t) h * P * N;
        const float * x = in.x + (size_t) h * P;
        const float * z = in.z + (size_t) h * P;
        int p = 0;
        for (; p + PB_M2 <= P; p += PB_M2) m2_rows<PB_M2>(S0, in.B, in.C, x, z, dt, va, pr.D[h], y + (size_t) h * P, p, P, N, dz);
        for (; p < P; p++) m2_rows<1>(S0, in.B, in.C, x, z, dt, va, pr.D[h], y + (size_t) h * P, p, P, N, dz);
        if (dz) {  // y = (y + D x) * silu(z), vectorised over p
            float * yh = y + (size_t) h * P;
            const vf vD = vset1(pr.D[h]);
            for (int q = 0; q < P; q += W) {
                vstore(yh + q, vmul(vfma(vD, vload(x + q), vload(yh + q)), vsilu(vload(z + q))));
            }
        }
    }
}

}  // namespace M3_NS
