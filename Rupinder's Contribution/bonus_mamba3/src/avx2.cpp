// AVX2 + FMA decode steps (8 floats per register). Requires N % 8 == 0.
// The state row S[p, :] is read and written exactly once per step and the output dot
// products with C_r are fused into the same pass, so the state makes one round trip.
#include <immintrin.h>

#include "m3.h"

static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

template <int R, bool PROJ>
static void m3_step(const Params& p, State& st, const Input& in, Rope mode, float* y) {
    const int H = p.H, P = p.P, N = p.N;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        alignas(32) float Brot[R * kMaxN];
        alignas(32) float Crot[R * kMaxN];
        rope_head<R>(p, h, mode, in, &st.ang[(size_t)h * p.S], Brot, Crot);
        const float dt = in.dt[h];
        const float alpha = std::exp(in.A[h] * dt);
        const float beta = (1.f - in.trap[h]) * dt * alpha;
        const float gamma = in.trap[h] * dt;
        const float* x = in.x + (size_t)h * P;
        const float* z = in.z + (size_t)h * P;
        float* v = &st.v[(size_t)h * P];
        float* k = &st.k[(size_t)h * R * N];
        float* S = &st.s[(size_t)h * P * N];
        const __m256 va = _mm256_set1_ps(alpha);
        for (int pp = 0; pp < P; ++pp) {
            float xpj[R];
            __m256 vg[R], vb[R], acc[R];
            for (int r = 0; r < R; ++r) {
                xpj[r] = PROJ ? p.xp[((size_t)r * H + h) * P + pp] : 1.f;
                vg[r] = _mm256_set1_ps(x[pp] * xpj[r] * gamma);
                vb[r] = _mm256_set1_ps(v[pp] * xpj[r] * beta);
                acc[r] = _mm256_setzero_ps();
            }
            float* row = S + (size_t)pp * N;
            for (int n = 0; n < N; n += 8) {
                __m256 s = _mm256_mul_ps(va, _mm256_loadu_ps(row + n));
                for (int r = 0; r < R; ++r) {
                    s = _mm256_fmadd_ps(vg[r], _mm256_load_ps(Brot + r * N + n), s);
                    s = _mm256_fmadd_ps(vb[r], _mm256_loadu_ps(k + r * N + n), s);
                }
                _mm256_storeu_ps(row + n, s);
                for (int r = 0; r < R; ++r) acc[r] = _mm256_fmadd_ps(s, _mm256_load_ps(Crot + r * N + n), acc[r]);
            }
            float out = 0.f;
            for (int r = 0; r < R; ++r) {
                const size_t ix = ((size_t)r * H + h) * P + pp;
                const float zz = z[pp] * (PROJ ? p.zp[ix] : 1.f);
                const float o = (hsum(acc[r]) + p.D[h] * x[pp] * xpj[r]) * silu(zz);
                out += o * (PROJ ? p.op[ix] : 1.f);
            }
            y[(size_t)h * P + pp] = out;
        }
        std::memcpy(k, Brot, sizeof(float) * R * N);
        std::memcpy(v, x, sizeof(float) * P);
    }
}

void siso_avx2(const Params& p, State& st, const Input& in, float* y) { m3_step<1, false>(p, st, in, Rope::Pairwise, y); }
void mimo_avx2(const Params& p, State& st, const Input& in, float* y) { m3_step<4, true>(p, st, in, Rope::Halved, y); }

void mamba2_avx2(const Params& p, State& st, const Input& in, float* y) {
    const int H = p.H, P = p.P, N = p.N;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        const float dt = in.dt[h];
        const __m256 vdA = _mm256_set1_ps(std::exp(dt * in.A[h]));
        float* S = &st.s[(size_t)h * P * N];
        for (int pp = 0; pp < P; ++pp) {
            const float x = in.x[(size_t)h * P + pp];
            const __m256 vxd = _mm256_set1_ps(x * dt);
            float* row = S + (size_t)pp * N;
            __m256 acc = _mm256_setzero_ps();
            for (int n = 0; n < N; n += 8) {
                __m256 s = _mm256_fmadd_ps(vxd, _mm256_loadu_ps(in.B + n), _mm256_mul_ps(vdA, _mm256_loadu_ps(row + n)));
                _mm256_storeu_ps(row + n, s);
                acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(in.C + n), acc);
            }
            const float zz = in.z[(size_t)h * P + pp];
            y[(size_t)h * P + pp] = (hsum(acc) + p.D[h] * x) * silu(zz);
        }
    }
}
