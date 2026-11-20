// Portable C++ decode steps. Build with auto-vectorisation disabled
// (-fno-tree-vectorize -fno-tree-slp-vectorize) so this is a true scalar baseline.
#include "m3.h"

template <int R, bool PROJ>
static void m3_step(const Params& p, State& st, const Input& in, Rope mode, float* y) {
    const int H = p.H, P = p.P, N = p.N;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        float Brot[R * kMaxN], Crot[R * kMaxN];
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
        for (int pp = 0; pp < P; ++pp) {
            float xpj[R], xg[R], xb[R], acc[R];
            for (int r = 0; r < R; ++r) {
                xpj[r] = PROJ ? p.xp[((size_t)r * H + h) * P + pp] : 1.f;
                xg[r] = x[pp] * xpj[r] * gamma;
                xb[r] = v[pp] * xpj[r] * beta;
                acc[r] = 0.f;
            }
            float* row = S + (size_t)pp * N;
            for (int n = 0; n < N; ++n) {
                float s = alpha * row[n];
                for (int r = 0; r < R; ++r) s += xg[r] * Brot[r * N + n] + xb[r] * k[r * N + n];
                row[n] = s;
                for (int r = 0; r < R; ++r) acc[r] += s * Crot[r * N + n];
            }
            float out = 0.f;
            for (int r = 0; r < R; ++r) {
                const size_t ix = ((size_t)r * H + h) * P + pp;
                const float zz = z[pp] * (PROJ ? p.zp[ix] : 1.f);
                const float o = (acc[r] + p.D[h] * x[pp] * xpj[r]) * silu(zz);
                out += o * (PROJ ? p.op[ix] : 1.f);
            }
            y[(size_t)h * P + pp] = out;
        }
        std::memcpy(k, Brot, sizeof(float) * R * N);
        std::memcpy(v, x, sizeof(float) * P);
    }
}

void siso_plain(const Params& p, State& st, const Input& in, float* y) { m3_step<1, false>(p, st, in, Rope::Pairwise, y); }
void mimo_plain(const Params& p, State& st, const Input& in, float* y) { m3_step<4, true>(p, st, in, Rope::Halved, y); }

void mamba2_plain(const Params& p, State& st, const Input& in, float* y) {
    const int H = p.H, P = p.P, N = p.N;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        const float dt = in.dt[h];
        const float dA = std::exp(dt * in.A[h]);
        float* S = &st.s[(size_t)h * P * N];
        for (int pp = 0; pp < P; ++pp) {
            const float x = in.x[(size_t)h * P + pp];
            const float xd = x * dt;
            float* row = S + (size_t)pp * N;
            float acc = 0.f;
            for (int n = 0; n < N; ++n) {
                const float s = dA * row[n] + xd * in.B[n];
                row[n] = s;
                acc += s * in.C[n];
            }
            const float zz = in.z[(size_t)h * P + pp];
            y[(size_t)h * P + pp] = (acc + p.D[h] * x) * silu(zz);
        }
    }
}
