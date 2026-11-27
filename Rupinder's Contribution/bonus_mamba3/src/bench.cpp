// Time per decode step for SISO, MIMO (rank 4) and Mamba-2 at the same state size.
// Usage: bench [--heads 64] [--headdim 64] [--dstate 128] [--layers 1,24] [--min-time 1.0] [--reps 3]
//   --layers 1    : one layer's state is reused every step, so it stays in cache
//   --layers 24   : steps cycle through 24 layers' states (48 MiB at the defaults; 24 layers of this shape =
//                   state-spaces/mamba3-siso-1.5b / mamba3-mimo-1.5b), larger than a laptop L3 cache, so each
//                   step streams its state from DRAM as in real decoding
// Threads: OMP_NUM_THREADS (heads are split across threads). Output: CSV on stdout.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "m3.h"

struct OwnedInput {
    std::vector<float> x, z, B, C, dt, A, trap, ang;
    Input view() const { return {x.data(), z.data(), B.data(), C.data(), dt.data(), A.data(), trap.data(), ang.data()}; }
};

int main(int argc, char** argv) {
    int H = 64, P = 64, N = 128, reps = 3;
    double min_time = 1.0;
    std::string layers_arg = "1,24";
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--heads") H = std::atoi(argv[i + 1]);
        else if (a == "--headdim") P = std::atoi(argv[i + 1]);
        else if (a == "--dstate") N = std::atoi(argv[i + 1]);
        else if (a == "--layers") layers_arg = argv[i + 1];
        else if (a == "--min-time") min_time = std::atof(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
    }
    if (N % 8 || N > kMaxN) { std::fprintf(stderr, "d_state must be a multiple of 8 and <= %d\n", kMaxN); return 2; }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, size_t n, float scale, float offset) {
        v.resize(n);
        for (auto& e : v) e = offset + scale * nd(rng);
    };
    Params p;
    p.H = H; p.P = P; p.N = N; p.R = kMaxRank; p.S = N / 4;
    fill(p.D, H, 1, 0);
    fill(p.Bb, (size_t)H * p.R * N, 0.1f, 1);
    fill(p.Cb, (size_t)H * p.R * N, 0.1f, 1);
    fill(p.xp, (size_t)p.R * H * P, 0.1f, 0.25f);
    fill(p.zp, (size_t)p.R * H * P, 0.1f, 1);
    fill(p.op, (size_t)p.R * H * P, 0.1f, 0.25f);
    std::vector<OwnedInput> pool(16);
    for (auto& in : pool) {
        fill(in.x, (size_t)H * P, 1, 0);
        fill(in.z, (size_t)H * P, 1, 0);
        fill(in.B, (size_t)p.R * N, 1, 0);
        fill(in.C, (size_t)p.R * N, 1, 0);
        in.dt.resize(H); in.A.resize(H); in.trap.resize(H);
        for (int h = 0; h < H; ++h) {
            in.dt[h] = 0.001f + 0.1f * ud(rng);
            in.A[h] = -(0.5f + ud(rng));
            in.trap[h] = ud(rng);
        }
        fill(in.ang, p.S, 1, 0);
    }
    std::vector<Input> views;
    for (auto& in : pool) views.push_back(in.view());
    std::vector<float> y((size_t)H * P);

    std::printf("# H=%d P=%d N=%d state/layer=%.2f MiB threads=%d min_time=%.2fs reps=%d\n", H, P, N,
                H * (double)P * N * 4 / (1 << 20), threads, min_time, reps);
    std::printf("kernel,impl,threads,layers,state_total_MiB,ns_per_step_mean,ns_per_step_min,ns_per_step_max,"
                "flops_per_step,bytes_per_step,arith_intensity,GFLOPs,GBps\n");

    std::stringstream ss(layers_arg);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const int L = std::atoi(tok.c_str());
        for (const Kernel& k : kKernels) {
            std::vector<State> st(L);
            for (auto& s : st) s.init(p, k.rank);
            size_t step = 0;
            auto pass = [&]() {
                for (int l = 0; l < L; ++l) k.fn(p, st[l], views[step++ % views.size()], y.data());
            };
            for (int w = 0; w < (L == 1 ? 200 : 2); ++w) pass();   // warm-up: touch every state page
            double mean = 0, mn = 1e300, mx = 0;
            for (int r = 0; r < reps; ++r) {
                long passes = 0;
                const auto t0 = std::chrono::steady_clock::now();
                double el = 0;
                do {
                    pass();
                    ++passes;
                    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                } while (el < min_time);
                const double ns = el * 1e9 / ((double)passes * L);
                mean += ns / reps; mn = std::min(mn, ns); mx = std::max(mx, ns);
            }
            const double fl = step_flops(p, k.rank), by = step_bytes(p, k.rank);
            std::printf("%s,%s,%d,%d,%.1f,%.1f,%.1f,%.1f,%.0f,%.0f,%.3f,%.2f,%.2f\n", k.name, k.impl, threads, L,
                        L * H * (double)P * N * 4 / (1 << 20), mean, mn, mx, fl, by, fl / by, fl / mean, by / mean);
            std::fflush(stdout);
        }
    }
    return 0;
}
