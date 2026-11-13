// Time per decode step (one layer) for Mamba-2, Mamba-3 SISO and Mamba-3 MIMO
// at the same state size, per ISA and thread count, in two memory regimes:
//   hot  : one layer's state is reused every step (stays in L2/L3)
//   cold : steps cycle through a pool of layer states much larger than L3, so the
//          state streams from DRAM every step (what a deep model sees on a laptop,
//          where L3 is 8-36 MB and 48 layers of state are ~96 MB)
//
// usage: bench <out.csv> [--isa a,b] [--threads 1,2,..] [--variants m2,siso,mimo2,mimo4,mimo8]
//              [--regimes hot,cold] [--reps 3] [--H 64 --P 64 --N 128 --S 32] [--pool_mib 512]
#include "common.h"
#include "m3_kernels.h"

#include <cstring>
#include <random>
#include <sstream>

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) out.push_back(t);
    return out;
}

struct Variant { std::string name; int R; };  // R = 0: Mamba-2

// analytic FLOPs per state element per step (see README):
//   Mamba-2      : a*S (1) + fma x*B (2) + fma readout (2)                       = 5
//   Mamba-3 rank R: a*S (1) + 2R fma (prev + current outer products, 4R) + R fma readouts (2R) = 1 + 6R
static double flops_per_elem(int R) { return R == 0 ? 5.0 : 1.0 + 6.0 * R; }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <out.csv> [options]\n", argv[0]); return 1; }
    std::string isa_s = "scalar,avx2,avx512", thr_s = "1,2,4,8,16,32", var_s = "m2,siso,mimo2,mimo4,mimo8",
                reg_s = "hot,cold";
    int reps = 3, H = 64, P = 64, N = 128, S = 32, pool_mib = 512;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--isa") isa_s = v; else if (k == "--threads") thr_s = v; else if (k == "--variants") var_s = v;
        else if (k == "--regimes") reg_s = v; else if (k == "--reps") reps = atoi(v.c_str());
        else if (k == "--H") H = atoi(v.c_str()); else if (k == "--P") P = atoi(v.c_str());
        else if (k == "--N") N = atoi(v.c_str()); else if (k == "--S") S = atoi(v.c_str());
        else if (k == "--pool_mib") pool_mib = atoi(v.c_str());
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    std::vector<Variant> variants;
    for (auto & v : split(var_s)) variants.push_back({v, v == "m2" ? 0 : v == "siso" ? 1 : atoi(v.c_str() + 4)});

    M3Impl impls[8];
    const int ni = m3_impls(impls, 8);
    FILE * csv = fopen(argv[1], "a");
    if (!csv) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(csv, 0, SEEK_END);
    if (ftell(csv) == 0)
        fprintf(csv, "variant,R,isa,threads,regime,rep,steps,us_per_step,gflops,gbps,flops_per_step,bytes_per_step,ai_flop_per_byte,H,P,N\n");

    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float scale, float off) { std::vector<float> v(n); for (auto & x : v) x = off + scale * nd(rng); return v; };

    const size_t state_elems = (size_t) H * P * N;
    for (const Variant & var : variants) {
        const int R = std::max(1, var.R);
        // parameters
        auto D = randv(H, 0.1f, 1.f), qb = randv((size_t) H * R * N, 0.1f, 1.f), kb = randv((size_t) H * R * N, 0.1f, 1.f);
        auto mx = randv((size_t) H * R * P, 0.1f, 1.f / R), mz = randv((size_t) H * R * P, 0.1f, 1.f), mo = randv((size_t) H * R * P, 0.1f, 1.f / R);
        std::vector<float> A(H, -4.f), dtb(H, -3.f);
        // a ring of 16 distinct token inputs
        const int NT = 16;
        auto q = randv((size_t) NT * R * N, 0.1f, 0.f), k = randv((size_t) NT * R * N, 0.1f, 0.f);
        auto x = randv((size_t) NT * H * P, 1.f, 0.f), z = randv((size_t) NT * H * P, 1.f, 0.f);
        auto adt = randv((size_t) NT * H, 0.01f, -0.05f), dt = randv((size_t) NT * H, 0.f, 0.05f);
        auto trap = randv((size_t) NT * H, 1.f, 0.f), ang = randv((size_t) NT * S, 1.f, 0.f);
        auto B = randv((size_t) NT * N, 0.1f, 0.f), C = randv((size_t) NT * N, 0.1f, 0.f);
        M3Dims d{H, P, N, R, S};
        M3Params pr{D.data(), qb.data(), kb.data(), mx.data(), mz.data(), mo.data()};
        M2Params p2{A.data(), D.data(), dtb.data()};

        const double flops = flops_per_elem(var.R) * state_elems;
        // bytes that must move per step if the per-layer state is not cache resident:
        // ssm state read + write, previous-token B (R*N) and x (P) state read + write, fp64 angle state
        const double bytes = 8.0 * state_elems + (var.R == 0 ? 0.0 : 8.0 * H * ((double) R * N + P) + 16.0 * H * S);

        for (const std::string & regime : split(reg_s)) {
            const size_t layer_bytes = state_elems * sizeof(float);
            const int L = regime == "hot" ? 1 : (int) std::max<size_t>(2, ((size_t) pool_mib << 20) / layer_bytes);
            std::vector<float> ssm((size_t) L * state_elems, 0.f);
            std::vector<double> angs((size_t) L * H * S, 0.0);
            std::vector<float> kst((size_t) L * H * R * N, 0.f), xst((size_t) L * H * P, 0.f), y((size_t) H * P);

            for (int ii = 0; ii < ni; ii++) {
                bool want = false;
                for (auto & s : split(isa_s)) want |= s == impls[ii].name;
                if (!want || !impls[ii].supported) continue;
                for (auto & ts : split(thr_s)) {
                    const int nth = atoi(ts.c_str());
                    auto step = [&](long it) {
                        const int l = (int) (it % L), t = (int) (it % NT);
                        if (var.R == 0) {
                            M2Step in{&x[(size_t) t * H * P], &z[(size_t) t * H * P], &B[(size_t) t * N], &C[(size_t) t * N], &dt[(size_t) t * H]};
                            float * s = &ssm[(size_t) l * state_elems];
                            par_heads(H, nth, [&](int h0, int h1) { impls[ii].m2(d, p2, s, in, y.data(), h0, h1, true); });
                        } else {
                            M3State st{&angs[(size_t) l * H * S], &ssm[(size_t) l * state_elems], &kst[(size_t) l * H * R * N], &xst[(size_t) l * H * P]};
                            M3Step in{&q[(size_t) t * R * N], &k[(size_t) t * R * N], &x[(size_t) t * H * P], &z[(size_t) t * H * P],
                                      &adt[(size_t) t * H], &dt[(size_t) t * H], &trap[(size_t) t * H], &ang[(size_t) t * S]};
                            m3_step_fn fn = var.R == 1 && var.name == "siso" ? impls[ii].siso : impls[ii].mimo;
                            par_heads(H, nth, [&](int h0, int h1) { fn(d, pr, st, in, y.data(), h0, h1); });
                        }
                    };
                    // warm up: one pass over the pool (or 50 steps), then time >= 0.3 s and >= 2 pool passes
                    for (long it = 0; it < std::max(L, 50); it++) step(it);
                    for (int rep = 1; rep <= reps; rep++) {
                        long it = 0;
                        const long min_steps = std::max<long>(2L * L, 100);
                        const double t0 = now_s();
                        double t1 = t0;
                        while (it < min_steps || t1 - t0 < 0.3) {
                            for (int j = 0; j < 20; j++) step(it++);
                            t1 = now_s();
                        }
                        const double sec = (t1 - t0) / it;
                        fprintf(csv, "%s,%d,%s,%d,%s,%d,%ld,%.3f,%.3f,%.3f,%.0f,%.0f,%.3f,%d,%d,%d\n", var.name.c_str(), var.R,
                                impls[ii].name, nth, regime.c_str(), rep, it, sec * 1e6, flops / sec * 1e-9, bytes / sec * 1e-9,
                                flops, bytes, flops / bytes, H, P, N);
                        fflush(csv);
                        printf("%-6s %-7s thr=%-2d %-4s rep=%d  %8.2f us/step  %7.2f GFLOP/s  %7.2f GB/s\n", var.name.c_str(),
                               impls[ii].name, nth, regime.c_str(), rep, sec * 1e6, flops / sec * 1e-9, bytes / sec * 1e-9);
                    }
                }
            }
        }
    }
    fclose(csv);
    return 0;
}
