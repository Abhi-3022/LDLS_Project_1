// Check every kernel against the float64 reference over all steps of case.bin.
// Usage: verify <case.bin> [tolerance]
// Error per step = max_p |y - y_ref| / max_p |y_ref|  (relative to that step's output scale).
// The final state is compared the same way. PASS if both stay below the tolerance.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "m3.h"

struct Case {
    Params p;
    int T = 0;
    std::vector<float> steps;                       // T x per-step block
    size_t step_len = 0;
    std::vector<double> y_ref[3], s_ref[3];         // siso, mimo, mamba2
};

static Case load_case(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic, 8) != "M3REF001") { std::fprintf(stderr, "bad file\n"); std::exit(1); }
    int32_t hdr[6];
    f.read(reinterpret_cast<char*>(hdr), sizeof hdr);
    Case c;
    Params& p = c.p;
    p.H = hdr[0]; p.P = hdr[1]; p.N = hdr[2]; p.R = hdr[3]; p.S = hdr[4]; c.T = hdr[5];
    auto rd = [&](auto& v, size_t n) { v.resize(n); f.read(reinterpret_cast<char*>(v.data()), n * sizeof(v[0])); };
    rd(p.D, p.H);
    rd(p.Bb, (size_t)p.H * p.R * p.N);
    rd(p.Cb, (size_t)p.H * p.R * p.N);
    rd(p.xp, (size_t)p.R * p.H * p.P);
    rd(p.zp, (size_t)p.R * p.H * p.P);
    rd(p.op, (size_t)p.R * p.H * p.P);
    c.step_len = 2 * (size_t)p.H * p.P + 2 * (size_t)p.R * p.N + 3 * (size_t)p.H + p.S;
    rd(c.steps, c.step_len * c.T);
    for (int v = 0; v < 3; ++v) {
        rd(c.y_ref[v], (size_t)c.T * p.H * p.P);
        rd(c.s_ref[v], (size_t)p.H * p.P * p.N);
    }
    if (!f) { std::fprintf(stderr, "truncated file\n"); std::exit(1); }
    return c;
}

static Input step_input(const Case& c, int t) {
    const Params& p = c.p;
    const float* b = c.steps.data() + c.step_len * t;
    Input in;
    in.x = b; b += (size_t)p.H * p.P;
    in.z = b; b += (size_t)p.H * p.P;
    in.B = b; b += (size_t)p.R * p.N;
    in.C = b; b += (size_t)p.R * p.N;
    in.dt = b; b += p.H;
    in.A = b; b += p.H;
    in.trap = b; b += p.H;
    in.ang = b;
    return in;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: verify case.bin [tol]\n"); return 2; }
    const double tol = argc > 2 ? std::atof(argv[2]) : 1e-4;
    Case c = load_case(argv[1]);
    const Params& p = c.p;
    std::printf("case: H=%d P=%d N=%d R=%d rope_angles=%d steps=%d, tolerance %.1e (relative)\n",
                p.H, p.P, p.N, p.R, p.S, c.T, tol);
    std::printf("%-7s %-6s %14s %10s %16s %16s  %s\n", "kernel", "impl", "worst_y_err", "at_step",
                "final_y_err", "final_state_err", "result");
    bool all_ok = true;
    std::vector<float> y((size_t)p.H * p.P);
    for (const Kernel& k : kKernels) {
        const int v = k.rank == 1 ? 0 : k.rank == 4 ? 1 : 2;
        State st;
        st.init(p, k.rank);
        double worst = 0, last = 0;
        int worst_t = 0;
        for (int t = 0; t < c.T; ++t) {
            k.fn(p, st, step_input(c, t), y.data());
            const double* yr = c.y_ref[v].data() + (size_t)t * p.H * p.P;
            double num = 0, den = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                num = std::max(num, std::fabs((double)y[i] - yr[i]));
                den = std::max(den, std::fabs(yr[i]));
            }
            last = num / den;
            if (last > worst) { worst = last; worst_t = t; }
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < st.s.size(); ++i) {
            num = std::max(num, std::fabs((double)st.s[i] - c.s_ref[v][i]));
            den = std::max(den, std::fabs(c.s_ref[v][i]));
        }
        const double serr = num / den;
        const bool ok = worst <= tol && serr <= tol;
        all_ok &= ok;
        std::printf("%-7s %-6s %14.3e %10d %16.3e %16.3e  %s\n", k.name, k.impl, worst, worst_t, last, serr,
                    ok ? "PASS" : "FAIL");
    }
    return all_ok ? 0 : 1;
}
