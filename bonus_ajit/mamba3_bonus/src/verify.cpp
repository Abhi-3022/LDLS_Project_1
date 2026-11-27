// Correctness: run every kernel implementation for T (=1000) decode steps on the
// reference inputs and compare against the official PyTorch step reference (fp64).
//
// Tolerance (stated): over all T steps,
//   max_t,h,p |y - y_ref| <= TOL * max |y_ref|      and the same for the final ssm state,
// with TOL = 1e-5. For scale, PyTorch's own fp32 run of the reference is ~2e-7 away.
//
// usage: verify <data_dir> <out.csv> [threads...]
#include "common.h"
#include "m3_kernels.h"

#include <cmath>
#include <cstring>

static const double TOL = 1e-5;

struct Err {
    double max_abs = 0, max_ref = 0;
    void add(double a, double r) {
        max_abs = std::max(max_abs, std::fabs(a - r));
        max_ref = std::max(max_ref, std::fabs(r));
    }
    double rel() const { return max_abs / max_ref; }
};

static FILE * csv;

static void report(const char * cs, const char * isa, int nth, int T, const Err & y, const Err & s,
                   double ang_err, double torch32, double torch32_s, const std::vector<double> & err_t) {
    const bool pass = y.rel() <= TOL && s.rel() <= TOL;
    printf("%-7s %-7s thr=%-2d  y_rel=%.2e  state_rel=%.2e  angle_err=%.1e  (torch fp32: y %.2e state %.2e)  %s\n",
           cs, isa, nth, y.rel(), s.rel(), ang_err, torch32, torch32_s, pass ? "PASS" : "FAIL");
    fprintf(csv, "%s,%s,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.0e,%s,", cs, isa, nth, T, y.max_abs, y.max_ref,
            y.rel(), s.rel(), ang_err, torch32, torch32_s, TOL, pass ? "PASS" : "FAIL");
    // error after 1, 10, 100, T steps (running max), to show it does not grow
    for (size_t i = 0; i < err_t.size(); i++) fprintf(csv, "%.6e%s", err_t[i], i + 1 < err_t.size() ? ";" : "\n");
    fflush(csv);
}

static double torch32_rel(const std::vector<double> & ref, const std::vector<float> & t32) {  // max|t32-ref| / max|ref|
    Err e;
    for (size_t i = 0; i < ref.size(); i++) e.add(t32[i], ref[i]);
    return e.rel();
}

static void run_m3(const std::string & dir, const char * cs, bool mimo, const std::vector<int> & threads) {
    const Case c = load_case(dir);
    const int H = c.H, P = c.P, N = c.N, S = c.S, R = c.R, T = c.T;
    auto q = load_bin<float>(dir, "q", (size_t) T * R * N), k = load_bin<float>(dir, "k", (size_t) T * R * N);
    auto v = load_bin<float>(dir, "v", (size_t) T * H * P), z = load_bin<float>(dir, "z", (size_t) T * H * P);
    auto adt = load_bin<float>(dir, "adt", (size_t) T * H), dt = load_bin<float>(dir, "dt", (size_t) T * H);
    auto trap = load_bin<float>(dir, "trap", (size_t) T * H), ang = load_bin<float>(dir, "ang", (size_t) T * S);
    auto D = load_bin<float>(dir, "D", H);
    auto qb = load_bin<float>(dir, "qb", (size_t) H * R * N), kb = load_bin<float>(dir, "kb", (size_t) H * R * N);
    auto mx = load_bin<float>(dir, "mx", (size_t) H * R * P), mz = load_bin<float>(dir, "mz", (size_t) H * R * P);
    auto mo = load_bin<float>(dir, "mo", (size_t) H * R * P);
    auto y_ref = load_bin<double>(dir, "y_ref", (size_t) T * H * P);
    auto s_ref = load_bin<double>(dir, "ssm_ref", (size_t) H * P * N);
    auto a_ref = load_bin<double>(dir, "ang_ref", (size_t) H * S);
    auto y32 = load_bin<float>(dir, "y_torch32", (size_t) T * H * P);
    auto s32 = load_bin<float>(dir, "ssm_torch32", (size_t) H * P * N);
    const double t32 = torch32_rel(y_ref, y32), t32s = torch32_rel(s_ref, s32);

    M3Dims d{H, P, N, R, S};
    M3Params pr{D.data(), qb.data(), kb.data(), mx.data(), mz.data(), mo.data()};
    M3Impl impls[8];
    const int ni = m3_impls(impls, 8);
    for (int ii = 0; ii < ni; ii++) {
        if (!impls[ii].supported) continue;
        for (int nth : threads) {
            std::vector<double> ang_s((size_t) H * S, 0.0);
            std::vector<float> ssm((size_t) H * P * N, 0.f);
            std::vector<float> kst((size_t) H * R * N, 0.f), xst((size_t) H * P, 0.f), y((size_t) H * P);
            M3State st{ang_s.data(), ssm.data(), kst.data(), xst.data()};
            m3_step_fn fn = mimo ? impls[ii].mimo : impls[ii].siso;
            Err ey;
            std::vector<double> err_t;
            for (int t = 0; t < T; t++) {
                M3Step in{&q[(size_t) t * R * N], &k[(size_t) t * R * N], &v[(size_t) t * H * P], &z[(size_t) t * H * P],
                          &adt[(size_t) t * H], &dt[(size_t) t * H], &trap[(size_t) t * H], &ang[(size_t) t * S]};
                par_heads(H, nth, [&](int h0, int h1) { fn(d, pr, st, in, y.data(), h0, h1); });
                for (size_t i = 0; i < y.size(); i++) ey.add(y[i], y_ref[(size_t) t * H * P + i]);
                if (t == 0 || t == 9 || t == 99 || t == T - 1) err_t.push_back(ey.rel());
            }
            Err es;
            for (size_t i = 0; i < ssm.size(); i++) es.add(ssm[i], s_ref[i]);
            double ae = 0;  // angles compared modulo 2*pi (the MIMO reference does not wrap)
            for (size_t i = 0; i < ang_s.size(); i++)
                ae = std::max(ae, std::fabs(std::remainder(ang_s[i] - a_ref[i], 2 * M_PI)));
            report(cs, impls[ii].name, nth, T, ey, es, ae, t32, t32s, err_t);
        }
    }
}

static void run_m2(const std::string & dir, const std::vector<int> & threads) {
    const Case c = load_case(dir);
    const int H = c.H, P = c.P, N = c.N, T = c.T;
    auto v = load_bin<float>(dir, "v", (size_t) T * H * P), z = load_bin<float>(dir, "z", (size_t) T * H * P);
    auto B = load_bin<float>(dir, "B", (size_t) T * N), C = load_bin<float>(dir, "C", (size_t) T * N);
    auto dt = load_bin<float>(dir, "dt", (size_t) T * H);
    auto A = load_bin<float>(dir, "A", H), D = load_bin<float>(dir, "D", H), dtb = load_bin<float>(dir, "dt_bias", H);
    auto y_ref = load_bin<double>(dir, "y_ref", (size_t) T * H * P);
    auto s_ref = load_bin<double>(dir, "ssm_ref", (size_t) H * P * N);
    auto y32 = load_bin<float>(dir, "y_torch32", (size_t) T * H * P);
    auto s32 = load_bin<float>(dir, "ssm_torch32", (size_t) H * P * N);
    const double t32 = torch32_rel(y_ref, y32), t32s = torch32_rel(s_ref, s32);
    M3Dims d{H, P, N, 1, 0};
    M2Params pr{A.data(), D.data(), dtb.data()};
    M3Impl impls[8];
    const int ni = m3_impls(impls, 8);
    for (int ii = 0; ii < ni; ii++) {
        if (!impls[ii].supported) continue;
        for (int nth : threads) {
            std::vector<float> ssm((size_t) H * P * N, 0.f), y((size_t) H * P);
            Err ey;
            std::vector<double> err_t;
            for (int t = 0; t < T; t++) {
                M2Step in{&v[(size_t) t * H * P], &z[(size_t) t * H * P], &B[(size_t) t * N], &C[(size_t) t * N], &dt[(size_t) t * H]};
                par_heads(H, nth, [&](int h0, int h1) { impls[ii].m2(d, pr, ssm.data(), in, y.data(), h0, h1, true); });
                for (size_t i = 0; i < y.size(); i++) ey.add(y[i], y_ref[(size_t) t * H * P + i]);
                if (t == 0 || t == 9 || t == 99 || t == T - 1) err_t.push_back(ey.rel());
            }
            Err es;
            for (size_t i = 0; i < ssm.size(); i++) es.add(ssm[i], s_ref[i]);
            report("mamba2", impls[ii].name, nth, T, ey, es, 0.0, t32, t32s, err_t);
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <data_dir> <out.csv> [threads...]\n", argv[0]); return 1; }
    const std::string data = argv[1];
    std::vector<int> threads;
    for (int i = 3; i < argc; i++) threads.push_back(atoi(argv[i]));
    if (threads.empty()) threads = {1, 8};
    csv = fopen(argv[2], "w");
    fprintf(csv, "case,isa,threads,steps,max_abs_err,max_abs_ref,rel_err,state_rel_err,angle_err,torch_fp32_rel_err,torch_fp32_state_rel_err,tol,result,rel_err_at_t1;t10;t100;tT\n");
    run_m3(data + "/siso", "siso", false, threads);
    for (const char * cs : {"mimo1", "mimo2", "mimo4", "mimo8"}) run_m3(data + "/" + cs, cs, true, threads);
    run_m2(data + "/mamba2", threads);
    // layer 0 of the released state-spaces/mamba3-{siso,mimo}-1.5b checkpoints on real token embeddings
    FILE * probe = fopen((data + "/real_siso/config.txt").c_str(), "r");
    if (probe) {
        fclose(probe);
        run_m3(data + "/real_siso", "real_siso", false, threads);
        run_m3(data + "/real_mimo4", "real_mimo4", true, threads);
    }
    fclose(csv);
    return 0;
}
