// ggml integration (stretch goal) and the ggml Mamba-2 baseline.
//
// Part A: ggml's own GGML_OP_SSM_SCAN (the operator llama.cpp uses for Mamba-2) on a
//         single decode token with the same layer shape. Checked against the official
//         reference (selective_state_update_ref without D/z) and against our Mamba-2
//         kernel, and timed per step.
// Part B: our Mamba-3 SISO and MIMO step wrapped as a ggml custom op (ggml_custom_4d)
//         inside a small ggml graph. 1000 steps on the reference data: graph output must
//         equal the direct kernel call bit-for-bit and match the reference; then timed.
//
// usage: ggml_m3 <data_dir> <out.csv> [threads...]
#include "common.h"
#include "m3_kernels.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstring>

static double rel_err(const float * a, const double * ref, size_t n, double * max_ref_out = nullptr) {
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { e = std::max(e, std::fabs(a[i] - ref[i])); m = std::max(m, std::fabs(ref[i])); }
    if (max_ref_out) *max_ref_out = m;
    return e / m;
}

static const M3Impl * best_impl() {
    static M3Impl impls[8];
    const int n = m3_impls(impls, 8);
    const M3Impl * best = &impls[0];
    for (int i = 0; i < n; i++) if (impls[i].supported) best = &impls[i];
    return best;  // last supported = widest SIMD
}

// ---------------- Part A: ggml SSM_SCAN (Mamba-2) ----------------
static void part_a(const std::string & dir, FILE * csv, const std::vector<int> & threads) {
    const Case c = load_case(dir);
    const int H = c.H, P = c.P, N = c.N, T = c.T;
    auto v = load_bin<float>(dir, "v", (size_t) T * H * P), z = load_bin<float>(dir, "z", (size_t) T * H * P);
    auto B = load_bin<float>(dir, "B", (size_t) T * N), C = load_bin<float>(dir, "C", (size_t) T * N);
    auto dt = load_bin<float>(dir, "dt", (size_t) T * H);
    auto A = load_bin<float>(dir, "A", H), D = load_bin<float>(dir, "D", H), dtb = load_bin<float>(dir, "dt_bias", H);
    auto y_ref = load_bin<double>(dir, "y_core_ref", (size_t) T * H * P);

    ggml_init_params ip = {(size_t) 256 << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, P, H, 1);
    ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, P, H, 1, 1);
    ggml_tensor * tdt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 1, 1);
    ggml_tensor * tA  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, H);
    ggml_tensor * tB  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, 1, 1, 1);
    ggml_tensor * tC  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N, 1, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ((int32_t *) ids->data)[0] = 0;
    memcpy(tA->data, A.data(), sizeof(float) * H);
    ggml_tensor * out = ggml_ssm_scan(ctx, s, x, tdt, tA, tB, tC, ids, 1);  // [y (P*H) | new state (N*P*H)]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const size_t SE = (size_t) H * P * N;

    std::vector<float> ssm_ours(SE), y_ours((size_t) H * P), zero_z((size_t) H * P, 0.f), Dz(H, 0.f);
    M3Dims d{H, P, N, 1, 0};
    M2Params pr{A.data(), D.data(), dtb.data()};
    const M3Impl * im = best_impl();

    for (int nth : threads) {
        memset(s->data, 0, ggml_nbytes(s));
        std::fill(ssm_ours.begin(), ssm_ours.end(), 0.f);
        double t_ggml = 0, t_ours = 0, e_ggml = 0, e_ours = 0, max_ref = 0, diff = 0;
        for (int t = 0; t < T; t++) {
            memcpy(x->data, &v[(size_t) t * H * P], sizeof(float) * H * P);
            for (int h = 0; h < H; h++) ((float *) tdt->data)[h] = dt[(size_t) t * H + h] + dtb[h];  // llama.cpp adds dt_bias before the op
            memcpy(tB->data, &B[(size_t) t * N], sizeof(float) * N);
            memcpy(tC->data, &C[(size_t) t * N], sizeof(float) * N);
            double t0 = now_s();
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            t_ggml += now_s() - t0;
            const float * yg = (const float *) out->data;
            memcpy(s->data, yg + (size_t) H * P, sizeof(float) * SE);  // feed the new state back (llama.cpp does this with a ggml_cpy)

            M2Step in{&v[(size_t) t * H * P], zero_z.data(), &B[(size_t) t * N], &C[(size_t) t * N], &dt[(size_t) t * H]};
            t0 = now_s();
            par_heads(H, nth, [&](int h0, int h1) { im->m2(d, pr, ssm_ours.data(), in, y_ours.data(), h0, h1, false); });
            t_ours += now_s() - t0;

            double mr;
            e_ggml = std::max(e_ggml, rel_err(yg, &y_ref[(size_t) t * H * P], (size_t) H * P, &mr) * mr);
            e_ours = std::max(e_ours, rel_err(y_ours.data(), &y_ref[(size_t) t * H * P], (size_t) H * P) * mr);
            max_ref = std::max(max_ref, mr);
            for (size_t i = 0; i < (size_t) H * P; i++) diff = std::max(diff, (double) std::fabs(yg[i] - y_ours[i]));
        }
        printf("A  mamba2  thr=%-2d  ggml SSM_SCAN %8.2f us/step (rel err %.2e)   ours[%s] %8.2f us/step (rel err %.2e)   max|ggml-ours| %.2e\n",
               nth, t_ggml / T * 1e6, e_ggml / max_ref, im->name, t_ours / T * 1e6, e_ours / max_ref, diff);
        fprintf(csv, "A,mamba2,ggml_ssm_scan,%d,%d,%.3f,%.3e,,\n", nth, T, t_ggml / T * 1e6, e_ggml / max_ref);
        fprintf(csv, "A,mamba2,ours_%s,%d,%d,%.3f,%.3e,%.3e,\n", im->name, nth, T, t_ours / T * 1e6, e_ours / max_ref, diff);
    }
    ggml_free(ctx);
}

// ---------------- Part B: Mamba-3 step as a ggml custom op ----------------
struct M3OpCtx {
    M3Dims     d;
    M3Params   pr;
    M3State    st;
    m3_step_fn fn;
};

// called by every ggml worker thread; each one owns a disjoint range of heads
static void m3_custom_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    M3OpCtx * c = (M3OpCtx *) ud;
    M3Step in{(const float *) dst->src[0]->data, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
              (const float *) dst->src[3]->data, (const float *) dst->src[4]->data, (const float *) dst->src[5]->data,
              (const float *) dst->src[6]->data, (const float *) dst->src[7]->data};
    const int H = c->d.H;
    c->fn(c->d, c->pr, c->st, in, (float *) dst->data, H * ith / nth, H * (ith + 1) / nth);
}

static void part_b(const std::string & dir, const char * cs, bool mimo, FILE * csv, const std::vector<int> & threads) {
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
    const M3Impl * im = best_impl();

    ggml_init_params ip = {(size_t) 64 << 20, nullptr, false};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * args[8] = {
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, R), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, R),
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, H), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, H),
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H),    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H),
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H),    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S),
    };
    std::vector<double> g_ang((size_t) H * S);
    std::vector<float> g_ssm((size_t) H * P * N), g_kst((size_t) H * R * N), g_xst((size_t) H * P);
    M3OpCtx opc{{H, P, N, R, S}, {D.data(), qb.data(), kb.data(), mx.data(), mz.data(), mo.data()},
                {g_ang.data(), g_ssm.data(), g_kst.data(), g_xst.data()}, mimo ? im->mimo : im->siso};
    ggml_tensor * y = ggml_custom_4d(ctx, GGML_TYPE_F32, P, H, 1, 1, args, 8, m3_custom_op, GGML_N_TASKS_MAX, &opc);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    for (int nth : threads) {
        // direct-call twin with its own state
        std::vector<double> d_ang((size_t) H * S, 0.0);
        std::vector<float> d_ssm((size_t) H * P * N, 0.f), d_kst((size_t) H * R * N, 0.f),
            d_xst((size_t) H * P, 0.f), d_y((size_t) H * P);
        M3State dst{d_ang.data(), d_ssm.data(), d_kst.data(), d_xst.data()};
        std::fill(g_ang.begin(), g_ang.end(), 0.0); std::fill(g_ssm.begin(), g_ssm.end(), 0.f);
        std::fill(g_kst.begin(), g_kst.end(), 0.f); std::fill(g_xst.begin(), g_xst.end(), 0.f);
        double t_graph = 0, t_direct = 0, max_ref = 0, err = 0, bitdiff = 0;
        for (int t = 0; t < T; t++) {
            const float * src[8] = {&q[(size_t) t * R * N], &k[(size_t) t * R * N], &v[(size_t) t * H * P], &z[(size_t) t * H * P],
                                    &adt[(size_t) t * H], &dt[(size_t) t * H], &trap[(size_t) t * H], &ang[(size_t) t * S]};
            for (int a = 0; a < 8; a++) memcpy(args[a]->data, src[a], ggml_nbytes(args[a]));
            double t0 = now_s();
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            t_graph += now_s() - t0;

            M3Step in{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};
            t0 = now_s();
            par_heads(H, nth, [&](int h0, int h1) { opc.fn(opc.d, opc.pr, dst, in, d_y.data(), h0, h1); });
            t_direct += now_s() - t0;

            const float * yg = (const float *) y->data;
            double mr;
            err = std::max(err, rel_err(yg, &y_ref[(size_t) t * H * P], (size_t) H * P, &mr) * mr);
            max_ref = std::max(max_ref, mr);
            for (size_t i = 0; i < (size_t) H * P; i++) bitdiff = std::max(bitdiff, (double) std::fabs(yg[i] - d_y[i]));
        }
        printf("B  %-6s  thr=%-2d  ggml custom op %8.2f us/step   direct %8.2f us/step   graph overhead %6.2f us   "
               "rel err vs ref %.2e   max|graph-direct| %.1e\n",
               cs, nth, t_graph / T * 1e6, t_direct / T * 1e6, (t_graph - t_direct) / T * 1e6, err / max_ref, bitdiff);
        fprintf(csv, "B,%s,ggml_custom_%s,%d,%d,%.3f,%.3e,%.3e,%.3f\n", cs, im->name, nth, T, t_graph / T * 1e6, err / max_ref,
                bitdiff, t_direct / T * 1e6);
    }
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <data_dir> <out.csv> [threads...]\n", argv[0]); return 1; }
    std::vector<int> threads;
    for (int i = 3; i < argc; i++) threads.push_back(atoi(argv[i]));
    if (threads.empty()) threads = {1, 8};
    FILE * csv = fopen(argv[2], "w");
    if (!csv) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fprintf(csv, "part,case,impl,threads,steps,us_per_step,rel_err_vs_ref,max_abs_diff_vs_ours,direct_us_per_step\n");
    const std::string data = argv[1];
    part_a(data + "/mamba2", csv, threads);
    part_b(data + "/siso", "siso", false, csv, threads);
    part_b(data + "/mimo4", "mimo4", true, csv, threads);
    fclose(csv);
    return 0;
}
