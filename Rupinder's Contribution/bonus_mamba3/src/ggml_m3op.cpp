// Stretch goal: the Mamba-3 / Mamba-2 decode-step kernels wrapped as a ggml operator and run in a
// small ggml graph.
//
// ggml_m3_step() adds one GGML_OP_CUSTOM node (ggml_custom_4d) whose 8 source tensors are the
// per-token kernel inputs (x, z, B, C, dt, A, trap, angle projection) and whose output is y [P, H].
// The layer's fixed parameters and its decode cache (SSM state, RoPE angles, previous B and x) live
// outside the graph and are passed through userdata, the same way llama.cpp keeps the recurrent
// state in a cache next to the compute graph. The op calls the unmodified kernels from m3.h.
//
// Test: for each of the 6 kernels, the graph is built once and computed for all T steps of
// case.bin (inputs copied into the leaf tensors each step). Per step, the graph output is compared
//   (a) with the float64 reference, using the same relative error and tolerance as verify.cpp, and
//   (b) with a direct call of the same kernel on its own state copy: it must match bit for bit.
//
// Usage: ggml_m3op <case.bin> [tolerance] [n_threads]
// The kernels parallelise over heads with OpenMP, so for timings set OMP_NUM_THREADS and pin the
// process (run_bonus.ps1 runs it with OMP_NUM_THREADS=1 on one P-core, like the 1-thread benchmark).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <omp.h>

#include "ggml-cpu.h"
#include "ggml.h"
#include "m3.h"

// ---- case.bin reader (same format and layout as verify.cpp) ----
struct Case {
    Params p;
    int T = 0;
    std::vector<float> steps;
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

// ---- the ggml operator ----
struct M3OpCtx {                // userdata of one op node: one layer of one model variant
    const Params* p;
    State*        st;           // decode cache, updated in place by every compute
    const Kernel* k;
};

enum { SRC_X, SRC_Z, SRC_B, SRC_C, SRC_DT, SRC_A, SRC_TRAP, SRC_ANG, N_SRC };

static void m3_step_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;       // each kernel processes the whole layer on one thread (n_tasks = 1)
    const M3OpCtx* c = static_cast<const M3OpCtx*>(userdata);
    auto src = [&](int i) { return static_cast<const float*>(dst->src[i]->data); };
    Input in;
    in.x = src(SRC_X);   in.z = src(SRC_Z);
    in.B = src(SRC_B);   in.C = src(SRC_C);
    in.dt = src(SRC_DT); in.A = src(SRC_A); in.trap = src(SRC_TRAP);
    in.ang = src(SRC_ANG);
    c->k->fn(*c->p, *c->st, in, static_cast<float*>(dst->data));
}

// y [P, H] = m3_step(x [P, H], z [P, H], B [N, R], C [N, R], dt [H], A [H], trap [H], ang [S])
static ggml_tensor* ggml_m3_step(ggml_context* ctx, ggml_tensor* x, ggml_tensor* z, ggml_tensor* B,
                                 ggml_tensor* C, ggml_tensor* dt, ggml_tensor* A, ggml_tensor* trap,
                                 ggml_tensor* ang, M3OpCtx* opctx) {
    const Params& p = *opctx->p;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == p.P && x->ne[1] == p.H);
    GGML_ASSERT(ggml_are_same_shape(x, z));
    GGML_ASSERT(B->ne[0] == p.N && ggml_are_same_shape(B, C));
    GGML_ASSERT(ggml_nelements(dt) == p.H && ggml_nelements(A) == p.H && ggml_nelements(trap) == p.H);
    GGML_ASSERT(ggml_nelements(ang) == p.S);
    ggml_tensor* args[N_SRC] = {x, z, B, C, dt, A, trap, ang};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, p.P, p.H, 1, 1, args, N_SRC, m3_step_compute, 1, opctx);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ggml_m3op case.bin [tol] [n_threads]\n"); return 2; }
    const double tol = argc > 2 ? std::atof(argv[2]) : 1e-4;
    const int n_threads = argc > 3 ? std::atoi(argv[3]) : 1;
    const Case c = load_case(argv[1]);
    const Params& p = c.p;
    std::printf("case: H=%d P=%d N=%d R=%d rope_angles=%d steps=%d, tolerance %.1e (relative), ggml threads %d, "
                "OpenMP threads in the kernels %d\n",
                p.H, p.P, p.N, p.R, p.S, c.T, tol, n_threads, omp_get_max_threads());
    std::printf("%-7s %-6s %14s %10s %16s %16s %15s %10s %10s  %s\n", "kernel", "impl", "worst_y_err", "at_step",
                "final_y_err", "final_state_err", "graph_vs_direct", "graph_us", "direct_us", "result");

    bool all_ok = true;
    const size_t HP = (size_t)p.H * p.P;
    for (const Kernel& k : kKernels) {
        const int v = k.rank == 1 ? 0 : k.rank == 4 ? 1 : 2;

        // graph: 8 leaf tensors -> one custom op node -> y
        ggml_init_params ip = {(size_t)16 << 20, nullptr, false};
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.P, p.H);
        ggml_tensor* tz = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.P, p.H);
        ggml_tensor* tB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.N, p.R);
        ggml_tensor* tC = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.N, p.R);
        ggml_tensor* tdt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.H);
        ggml_tensor* tA = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.H);
        ggml_tensor* ttrap = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.H);
        ggml_tensor* tang = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.S);
        State st_graph;
        st_graph.init(p, k.rank);
        M3OpCtx opctx{&p, &st_graph, &k};
        ggml_tensor* ty = ggml_m3_step(ctx, tx, tz, tB, tC, tdt, tA, ttrap, tang, &opctx);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ty);
        if (&k == &kKernels[0])
            std::printf("graph: %d node(s), op %s with %d source tensors -> y [%lld, %lld]\n",
                        ggml_graph_n_nodes(gf), ggml_op_desc(ggml_graph_node(gf, 0)), N_SRC,
                        (long long)ty->ne[0], (long long)ty->ne[1]);

        // direct-call twin on its own state
        State st_direct;
        st_direct.init(p, k.rank);
        std::vector<float> y_direct(HP);

        double worst = 0, last = 0, max_diff = 0, t_graph = 0, t_direct = 0;
        int worst_t = 0;
        for (int t = 0; t < c.T; ++t) {
            const Input in = step_input(c, t);
            std::memcpy(tx->data, in.x, ggml_nbytes(tx));
            std::memcpy(tz->data, in.z, ggml_nbytes(tz));
            std::memcpy(tB->data, in.B, ggml_nbytes(tB));
            std::memcpy(tC->data, in.C, ggml_nbytes(tC));
            std::memcpy(tdt->data, in.dt, ggml_nbytes(tdt));
            std::memcpy(tA->data, in.A, ggml_nbytes(tA));
            std::memcpy(ttrap->data, in.trap, ggml_nbytes(ttrap));
            std::memcpy(tang->data, in.ang, ggml_nbytes(tang));

            double t0 = now_s();
            if (ggml_graph_compute_with_ctx(ctx, gf, n_threads) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "graph compute failed\n");
                return 1;
            }
            t_graph += now_s() - t0;
            t0 = now_s();
            k.fn(p, st_direct, in, y_direct.data());
            t_direct += now_s() - t0;

            const float* yg = static_cast<const float*>(ty->data);
            const double* yr = c.y_ref[v].data() + (size_t)t * HP;
            double num = 0, den = 0;
            for (size_t i = 0; i < HP; ++i) {
                num = std::max(num, std::fabs((double)yg[i] - yr[i]));
                den = std::max(den, std::fabs(yr[i]));
                max_diff = std::max(max_diff, (double)std::fabs(yg[i] - y_direct[i]));
            }
            last = num / den;
            if (last > worst) { worst = last; worst_t = t; }
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < st_graph.s.size(); ++i) {
            num = std::max(num, std::fabs((double)st_graph.s[i] - c.s_ref[v][i]));
            den = std::max(den, std::fabs(c.s_ref[v][i]));
            max_diff = std::max(max_diff, (double)std::fabs(st_graph.s[i] - st_direct.s[i]));
        }
        const double serr = num / den;
        const bool ok = worst <= tol && serr <= tol && max_diff == 0.0;
        all_ok &= ok;
        std::printf("%-7s %-6s %14.3e %10d %16.3e %16.3e %15.1e %10.2f %10.2f  %s\n", k.name, k.impl, worst, worst_t,
                    last, serr, max_diff, t_graph / c.T * 1e6, t_direct / c.T * 1e6, ok ? "PASS" : "FAIL");
        ggml_free(ctx);
    }
    std::printf("PASS = graph output within tolerance of the fp64 reference at every step, final state within\n"
                "tolerance, and graph output and state bit-identical to a direct kernel call (graph_vs_direct = 0).\n");
    return all_ok ? 0 : 1;
}
