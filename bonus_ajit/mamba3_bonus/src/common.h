// Small helpers shared by verify.cpp, bench.cpp and ggml_m3.cpp
#pragma once
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

template <typename T>
static std::vector<T> load_bin(const std::string & dir, const std::string & name, size_t expect) {
    std::string path = dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::vector<T> v(expect);
    size_t got = fread(v.data(), sizeof(T), expect, f);
    if (got != expect || fgetc(f) != EOF) {
        fprintf(stderr, "%s: expected %zu elements\n", path.c_str(), expect);
        exit(1);
    }
    fclose(f);
    return v;
}

struct Case {  // contents of data/<case>/config.txt: H P N S R T
    int H, P, N, S, R, T;
};

static Case load_case(const std::string & dir) {
    Case c{};
    FILE * f = fopen((dir + "/config.txt").c_str(), "r");
    if (!f || fscanf(f, "%d %d %d %d %d %d", &c.H, &c.P, &c.N, &c.S, &c.R, &c.T) != 6) {
        fprintf(stderr, "bad %s/config.txt\n", dir.c_str());
        exit(1);
    }
    fclose(f);
    return c;
}

static inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// split heads over the OpenMP team and run fn(h0, h1) on each thread
template <typename F>
static void par_heads(int H, int nth, F fn) {
    if (nth <= 1) { fn(0, H); return; }
    #pragma omp parallel num_threads(nth)
    {
        const int t = omp_get_thread_num(), n = omp_get_num_threads();
        fn(H * t / n, H * (t + 1) / n);
    }
}
