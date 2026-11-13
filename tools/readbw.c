// Read-only memory bandwidth: OpenMP sum over a large float array.
// Decode in llama.cpp is (almost) pure weight reads, so this is the bandwidth
// the roofline should use alongside STREAM (whose kernels also write).
// The reduction uses 8 independent SIMD accumulators: a plain `s += a[i]` loop is
// bound by add latency (~2-4 GB/s per core), not by memory.
// Build: gcc -O3 -march=native -fopenmp readbw.c -o readbw
// Usage: ./readbw [GiB=2] [iters=20]   -> prints best GB/s (1e9 bytes)
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static float sum_range(const float * a, size_t n) {
#if defined(__AVX512F__)
    __m512 acc[8];
    for (int j = 0; j < 8; j++) acc[j] = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 128 <= n; i += 128)
        for (int j = 0; j < 8; j++) acc[j] = _mm512_add_ps(acc[j], _mm512_load_ps(a + i + 16*j));
    for (int j = 1; j < 8; j++) acc[0] = _mm512_add_ps(acc[0], acc[j]);
    float s = _mm512_reduce_add_ps(acc[0]);
#elif defined(__AVX2__)
    __m256 acc[8];
    for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= n; i += 64)
        for (int j = 0; j < 8; j++) acc[j] = _mm256_add_ps(acc[j], _mm256_load_ps(a + i + 8*j));
    for (int j = 1; j < 8; j++) acc[0] = _mm256_add_ps(acc[0], acc[j]);
    float t[8]; _mm256_storeu_ps(t, acc[0]);
    float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
#else
    float s = 0.0f; size_t i = 0;
#endif
    for (; i < n; i++) s += a[i];
    return s;
}

int main(int argc, char ** argv) {
    const double gib = argc > 1 ? atof(argv[1]) : 2.0;
    const int iters = argc > 2 ? atoi(argv[2]) : 20;
    const size_t n = ((size_t)(gib * (1ull << 30)) / sizeof(float)) & ~(size_t)1023;
    float * a = aligned_alloc(64, n * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;  // first touch on the executing threads' node

    double best = 1e30, sink = 0.0;
    for (int it = 0; it < iters; it++) {
        double t0 = omp_get_wtime();
        float s = 0.0f;
        #pragma omp parallel reduction(+:s)
        {
            const int nt = omp_get_num_threads(), id = omp_get_thread_num();
            const size_t chunk = (n / nt) & ~(size_t)127;
            const size_t lo = id * chunk, hi = id == nt - 1 ? n : lo + chunk;
            s += sum_range(a + lo, hi - lo);
        }
        double dt = omp_get_wtime() - t0;
        sink += s;
        if (it > 0 && dt < best) best = dt;  // skip the first (warm-up) pass
    }
    printf("threads=%d bytes=%zu best_s=%.6f read_GBps=%.2f (sink %.0f)\n",
           omp_get_max_threads(), n * sizeof(float), best, n * sizeof(float) / best / 1e9, sink);
    free(a);
    return 0;
}
