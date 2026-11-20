// Read-only memory bandwidth probe: the bandwidth the decode roofline needs.
// Decoding a token streams every weight once and writes almost nothing, while STREAM always mixes reads with
// writes. Each thread sums its own slice of a large array with AVX-512 loads; the array is first touched by the
// thread that later reads it, so its pages sit on that thread's NUMA node.
// Prints the best and the mean bandwidth of passes 2-10 (pass 1 is a warm-up).
// Build: gcc -O3 -march=native -fopenmp readbw.c -o readbw      (on Rudra: ./00_setup_login.sh tools)
// Run:   OMP_NUM_THREADS=4 ./readbw [GiB]                       (2 GiB in the report job)
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now(void) { return omp_get_wtime(); }

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 4.0;          // default 4 GiB; 2 GiB is still 57x Rudra's 35.75 MB L3
    size_t n = (size_t)(gib * (1ull << 30)) / sizeof(float);
    n &= ~(size_t)1023;
    float *a = aligned_alloc(64, n * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = 1.0f;          // first touch by the reading thread

    double best = 0, sum_bw = 0; int reps = 10; volatile float sink = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        float total = 0;
        #pragma omp parallel for schedule(static) reduction(+:total)
        for (size_t i = 0; i < n; i += 64) {
            __m512 s0 = _mm512_load_ps(a + i),      s1 = _mm512_load_ps(a + i + 16);
            __m512 s2 = _mm512_load_ps(a + i + 32), s3 = _mm512_load_ps(a + i + 48);
            total += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
        }
        double dt = now() - t0, bw = n * sizeof(float) / dt / 1e9;
        sink += total; if (r > 0) { sum_bw += bw; if (bw > best) best = bw; }
    }
    printf("threads=%d read_GBps_best=%.1f read_GBps_mean=%.1f\n", omp_get_max_threads(), best, sum_bw / (reps - 1));
    return 0;
}
