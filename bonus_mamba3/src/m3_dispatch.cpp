#include "m3_kernels.h"

int m3_impls(M3Impl * out, int max) {
    int n = 0;
    auto add = [&](M3Impl v) { if (n < max) out[n++] = v; };
#if defined(__x86_64__)
    add({"scalar", m3_scalar::mamba3_siso, m3_scalar::mamba3_mimo, m3_scalar::mamba2, true});
    add({"avx2",   m3_avx2::mamba3_siso,   m3_avx2::mamba3_mimo,   m3_avx2::mamba2,
         __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")});
    add({"avx512", m3_avx512::mamba3_siso, m3_avx512::mamba3_mimo, m3_avx512::mamba2,
         (bool) __builtin_cpu_supports("avx512f")});
#elif defined(__aarch64__)
    add({"scalar", m3_scalar::mamba3_siso, m3_scalar::mamba3_mimo, m3_scalar::mamba2, true});
    add({"neon",   m3_neon::mamba3_siso,   m3_neon::mamba3_mimo,   m3_neon::mamba2,   true});
#else
    add({"scalar", m3_scalar::mamba3_siso, m3_scalar::mamba3_mimo, m3_scalar::mamba2, true});
#endif
    return n;
}
