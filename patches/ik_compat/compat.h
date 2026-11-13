// Force-included when building ik_llama.cpp with GCC 9 (see scripts/01_build.sh).
#pragma once
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 10
#include <immintrin.h>
// GCC 10 added this intrinsic; zero-extend a 256-bit vector to 512 bits.
static inline __m512 _mm512_zextps256_ps512(__m256 a) {
    return _mm512_insertf32x8(_mm512_setzero_ps(), a, 0);
}
#endif
#ifdef __cplusplus
#include <atomic>  // newer libstdc++ pulls this in transitively; ggml-backend.cpp relies on it
#endif
