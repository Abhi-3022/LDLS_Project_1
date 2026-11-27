#if defined(__x86_64__)
// avx512 build of the decode-step kernels (see m3_kernel.inc)
#define M3_ISA_AVX512
#define M3_NS m3_avx512
#include "m3_kernel.inc"
#endif
