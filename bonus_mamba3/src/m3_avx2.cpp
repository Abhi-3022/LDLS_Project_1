#if defined(__x86_64__)
// avx2 build of the decode-step kernels (see m3_kernel.inc)
#define M3_ISA_AVX2
#define M3_NS m3_avx2
#include "m3_kernel.inc"
#endif
