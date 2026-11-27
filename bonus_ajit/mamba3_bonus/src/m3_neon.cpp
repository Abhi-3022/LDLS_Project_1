#if defined(__aarch64__)
// neon build of the decode-step kernels (see m3_kernel.inc)
#define M3_ISA_NEON
#define M3_NS m3_neon
#include "m3_kernel.inc"
#endif
