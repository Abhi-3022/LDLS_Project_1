# Mamba-3 decode-step kernels (bonus). `make` builds verify, bench and ggml_m3.
# Each ISA gets its own object so one binary can dispatch at runtime.
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra -Wno-unused-function
LLAMA    ?= ../third_party/llama.cpp
GGML_INC  = -I$(LLAMA)/ggml/include
GGML_LIB  = -L$(LLAMA)/build-cpu/bin -Wl,-rpath,$(abspath $(LLAMA)/build-cpu/bin) -lggml -lggml-base -lggml-cpu
B = build
ARCH := $(shell uname -m)

ifeq ($(ARCH),x86_64)
KOBJ = $(B)/m3_scalar.o $(B)/m3_avx2.o $(B)/m3_avx512.o
else
KOBJ = $(B)/m3_scalar.o $(B)/m3_neon.o
endif
KOBJ += $(B)/m3_dispatch.o

all: $(B)/verify $(B)/bench $(B)/ggml_m3

$(B):
	mkdir -p $(B)

# plain C++ baseline: no SIMD ISA flags and no auto-vectorisation
$(B)/m3_scalar.o: src/m3_scalar.cpp src/m3_kernel.inc src/m3_kernels.h | $(B)
	$(CXX) $(CXXFLAGS) -fno-tree-vectorize -fno-tree-slp-vectorize -c $< -o $@
$(B)/m3_avx2.o: src/m3_avx2.cpp src/m3_kernel.inc src/m3_kernels.h | $(B)
	$(CXX) $(CXXFLAGS) -mavx2 -mfma -c $< -o $@
$(B)/m3_avx512.o: src/m3_avx512.cpp src/m3_kernel.inc src/m3_kernels.h | $(B)
	$(CXX) $(CXXFLAGS) -mavx512f -mavx512dq -mavx2 -mfma -c $< -o $@
$(B)/m3_neon.o: src/m3_neon.cpp src/m3_kernel.inc src/m3_kernels.h | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(B)/m3_dispatch.o: src/m3_dispatch.cpp src/m3_kernels.h | $(B)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(B)/verify: src/verify.cpp src/common.h $(KOBJ)
	$(CXX) $(CXXFLAGS) $< $(KOBJ) -o $@
$(B)/bench: src/bench.cpp src/common.h $(KOBJ)
	$(CXX) $(CXXFLAGS) $< $(KOBJ) -o $@
$(B)/ggml_m3: src/ggml_m3.cpp src/common.h $(KOBJ)
	$(CXX) $(CXXFLAGS) $(GGML_INC) $< $(KOBJ) $(GGML_LIB) -o $@

clean:
	rm -rf $(B)

.PHONY: all clean
