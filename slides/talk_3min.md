# 3-minute talk script (slides/talk.pptx, 12 slides)

About 430 spoken words at ~145 words per minute. Times are cumulative.

**1. Title (0:00–0:10)**
We compared small mixture-of-experts models, Granite-3.1, with dense Llama-3.2 on a CPU using llama.cpp, and for the bonus we wrote a Mamba-3 decode step.

**2. Four questions, one socket (0:10–0:25)**
We asked four things. Does MoE decode follow active or total parameters? What does routing cost? How close are we to the bandwidth limit? And when does attention take over? We used one Xeon socket: 55 pinned cores and 136 GB/s, with every run under 4.5 GB and three repeats each.

**3. Predict first (0:25–0:40)**
Before measuring, we counted the bytes read per token. Granite-1B reads 0.27 GB per token and Llama-1B reads 0.80, even though both files are 0.8 GB. So if decode follows active parameters, Granite should be about three times faster.

**4. Thread scaling (0:40–0:55)**
Prefill is compute-bound: it scales 22× up to 55 cores. Decode flattens once bandwidth saturates at around 16 to 24 threads, and for the MoE models more cores actually hurt. One busy core in the pinned set slowed everything tenfold.

**5. Context length (0:55–1:10)**
Long context removes the MoE advantage. From 1k to 16k, Granite-1B drops from 183 to 52 tokens per second. Its KV cache is larger, so at 16k it is slower than Llama. Attention passes half of decode time between 4k and 8k.

**6. Why we miss the roofline (1:10–1:25)**
There are four reasons. On few cores, Q4 dequantization is the limit. Every graph node ends in a barrier. Single-query attention reads the KV cache inefficiently. And small expert matmuls give each thread too little work. Dense models reach 78% of the roofline; MoE reaches 46%.

**7. Profile and routing cost (1:25–1:40)**
Our own per-operator timer shows that routing and expert combine take 15% of Granite decode, 0.8 ms per token. Almost none of that is arithmetic; it is waiting at barriers. The MoE graph has 823 nodes per token against 277 for Llama.

**8. Active or total? (1:40–1:55)**
So decode follows active parameters, but only halfway. Granite-1B is 1.66× faster than Llama-1B where the bytes predict 2.9×, and 2.4× against 3.6× at 3B. Prefill gains nothing, because 512 tokens touch every expert.

**9. Our experiment: experts per token (1:55–2:10)**
We changed only the number of routed experts, k. Decode time grows linearly, by 243 microseconds per expert, so time follows active parameters. With all experts on, Granite takes 1.45× longer than Llama at the same bytes: that is the cost of the MoE machinery. Perplexity is best at the trained k of 8.

**10. ik_llama.cpp and the reference test (2:10–2:25)**
ik_llama.cpp fuses the MoE ops and gains 39% on Granite-1B decode, while dense decode is unchanged. That confirms synchronisation is the gap. Our shared reference result is 1873 tokens per second prefill and 124 decode.

**11. Bonus: Mamba-3 (2:25–2:45)**
We wrote Mamba-3 SISO and MIMO decode kernels in C++ for AVX2, AVX-512 and NEON. All 48 configurations match the official reference over 1,000 steps. MIMO-4 does 3.6× the FLOPs, yet costs only 1.31× on one core and 1.03× at 32 cores. So on a full socket it is nearly free, but on a laptop-class AVX2 core it costs 1.58×.

**12. What we learned (2:45–3:00)**
On a CPU, decode is memory-bound only with enough cores. With few cores dequantization limits it, and with many the barriers do. MoE cuts bytes threefold but triples the graph, so it gets half the promised speed-up. Fused kernels win much of that back. And always pin to quiet cores.
