# PARAM Rudra deliverables (Project 1, Option 1)

| Item | What it is |
| --- | --- |
| [`REPORT.md`](REPORT.md) | The final PARAM Rudra report: allocation limits, hardware record, memory bandwidth, the shared reference run, thread scaling against the workstation, llama.cpp timings, the roofline check and its gap, profiling status, limitations |
| [`scripts/`](scripts/README.md) | The cleaned Slurm scripts that produce every number: setup, limits, two 4-core jobs, report generators |
| [`results/`](results/README.md) | Raw data and per-job reports of Rudra jobs 187667, 187669, 187670 and 187685, plus the login-node limits record |

Headline: Llama-3.2-1B-Instruct Q4_K_M on 4 cores of a Rudra node, pp512 209.68 ± 0.41 tok/s, tg128 42.13 ± 0.02 tok/s, peak RSS 1.32 GB (job 187670).
