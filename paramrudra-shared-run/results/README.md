# PARAM Rudra results

Every result folder from the Rudra jobs of 20 Nov 2026. All runs used 4 CPUs on a shared `cpu` node under QOS `aicte_qip`, 1B Q4_K_M, a Q8_0 KV cache and llama.cpp `827466b`. The final numbers are in [`../REPORT.md`](../REPORT.md).

| Folder | Job | Node | What it holds | Used in the report for |
| --- | --- | --- | --- | --- |
| [`rudra_187685/`](rudra_187685/REPORT.md) | 187685, 75 s | rdcn25 | Limits, hardware record, STREAM and read probe at 1 and 4 threads, reference run at 4 threads, llama-completion timings; perf not run (not installed) | Limits, hardware, bandwidth, timings, roofline |
| [`ref_187670/`](ref_187670/REPORT.md) | 187670, 22 s | rdcn20 | The reference run at 4 threads, with the 1–3-thread runs of 187669 copied in (marked in `reference/*.source`) | Headline reference number, thread scaling |
| [`ref_187669/`](ref_187669/REPORT.md) | 187669, 128 s | rdcn20 | The reference run at 1, 2 and 3 threads | Thread scaling |
| [`ref_187667/`](ref_187667/REPORT.md) | 187667, 22 s | rdcn20 | First 4-thread reference run. Peak RSS was not recorded yet, because the node has no `/usr/bin/time` | Repeatability check |
| [`limits_20261120_181502/`](limits_20261120_181502/limits.txt) | (login02) | login02 | `sacctmgr`, `sinfo`, `scontrol`, `sacct` output, and the profilers on offer | Limits (also copied into `rudra_187685/limits/`) |

These folders were made by the earlier version of the scripts, whose settings are in each folder's `config_used.sh`. The cleaned scripts in [`../scripts/`](../scripts/) run the same commands with the same settings. Thread scaling is now one job (`10_reference.sbatch`, 1–4 threads) instead of two. Each folder's `REPORT.md` was regenerated with the cleaned `report_reference.py` / `report_rudra.py`; the measured data is unchanged.

Inside every job folder:

- `REPORT.md`
- `manifest.txt`: software, build, model sha256, allocation
- `node/`: hardware record
- `commands.txt`: every command with its exit code and time
- `reference/`: llama-bench CSV, per-repeat JSONL, and peak RSS in `.time`
- `bw/`, `timings/`, `perf/` and `limits/`: in `rudra_` folders only
- the Slurm `.out` and `.err` logs

Not included:

- The STREAM and readbw binaries.
- `rudra_local_20261120_181533`, an accidental partial run on the login node whose numbers are not a compute node's.
- `limits_20261120_181449`, an earlier copy of the same limits.
