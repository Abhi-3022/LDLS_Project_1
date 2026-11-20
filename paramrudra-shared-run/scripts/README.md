# PARAM Rudra scripts

These scripts produce every PARAM Rudra number in the report. They cover the shared reference run, thread scaling over 4 cores, the Slurm limits, the hardware record, memory bandwidth, llama.cpp's own timings and the roofline check. There are two Slurm jobs, each under 3 minutes on 4 cores, because the QOS allows one user 4 CPUs for 5 minutes.

## Run it

Copy this folder to Rudra, then run the following on the login node. Compute nodes have no internet and no C headers.

```bash
./00_setup_login.sh          # once, ~10 min: llama.cpp 827466b, the 1B Q4_K_M model (sha256 checked), build, tools
./50_limits_login.sh         # what Slurm allows you -> results/limits_<date>/
./submit.sh reference report # the two jobs; the QOS lets both run at once
squeue -u $USER              # wait until both are gone
cat results/rudra_<jobid>/REPORT.md results/ref_<jobid>/REPORT.md
./collect.sh                 # -> rudra_results.tar.gz to copy home
```

## What each file does

| File | Where it runs | What it does | Output |
| --- | --- | --- | --- |
| `config.sh` | sourced by all | Every setting in one place: partition, compilers, paths, KV type, repeats, thread counts. Any value can be overridden from the environment. | |
| `lib.sh` | sourced by all | Loads gcc 12, Spack cmake and numactl. Works out which physical cores the job got. `place` pins threads one per core with memory on that socket. `timev` records peak RSS, `logged` records every command, `write_manifest` records the software. | |
| `00_setup_login.sh` | login node | Steps `fetch`, `env`, `build`, `tools` and `doctor`, run separately or all together. | `third_party/`, `models/`, `.env/`, `tools-bin/` |
| `50_limits_login.sh` | login node | Read-only queries: `sacctmgr`, `sinfo`, `scontrol` and `sacct` limits, plus which profilers exist. | `results/limits_<date>/` |
| `submit.sh` | login node | Submits `reference` and/or `report`; `--dry-run` prints the commands. | job ids |
| `10_reference.sbatch` | compute job | The shared reference run (`llama-bench -p 512 -n 128 -r 3`) at 4, 3, 2 and 1 threads, plus the node record. | `results/ref_<jobid>/` |
| `60_report.sbatch` | compute job | STREAM and the read-only probe at 1 and 4 threads, the reference run, llama-completion timings and perf (skipped if missing), each 3 times. | `results/rudra_<jobid>/` |
| `node_info.sh` | inside jobs | Hardware record: lscpu, caches, SIMD flags, memory, NUMA, governor, OS. | `<result>/node/` |
| `report_reference.py` | inside job / home | `REPORT.md` and `reference_summary.csv` for a `ref_` folder: headline and thread scaling. | |
| `report_rudra.py` | inside job / home | `REPORT.md` for a `rudra_` folder: limits, hardware, bandwidth, reference, timings, profile and roofline. | |
| `common.py` | imported | File readers, mean ± std, hardware tables, and the model's bytes per token. | |
| `readbw.c` | built by `tools` | Read-only bandwidth probe (AVX-512 loads), the bandwidth that decode needs. | |
| `prompt_512.txt` | report job | The first 2,140 characters of wikitext-2 test, which is exactly 512 tokens. | |
| `models.sha256` | setup, manifest | The expected checksum of the GGUF file. | |

## How every run is set up

- **Model and test:** Llama-3.2-1B-Instruct Q4_K_M (bartowski GGUF), llama.cpp commit `827466b`, CPU backend, `-DGGML_NATIVE=ON` (which resolves to `cascadelake`), gcc 12.3.
- **KV cache:** Q8_0 with flash attention (`-ctk q8_0 -ctv q8_0 -fa 1`), because of the project rule "No F16 anywhere".
- **Placement:** one thread per physical core on socket 0, using `taskset` with `OMP_PROC_BIND=close OMP_PLACES=cores`, and memory bound to socket 0 with `numactl --membind`. 4 cores never leave one socket.
- **Repeats:** every measurement runs 3 times, and the reports give mean ± sample std with min–max.
- **Peak memory:** compute nodes have no `/usr/bin/time`, so `timev` falls back to Python's `getrusage`.
- **No pollution:** nothing goes outside this folder: no `pip --user`, `~/.local`, `~/.cache`, `~/.spack` or `~/.bashrc`. `module load` and `spack load` only change the running script's PATH. To undo everything, delete the folder.

## Known gaps

- **perf is not installed** on Rudra's login or compute nodes, so `60_report.sbatch` writes `perf/NOT_RUN.txt` instead of a profile. The login node allows profiling your own processes (`perf_event_paranoid=0`) and offers VTune (`module load compiler/oneapi-2024/vtune/2024.1`). Set `PERF_BIN` to use a perf binary from elsewhere.
- **4 cores and 5 minutes only:** whole-socket and whole-node runs don't fit the QOS, and neither do long-context prefill or decode (6–8 min on 4 cores). Only the 1B Q4_K_M reference model is run here.
