# CPU Inference Project, Option 1: dense transformer baseline

Llama-3.2-1B and Llama-3.2-3B on llama.cpp, CPU only, on an Intel Core Ultra 7 155H laptop (Windows 11).
This repository has everything needed to rerun our measurements: build notes, benchmark scripts,
raw results, analysis and plotting code. The bonus (Mamba-3 decode step on CPU) is in `bonus_mamba3/`.

Team no.: 3

## Main results (6 threads, one per P-core)

| | 1B Q4_K_M | 1B Q8_0 | 3B Q4_K_M |
|---|---|---|---|
| Decode tok/s at 1k / 8k / 16k context | 41.8 / 16.6 / 6.9 | 29.4 / 14.4 / 6.4 | 15.8 / 5.2 / - |
| Prefill tok/s at 512 / 4k / largest prompt | 304.5 / 227.1 / 122.2 | 159.2 / 137.0 / 90.9 | 110.8 / 89.1 / 72.5 |
| Predicted decode tok/s at 1k (bandwidth roofline) | 62.5 | 38.7 | 24.5 |
| Measured / predicted at 1k / 8k / 16k | 67% / 34% / 18% | 76% / 44% / 23% | 65% / 29% / - |
| Perplexity (fixed sample) | 9.85 | 9.58 | 7.48 |

Memory bandwidth (STREAM Triad, 6 P-cores): 52.1 GB/s. Largest context: 16k for the 1B, 8k for the 3B.
Full numbers: `results/summary.csv` and `results/roofline.csv`; plots: `results/plots/`.

## Folder layout

```
README.md              this file: how to rerun everything
BUILD_NOTES.md         tools, llama.cpp build, model download, problems we hit
hardware.md            the laptop, which logical processors are P-cores, thread masks
requirements.txt       Python packages
data/                  sample.txt for the perplexity check
scripts/               one script per step (PowerShell for running, Python for analysis)
results/               raw CSVs, derived tables and plots (see results/README.md)
bonus_mamba3/          Mamba-3 SISO/MIMO decode step in C++ (own README)
```

## How to rerun everything

**0. Set up** (once): follow [BUILD_NOTES.md](BUILD_NOTES.md) to install the tools, build llama.cpp at
commit `a946eb541` and download the three models. Then:

```powershell
cd <this repository>
pip install -r requirements.txt
Set-ExecutionPolicy -Scope Process Bypass      # allow the .ps1 scripts in this window
```

**Before measuring on a different laptop**, edit `scripts/settings.ps1`: the folder paths and the
P-core masks (`$PCORE_MASK`, see [hardware.md](hardware.md) for how to find them). After step 1, put your
own bandwidth into `BANDWIDTH_MB_S` in `scripts/step7_analyze_results.py`.

**Before every measuring step:** charger connected, "Best performance" power mode, sleep off, all other
applications closed, nothing else running (the scripts log free memory before each run).

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Memory bandwidth | `.\scripts\step1_stream.ps1` | 2 min | `results\stream.txt` (use the mean Triad rate) |
| 2. One prompt per model | `.\scripts\step2_run_one_prompt.ps1` | 2 min | answer on screen (type `/exit` after each) |
| 3. Perplexity | `.\scripts\step3_perplexity.ps1` | 5 min | `results\perplexity.txt` |
| 4. Predicted decode speed | `python scripts\step4_predict_decode.py --bandwidth 52103 --models-dir $HOME\models --out results\prediction.csv` | seconds | `results\prediction.csv` |
| 5. Benchmark sweep | `.\scripts\step5_sweep.ps1` | several hours | `results\runs\*.csv`, `results\peak_memory.csv` |
| 6a. Profile (VTune) | `.\scripts\step6_profile.ps1` | 10 min | `results\vtune\*.csv` |
| 6b. Operator breakdown | `python scripts\step6_operator_breakdown.py results` | seconds | `results\operator_breakdown.csv` |
| 7. Analysis + roofline | `python scripts\step7_analyze_results.py results` | seconds | `results\summary.csv`, `results\roofline.csv` |
| 8. Own experiment A: core 0 in the mask | `.\scripts\step8_experiment_pinning.ps1` | 15 min | `results\exp_pinning\summary.csv` |
| 9. Own experiment B: turbo boost off | `.\scripts\step9_experiment_boost_off.ps1` | 5 min | `results\boost_off\capped_scaling.csv` |
| Plots | `python scripts\make_plots.py results` | seconds | `results\plots\*.png` |

Step 5 skips runs that already have a CSV, so it can be stopped and restarted; delete a run's CSV to redo it.
Step 9 switches turbo boost back on when it finishes.

**Only re-analysing our data** (no laptop runs needed): steps 6b, 7 and the plots work directly on the
files in `results/`, on any OS with Python:

```
python scripts/step6_operator_breakdown.py results
python scripts/step7_analyze_results.py results
python scripts/make_plots.py results
```

## The plots

| File | Shows |
|---|---|
| `1_decode_vs_context.png` | Decode speed vs context length (required) |
| `2_prefill_vs_prompt.png` | Prefill speed vs prompt length (required) |
| `3_thread_scaling.png` | Speed-up vs threads, prefill and decode (required) |
| `4_operator_breakdown.png` | % of CPU time per operator type, prefill 4k and decode 1k / 8k (required) |
| `5_predicted_vs_measured.png` | Predicted vs measured decode speed (required) |
| `6_experiment_pinning.png` | Own experiment A: thread mask with vs without core 0 |
| `7_boost_off_scaling.png` | Own experiment B: thread scaling with turbo boost on vs off |

## How the prediction works

Each decoded token reads every weight once plus the whole KV cache, so decode speed is limited by memory
bandwidth:

```
predicted tok/s = bandwidth / (weight bytes + context x KV bytes per token)
KV bytes per token = layers x KV heads x head dim x 2 (K and V) x 2 bytes (f16)
                   = 32,768 B (1B)   114,688 B (3B)
```

Example, 1B Q4_K_M at 1k context: 52,103 MB/s / (799.9 MB + 1,024 x 32,768 B) = 52,103 / 833.4 = 62.5 tok/s.

## Known limitations

- 1B Q8_0 prefill at 2 threads stayed noisy (up to 19% spread between repetitions).
- Only the 1B Q4_K_M was profiled with VTune, as it was not explicitly mentioned to get the prefill for all the models.
- The 3B at 8k context peaked at 5.45 GB of working set, over the 4.5 GB budget; llama.cpp's own
  allocation log for that run adds up to 3.29 GB (weights, repacked weights, KV cache, compute buffer),
  so the reported peak most likely counts the memory-mapped and repacked weights twice.
