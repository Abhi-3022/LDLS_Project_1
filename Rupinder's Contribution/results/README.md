# Results

All files here come from the laptop in [../hardware.md](../hardware.md). Raw files are exactly as the
scripts wrote them; derived files can be rebuilt from the raw files with the Python scripts.

## Raw (measured)

| File / folder | Written by | Content |
|---|---|---|
| `runs/*.csv` (102 files) | `step5_sweep.ps1` | One `llama-bench` CSV per run: 3 models x 1-6 threads x 3 prefill lengths / 3 decode contexts (3B: 2 of each). `avg_ts` and `stddev_ts` are the mean and standard deviation of 3 repetitions |
| `peak_memory.csv` | `step5_sweep.ps1` | Free memory before each run and peak working set during it (GB) |
| `prediction.csv` | `step4_predict_decode.py` | Predicted decode speed per model and context, all values read from the GGUF files |
| `vtune/*.csv` | `step6_profile.ps1` | VTune hotspots, CPU time per function: `prefill_4k`, `decode_1k`, `decode_8k` (1B Q4_K_M, 6 threads) |
| `exp_pinning/` | `step8_experiment_pinning.ps1` | Own experiment A: `.json` = llama-bench output (5 samples), `.freq.txt` = clock of each P-core once a second (% of base clock), `summary.csv` = one row per case |
| `boost_off/capped_scaling.csv` | `step9_experiment_boost_off.ps1` | Own experiment B: 1B Q4_K_M speed at 1-6 threads with turbo boost off |

STREAM and perplexity were run before the result files were standardised, and only their console
output was kept. Their values (also in the report) are:

| Measurement | Value |
|---|---|
| STREAM Triad, 6 P-cores, 3 runs | 51.7, 52.3, 52.3 GB/s → mean **52,103 MB/s** (used in all predictions) |
| Perplexity, 512-token chunks, `data/sample.txt` | 1B Q4_K_M **9.85**, 1B Q8_0 **9.58**, 3B Q4_K_M **7.48** |

Running `step1_stream.ps1` and `step3_perplexity.ps1` writes `stream.txt` and `perplexity.txt` here.

## Derived (rebuilt by the Python scripts)

| File | Written by | Content |
|---|---|---|
| `summary.csv` | `step7_analyze_results.py` | One row per run: model, kind, threads, prompt, context, tok/s, std. dev., spread, memory |
| `roofline.csv` | `step7_analyze_results.py` | Predicted vs measured decode speed at 6 threads, % of prediction, effective GB/s |
| `operator_breakdown.csv` | `step6_operator_breakdown.py` | % of CPU time per operator type, per profile |
| `plots/1-7_*.png` | `make_plots.py` | The five required plots plus the two own-experiment plots |

`bonus_mamba3/results/` holds the Mamba-3 bonus benchmark CSVs and plot.
