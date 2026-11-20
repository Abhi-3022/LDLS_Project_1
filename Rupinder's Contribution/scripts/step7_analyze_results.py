"""Steps 5 and 7: collect the sweep results and compare predicted with measured decode speed.

Usage:  python scripts/step7_analyze_results.py results

Reads   results/runs/*.csv        one llama-bench CSV per run (from step5_sweep.ps1)
        results/peak_memory.csv   free memory before and peak memory of every run
Writes  results/summary.csv       one row per run: speed, spread, memory
        results/roofline.csv      predicted vs measured decode speed (6 threads)
"""
import csv
import sys
from pathlib import Path

# ---- inputs measured earlier -------------------------------------------------
BANDWIDTH_MB_S = 52103          # STREAM Triad mean (results/stream.txt), 1 MB = 1,000,000 bytes
WEIGHT_BYTES = {                # tensor bytes from step4_predict_decode.py (= llama-bench "size")
    "1B Q4_K_M": 799_862_912,
    "1B Q8_0": 1_313_251_456,
    "3B Q4_K_M": 2_011_539_712,
}
KV_BYTES_PER_TOKEN = {          # layers x KV heads x head dim x 2 (K and V) x 2 bytes (F16)
    "1B Q4_K_M": 16 * 8 * 64 * 2 * 2,      # 32,768
    "1B Q8_0": 16 * 8 * 64 * 2 * 2,        # 32,768
    "3B Q4_K_M": 28 * 8 * 128 * 2 * 2,     # 114,688
}
MEMORY_BUDGET_GB = 4.5
SPREAD_LIMIT_PCT = 3.0

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def short_name(model_path):
    """C:\\...\\Llama-3.2-1B-Instruct-Q4_K_M.gguf -> 1B Q4_K_M"""
    file = model_path.replace("\\", "/").split("/")[-1]
    return file.replace("Llama-3.2-", "").replace("-Instruct-", " ").replace(".gguf", "")


# ---- 1. memory log: keep the last line of each run (a rerun replaces the old line) --------
memory = {}
with open(results / "peak_memory.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        memory[row["run"]] = row

# ---- 2. one row per run -----------------------------------------------------------------
rows = []
for path in sorted((results / "runs").glob("*.csv")):
    with open(path, encoding="utf-8-sig") as f:
        bench = next(csv.DictReader(f))          # llama-bench writes one data row per test
    run = path.stem
    mem = memory[run]
    speed = float(bench["avg_ts"])               # tokens/s, mean of the 3 repetitions
    std = float(bench["stddev_ts"])
    rows.append({
        "run": run,
        "kind": run.split("_")[0],               # prefill or decode
        "model": short_name(bench["model_filename"]),
        "threads": int(bench["n_threads"]),
        "prompt": int(bench["n_prompt"]),
        "context": int(bench["n_depth"]),
        "tok_s": round(speed, 2),
        "stddev": round(std, 2),
        "spread_pct": round(100 * std / speed, 1),
        "free_gb_before": round(float(mem["free_gb_before"]) * 2**30 / 1e9, 2),   # GiB -> GB
        "peak_gb": round(float(mem["peak_rss_gb"]) * 2**30 / 1e9, 2),            # GiB -> GB
    })

with open(results / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} runs -> {results / 'summary.csv'}\n")

# ---- 3. checks ----------------------------------------------------------------------------
print(f"Runs whose 3 repetitions differ by more than {SPREAD_LIMIT_PCT}% (standard deviation / mean):")
for r in sorted(rows, key=lambda r: -r["spread_pct"]):
    if r["spread_pct"] > SPREAD_LIMIT_PCT:
        print(f"  {r['run']:50s} {r['tok_s']:8.2f} tok/s  spread {r['spread_pct']:5.1f}%")

print(f"\nRuns over the {MEMORY_BUDGET_GB} GB memory budget:")
for r in rows:
    if r["peak_gb"] > MEMORY_BUDGET_GB:
        print(f"  {r['run']:50s} peak {r['peak_gb']:.2f} GB  (free before: {r['free_gb_before']:.2f} GB)")
print(f"Lowest free memory before a run: {min(r['free_gb_before'] for r in rows):.2f} GB")

# ---- 4. predicted vs measured decode speed (6 threads) ------------------------------------
print("\nPredicted vs measured decode speed, 6 threads")
print("  predicted = bandwidth / (weight bytes + context x KV bytes per token)")
roofline = []
decode_6 = [r for r in rows if r["kind"] == "decode" and r["threads"] == 6]
for r in sorted(decode_6, key=lambda r: (r["model"], r["context"])):
    m, ctx = r["model"], r["context"]
    bytes_per_token = WEIGHT_BYTES[m] + ctx * KV_BYTES_PER_TOKEN[m]
    predicted = BANDWIDTH_MB_S * 1e6 / bytes_per_token
    measured = r["tok_s"]
    percent = 100 * measured / predicted
    effective_gb_s = measured * bytes_per_token / 1e9     # bytes the measured speed implies per second
    print(f"  {m:10s} ctx {ctx:6d}: {bytes_per_token / 1e6:7.1f} MB/token -> predicted {predicted:5.1f}, "
          f"measured {measured:5.1f} tok/s = {percent:3.0f}%  (effective {effective_gb_s:4.1f} GB/s)")
    roofline.append({"model": m, "context": ctx, "bytes_per_token_mb": round(bytes_per_token / 1e6, 1),
                     "predicted_tok_s": round(predicted, 1), "measured_tok_s": measured,
                     "measured_pct": round(percent, 1), "effective_gb_s": round(effective_gb_s, 1)})

with open(results / "roofline.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(roofline[0]))
    w.writeheader()
    w.writerows(roofline)
print(f"\n-> {results / 'roofline.csv'}")
