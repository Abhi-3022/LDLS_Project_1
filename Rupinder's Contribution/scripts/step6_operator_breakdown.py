"""Step 6: group VTune's per-function CPU time into operator types and list the top 3 functions.

Usage:  python scripts/step6_operator_breakdown.py results
Reads   results/vtune/prefill_4k.csv, decode_1k.csv, decode_8k.csv   (from step6_profile.ps1)
Writes  results/operator_breakdown.csv   (% of CPU time per operator type, per profile)
"""
import csv
import sys
from pathlib import Path

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
PROFILES = ["prefill_4k", "decode_1k", "decode_8k"]

# A function belongs to the first type whose keywords appear in its name.
TYPES = [
    ("Thread sync (waiting)", ["vcomp", "DelayExecution", "SwitchToThread", "WaitForSingleObject",
                               "atomic_", "ggml_barrier", "threadpool_chunk"]),
    ("Attention", ["flash_attn", "simd_gemm", "vec_dot_f16_f32", "vec_mad_f16_f32", "expf",
                   "vec_max_f32", "soft_max", "fp16_to_fp32", "fp32_to_fp16", "vec_scale_f32"]),
    ("Weight matmul Q4_K", ["q4_K"]),
    ("Weight matmul Q6_K", ["q6_K", "tiled"]),
    ("Weight matmul other", ["q8_K", "mul_mat", "hsum_float_8", "nearest_int", "fabsf", "repack"]),
]
OTHER = "Other (norm, RoPE, SiLU, add, runtime)"


def operator_type(function, module):
    if module == "VCOMP140.DLL":            # Microsoft OpenMP runtime = threads waiting
        return TYPES[0][0]
    for name, keywords in TYPES:
        if any(k in function for k in keywords):
            return name
    return OTHER


table = {}   # type -> {profile: percent}
for prof in PROFILES:
    with open(results / "vtune" / f"{prof}.csv", encoding="utf-8-sig") as f:
        functions = [(r["Function"], r["Module"], float(r["CPU Time"])) for r in csv.DictReader(f)]
    total = sum(t for _, _, t in functions)

    print(f"{prof}: total CPU time {total:.1f} s (all 6 threads)")
    for fn, mod, t in sorted(functions, key=lambda x: -x[2])[:3]:
        print(f"  top: {fn:45s} {t:7.1f} s  {100 * t / total:5.1f}%  [{operator_type(fn, mod)}]")

    per_type = {}
    for fn, mod, t in functions:
        k = operator_type(fn, mod)
        per_type[k] = per_type.get(k, 0) + t
    for k, t in per_type.items():
        table.setdefault(k, {})[prof] = round(100 * t / total, 1)

order = [name for name, _ in TYPES] + [OTHER]
with open(results / "operator_breakdown.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["operator_type"] + PROFILES)
    for k in order:
        w.writerow([k] + [table.get(k, {}).get(p, 0.0) for p in PROFILES])

print("\n% of CPU time by operator type:")
print(f"  {'':40s}" + "".join(f"{p:>12s}" for p in PROFILES))
for k in order:
    print(f"  {k:40s}" + "".join(f"{table.get(k, {}).get(p, 0.0):12.1f}" for p in PROFILES))
print(f"\n-> {results / 'operator_breakdown.csv'}")
