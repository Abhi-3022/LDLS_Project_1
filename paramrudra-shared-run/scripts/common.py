"""Shared by report_reference.py and report_rudra.py (standard library only: runs on Rudra or at home).

Readers for the files the jobs write, the statistics every table uses, and the hardware/software tables.
"""
import json, os, re

# Bytes the 1B Q4_K_M model reads per decoded token. Every weight is streamed once per token: 799,862,912 B, the sum
# of the stored tensor sizes in the GGUF file (computed on the workstation with p1-dense/scripts/gguf_bytes.py;
# the tied embedding counts in full because it is also the output matrix). The KV cache adds, per token of context,
# 2 (K and V) x 16 layers x 8 KV heads x 64 dims x 2 B = 32,768 B with an F16 cache.
W_1B_Q4 = 799_862_912
KV_1B_F16 = 32_768
KV_REL = {"f16": 1.0, "q8_0": 34 / 64, "q4_0": 18 / 64}   # Q8_0 stores 32 values in 34 bytes (F16: 64 bytes)
THEORY_SOCKET = 140.8   # GB/s per socket: 6 channels x DDR4-2933 x 8 B (Rudra manual)


def kv(path):
    """'key: value' lines -> dict (manifest.txt, lscpu.txt, limits_summary.txt); first value of a key wins."""
    d = {}
    try:
        for line in open(path):
            if ":" in line:
                k, v = line.split(":", 1)
                d.setdefault(k.strip(), v.strip())
    except OSError:
        pass
    return d


def samples(jsonl_path):
    """llama-bench -oe jsonl -> {"pp512": [tok/s of each repeat], "tg128": [...]}"""
    out = {}
    try:
        for line in open(jsonl_path):
            r = json.loads(line)
            test = f"tg{r['n_gen']}" if int(r["n_gen"]) else f"pp{r['n_prompt']}"
            out[test] = [round(x, 2) for x in r.get("samples_ts", [])]
    except (OSError, ValueError):
        pass
    return out


def peak_rss_gb(time_path):
    """peak resident memory (GB) from a timev / GNU time -v file, '' if missing"""
    try:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(time_path).read())
        return round(int(m.group(1)) / 1024 ** 2, 3) if m else ""
    except OSError:
        return ""


def stats(xs):
    """(mean, sample std, min, max, n), or None for no values"""
    xs = [float(x) for x in xs]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd, min(xs), max(xs), len(xs)


def fmt(s, nd=2):
    """'mean ± std (min–max, n=N)', the format of every measured value in the reports"""
    if not s:
        return "?"
    m, sd, lo, hi, n = s
    return f"{m:.{nd}f} ± {sd:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f}, n={n})"


def record(d):
    """Hardware and Software tables (markdown lines) from <d>/manifest.txt and <d>/node/"""
    m = kv(os.path.join(d, "manifest.txt"))
    lc = kv(os.path.join(d, "node", "lscpu.txt"))
    pw = kv(os.path.join(d, "node", "power_and_vm.txt"))
    mem = kv(os.path.join(d, "node", "meminfo.txt")).get("MemTotal", "?")
    try:
        osr = re.search(r'PRETTY_NAME="([^"]+)"', open(os.path.join(d, "node", "system.txt")).read()).group(1)
    except (OSError, AttributeError):
        osr = "?"
    g = lambda k: m.get(k, "?")
    shared = "" if g("physical_cores_used") in ("48", "?") else "; shared node: other jobs may use the same memory bandwidth"
    L = ["## Hardware\n", "| | |\n| --- | --- |"]
    for k, v in [("Node", f"{g('node')} (partition `{g('partition')}`, job {g('job_id')})"),
                 ("Allocation", f"{g('physical_cores_used')} cores (CPUs {g('allowed_cpus')}), {g('slurm_mem_per_node_MB')} MB{shared}"),
                 ("CPU", lc.get("Model name", "?")),
                 ("Cores", f"{lc.get('Socket(s)', '?')} sockets × {lc.get('Core(s) per socket', '?')} cores, one core type, "
                           f"{lc.get('Thread(s) per core', '?')} thread(s) per core"),
                 ("Max clock", f"{lc.get('CPU max MHz', '?')} MHz"),
                 ("Caches", f"L1d {lc.get('L1d cache', '?')}, L2 {lc.get('L2 cache', '?')}, L3 {lc.get('L3 cache', '?')}"),
                 ("NUMA", f"{lc.get('NUMA node(s)', '?')} nodes (one per socket)"),
                 ("Memory", (f"{int(mem.split()[0]) / 1024 ** 2:.0f} GB" if mem[:1].isdigit() else mem)
                  + " DDR4-2933, 6 channels per socket (Rudra manual)"),
                 ("SIMD", g("simd")),
                 ("Governor / SMT / THP", f"{pw.get('governor', '?')} / {pw.get('smt', '?')} / {pw.get('thp', '?')}"),
                 ("OS", osr)]:
        L.append(f"| {k} | {v} |")
    sha_ok = "matches" if g("model_sha256") == g("model_sha256_expected") else "DOES NOT MATCH"
    L += ["\n## Software\n", "| | |\n| --- | --- |"]
    for k, v in [("llama.cpp", f"commit `{g('llama_cpp_commit')}` ({g('llama_cpp_local_changes')} locally changed)"),
                 ("Build", f"CMake {g('cmake_CMAKE_BUILD_TYPE')}, GGML_NATIVE={g('cmake_GGML_NATIVE')} "
                           f"(-march=native = `{g('march_native_resolves_to')}`), OpenMP={g('cmake_GGML_OPENMP')}, CPU backend only"),
                 ("Compiler", g("compiler_cxx")),
                 ("Modules / Spack", f"{g('modules_loaded')} / {g('spack_loaded')}"),
                 ("Model", f"`{os.path.basename(g('model'))}`, {g('model_bytes')} bytes, sha256 `{g('model_sha256')[:16]}…` ({sha_ok} the expected checksum)"),
                 ("Run time", f"{g('date_start')} to {g('date_end')} ({g('job_runtime_s')} s)")]:
        L.append(f"| {k} | {v} |")
    return L
