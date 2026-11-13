#!/usr/bin/env bash
# Hardware record: CPU model, cores, caches, SIMD, RAM, NUMA, frequency policy, load.
# Bandwidth comes from 02_stream.sh. Output: hw/hardware.txt and hw/hardware.json
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$PROJ/hw"
OUT="$PROJ/hw/hardware.txt"

{
    echo "# Hardware record ($(date -Iseconds), host $(hostname))"
    echo
    echo "## lscpu"
    lscpu
    echo
    echo "## caches"
    lscpu --caches 2>/dev/null || true
    echo
    echo "## SIMD flags present"
    grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse4_2|avx|avx2|fma|f16c|avx512[a-z_0-9]*|amx[a-z_0-9]*|avx_vnni)$' | sort | paste -sd' ' -
    echo
    echo "## NUMA"
    numactl -H
    echo
    echo "## memory"
    head -5 /proc/meminfo
    echo
    echo "## frequency policy"
    echo "driver:   $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "max kHz:  $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)"
    echo "boost:    $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo n/a)"
    echo "THP:      $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo
    echo "## kernel / toolchain"
    uname -a
    gcc --version | head -1
    cmake --version | head -1
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)  (3 = perf blocked for users)"
    echo
    echo "## load at record time"
    uptime
    echo
    echo "## placement used for all runs"
    echo "NUMA node $NODE, cores: $CORES"
} > "$OUT"

"$PY" - "$OUT" <<'EOF'
import json, re, subprocess, sys
txt = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
kv = dict(re.findall(r"^([^:]+):\s+(.*)$", txt, re.M))
flags = open("/proc/cpuinfo").read().split("flags")[1].split("\n")[0].split()
simd = sorted(f for f in flags if re.match(r"^(avx2|fma|avx512\w*|amx\w*|avx_vnni)$", f))
mem_kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
cache = {}
for i in range(4):
    base = f"/sys/devices/system/cpu/cpu0/cache/index{i}/"
    lvl, typ, size = (open(base + f).read().strip() for f in ("level", "type", "size"))
    cache[f"L{lvl}{'d' if typ == 'Data' else 'i' if typ == 'Instruction' else ''}"] = size
d = {
    "cpu_model": kv.get("Model name"),
    "sockets": int(kv.get("Socket(s)", 0)),
    "cores_per_socket": int(kv.get("Core(s) per socket", 0)),
    "threads_per_core": int(kv.get("Thread(s) per core", 0)),
    "numa_nodes": int(kv.get("NUMA node(s)", 0)),
    "cache_per_instance": cache,  # L1/L2 per core, L3 per socket
    "max_mhz": kv.get("CPU max MHz"),
    "simd": simd,
    "ram_gb": round(mem_kb / 2**20, 1),
    "core_type": "homogeneous (no P/E split); SMT siblings excluded",
}
json.dump(d, open(sys.argv[1].replace(".txt", ".json"), "w"), indent=2)
print(json.dumps(d, indent=2))
EOF
echo "wrote $OUT"
