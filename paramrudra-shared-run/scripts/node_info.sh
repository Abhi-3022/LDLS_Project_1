#!/usr/bin/env bash
# Hardware record for the report: CPU model, core types and count, cache sizes, SIMD support, RAM, NUMA, OS.
# Called by both jobs as ./node_info.sh <outdir>; writes one small text file per item. Reads only, needs no root.
set -uo pipefail
OUT=${1:?outdir}; mkdir -p "$OUT"
{ date -Is; hostname; uname -a; cat /etc/os-release 2>/dev/null; } > "$OUT/system.txt"
lscpu > "$OUT/lscpu.txt" 2>&1
lscpu -e > "$OUT/lscpu_e.txt" 2>&1 || true
lscpu -C > "$OUT/lscpu_cache.txt" 2>&1 || true
for i in 0 1 2 3; do d=/sys/devices/system/cpu/cpu0/cache/index$i; [[ -d $d ]] && echo "L? $(cat $d/level) $(cat $d/type) $(cat $d/size) shared_with=$(cat $d/shared_cpu_list)"; done > "$OUT/cache_cpu0.txt"
grep -o -w -E "sse4_2|avx|avx2|fma|avx512[a-z_0-9]*|amx[a-z_0-9]*|avx_vnni" /proc/cpuinfo | sort -u | tr '\n' ' ' > "$OUT/simd.txt"
cat /proc/meminfo | head -5 > "$OUT/meminfo.txt"
(command -v numactl >/dev/null && numactl -H) > "$OUT/numa.txt" 2>&1 || echo "numactl not available" > "$OUT/numa.txt"
(command -v dmidecode >/dev/null && dmidecode -t memory 2>/dev/null | grep -E "Speed|Size|Type:|Locator" | head -40) > "$OUT/dimms.txt" 2>&1 || true
(command -v lstopo-no-graphics >/dev/null && lstopo-no-graphics --no-io) > "$OUT/topology.txt" 2>&1 || true
{ echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)";
  echo "max_freq_khz: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo n/a)";
  echo "smt: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)";
  echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)";
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"; } > "$OUT/power_and_vm.txt"
env | grep -E "^SLURM_" | sort > "$OUT/slurm_env.txt" 2>/dev/null || true
(command -v scontrol >/dev/null && scontrol show node "${SLURMD_NODENAME:-$(hostname)}") > "$OUT/scontrol_node.txt" 2>&1 || true
echo "node record written to $OUT" >&2
