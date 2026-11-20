#!/usr/bin/env bash
# Run on the LOGIN node, before submitting the report job:   ./50_limits_login.sh
# Records what Slurm allows this user and what it actually gave earlier jobs:
#   association (account, QOS)  -> sacctmgr show assoc
#   QOS limits (CPUs, wall time, jobs per user)  -> sacctmgr show qos
#   partition and node limits  -> sinfo, scontrol show partition
#   jobs of the last 2 days and the most any of them got  -> sacct
#   profiling tools on offer (perf, perf_event_paranoid, VTune modules)
# Only queries: nothing is submitted or changed. Output: results/limits_<date>/limits.txt (every command and its
# output) and limits_summary.txt (key: value lines that 60_report.sbatch copies and report_rudra.py prints).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
# shellcheck disable=SC1091
source ./config.sh
OUT=$RESULTS_ROOT/limits_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
F=$OUT/limits.txt
sec() { printf '\n===== %s =====\n$ %s\n' "$1" "$2" >> "$F"; bash -c "$2" >> "$F" 2>&1 || echo "(command failed: exit $?)" >> "$F"; }

{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "user: $USER"; echo "slurm: $(sinfo --version 2>/dev/null)"; } > "$F"
sec "associations (account, partition, QOS)" "sacctmgr -n -P show assoc user=$USER format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxTRESPerNode,MaxJobs,MaxSubmit,MaxWall"
QOSLIST=$(sacctmgr -n -P show assoc user="$USER" format=QOS 2>/dev/null | tr ',' '\n' | sort -u | grep . | paste -sd, -)
sec "QOS limits (${QOSLIST:-none found})" "sacctmgr -n -P show qos ${QOSLIST:+$QOSLIST} format=Name,Priority,MaxWall,MaxTRES,MaxTRESPerNode,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GrpTRES,MinTRES,Flags"
sec "partitions" "sinfo -o '%P|%a|%l|%D|%c|%m|%G|%f' "
sec "partition $PARTITION (scontrol)" "scontrol show partition $PARTITION"
sec "nodes of $PARTITION (state, CPUs, memory)" "sinfo -p $PARTITION -N -o '%N|%T|%c|%m|%C' | head -20"
sec "your jobs, last 2 days (sacct)" "sacct -u $USER -S $(date -d '2 days ago' +%F) -X -P --format=JobID,JobName,Partition,QOS,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,NodeList | tail -40"
sec "max resources your jobs actually got (sacct, last 2 days)" "sacct -u $USER -S $(date -d '2 days ago' +%F) -X -n -P --format=AllocCPUS,Elapsed,ReqMem | sort -t'|' -k1,1n | tail -1"
sec "profiling tools on the login node" "command -v perf; perf --version 2>&1 | head -1; echo perf_event_paranoid=\$(cat /proc/sys/kernel/perf_event_paranoid); (module -t avail 2>&1 | grep -iE 'perf|vtune|likwid|papi' | head)"

# short key: value summary for report_rudra.py
qos=$(sacctmgr -n -P show qos "${QOSLIST%%,*}" format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU 2>/dev/null | head -1)
{
  echo "date: $(hostname), $(date -Is)"; echo "user: $USER"; echo "account: $(sacctmgr -n -P show assoc user="$USER" format=Account 2>/dev/null | sort -u | paste -sd, -)"
  echo "qos: ${QOSLIST:-?}"
  IFS='|' read -r n w tpu jpu spu <<< "$qos"
  echo "qos_maxwall: ${w:-none}"; echo "qos_max_tres_per_user: ${tpu:-none}"; echo "qos_max_jobs_per_user: ${jpu:-none}"; echo "qos_max_submit_per_user: ${spu:-none}"
  echo "partition: $PARTITION"
  echo "partition_maxtime: $(scontrol show partition "$PARTITION" 2>/dev/null | grep -o 'MaxTime=[^ ]*' | cut -d= -f2)"
  echo "partition_nodes: $(scontrol show partition "$PARTITION" 2>/dev/null | grep -o ' TotalNodes=[^ ]*' | cut -d= -f2)"
  echo "partition_cpus: $(scontrol show partition "$PARTITION" 2>/dev/null | grep -o 'TotalCPUs=[^ ]*' | cut -d= -f2)"
  echo "login_perf: $(command -v perf || echo none)"
} > "$OUT/limits_summary.txt"
cat "$OUT/limits_summary.txt"
echo "full record: $F"
