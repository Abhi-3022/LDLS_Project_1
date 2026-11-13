#!/usr/bin/env bash
# measurement queue: waits for the thread sweep (pid 77562), then runs the remaining steps one at a time
cd /home/abhisheky/LDLS_Project_1
while kill -0 77562 2>/dev/null; do sleep 20; done
for s in lengths correct profile experts ik reference bonus plots; do
    echo "=== $(date -Iseconds) step $s" 
    bash scripts/run_all.sh $s > results/logs/step_$s.log 2>&1 || echo "step $s FAILED (see results/logs/step_$s.log)"
done
echo "=== $(date -Iseconds) queue done"
