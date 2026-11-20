# Step 9 (own experiment, part B): is the weak thread scaling caused by the clock dropping as more cores work?
# Turbo boost is switched off (all cores then run at the same base clock), and the thread-scaling test is repeated:
#   1B Q4_K_M, 1-6 threads (one per P-core), prefill 512 tokens and decode 64 tokens, 3 repetitions each.
# Compare with the normal sweep: prefill 512 speed-up at 6 threads was 4.07x with boost on.
# Output: results\boost_off\capped_scaling.csv   (threads, test, tok_s)   about 5 minutes
#
# If powercfg reports "access denied", run PowerShell as Administrator.
# The script always switches boost back on at the end (also if it is stopped with Ctrl+C).

. "$PSScriptRoot\settings.ps1"

$out = "$RESULTS\boost_off"
New-Item -ItemType Directory -Force $out | Out-Null
$csv = "$out\capped_scaling.csv"
$model = "$MODELS\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
$tests = [ordered]@{ prefill512 = "-p 512 -n 0"; decode64 = "-p 0 -n 64" }

# PERFBOOSTMODE: 0 = disabled, 2 = aggressive (Windows default on this laptop)
function Set-Boost($mode) {
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE $mode
    powercfg /setactive SCHEME_CURRENT
}

try {
    Set-Boost 0
    Start-Sleep 5
    # Check: with boost off, "% Processor Performance" of a busy P-core should stay at or below 100 (= base clock)
    Write-Host "Boost off. Clock of P-core 10 (% of base):"
    typeperf "\Processor Information(0,10)\% Processor Performance" -sc 3

    "threads,test,tok_s" | Set-Content $csv
    foreach ($t in 1..6) {
        foreach ($test in $tests.Keys) {
            $json = "$out\t${t}_$test.json"
            Run-Pinned "$LLAMA_BIN\llama-bench.exe" "-m `"$model`" -t $t -r 3 -o json $($tests[$test])" `
                       $PCORE_MASK[$t] $json "$out\t${t}_$test.err"
            $ts = (Get-Content $json -Raw | ConvertFrom-Json)[0].avg_ts
            "$t,$test,$ts" | Add-Content $csv
            Write-Host ("{0} threads  {1,-10} {2,8:F1} tok/s" -f $t, $test, $ts)
        }
    }
}
finally {
    Set-Boost 2
    Write-Host "Boost switched back on."
}
Write-Host "Saved $csv"
