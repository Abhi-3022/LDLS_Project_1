# Step 5: benchmark sweep with llama-bench.
#   3 model files x 1-6 threads x
#   prefill: prompt lengths 512, 4k, largest allowed (16k for 1B, 8k for 3B)
#   decode : context lengths 1k, 8k, largest allowed; 64 tokens generated after filling the context
# Every run is repeated 3 times (-r 3); llama-bench reports mean and standard deviation.
# Free memory is logged before every run and peak memory is sampled every 200 ms during it.
#
# Output: results\runs\<run>.csv  (one llama-bench CSV per run)
#         results\peak_memory.csv (run, free memory before, peak memory in GiB)
# Finished runs are skipped, so the script can be stopped and restarted. Delete a run's CSV to redo it.
# Takes several hours: plug in the charger, close other apps, disable sleep (powercfg /change standby-timeout-ac 0).

. "$PSScriptRoot\settings.ps1"

$runs = "$RESULTS\runs"
New-Item -ItemType Directory -Force $runs | Out-Null
$memFile = "$RESULTS\peak_memory.csv"
if (!(Test-Path $memFile)) { "run,free_gb_before,peak_rss_gb" | Set-Content $memFile }

$largest = @{
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf" = 16384
    "Llama-3.2-1B-Instruct-Q8_0.gguf"   = 16384
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf" = 8192
}

function Bench($name, $model, $threads, $testArgs) {
    $csv = "$runs\$name.csv"
    if ((Test-Path $csv) -and (@(Get-Content $csv).Count -ge 2)) { Write-Host "skip $name"; return }

    $freeGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
    $arguments = "-m `"$MODELS\$model`" -t $threads -r 3 -o csv $testArgs"
    $p = Start-Process "$LLAMA_BIN\llama-bench.exe" -ArgumentList $arguments -NoNewWindow -PassThru `
         -RedirectStandardOutput $csv -RedirectStandardError "$runs\$name.err"
    $p.ProcessorAffinity = [IntPtr]$PCORE_MASK[$threads]

    $peak = 0
    while (-not $p.HasExited) {
        try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
        Start-Sleep -Milliseconds 200
    }
    "$name,$freeGB,$([math]::Round($peak / 1GB, 3))" | Add-Content $memFile
    Write-Host "$(Get-Date -Format HH:mm:ss) done $name"
}

foreach ($model in $MODEL_FILES) {
    $short = $model -replace '\.gguf', ''
    $lengths = @(512, 4096, $largest[$model]) | Select-Object -Unique
    $contexts = @(1024, 8192, $largest[$model]) | Select-Object -Unique
    foreach ($t in 1..6) {
        foreach ($n in $lengths)  { Bench "prefill_${short}_p${n}_t$t"  $model $t "-p $n -n 0" }
        foreach ($d in $contexts) { Bench "decode_${short}_ctx${d}_t$t" $model $t "-p 0 -n 64 -d $d" }
    }
}
