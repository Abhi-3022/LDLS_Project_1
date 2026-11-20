# Step 8 (own experiment): does leaving core 0 out of the thread mask change speed or stability?
#   masks with core 0    : 2 threads 0x401 (logical 0,10)     5 threads 0x15401 (0,10,12,14,16)
#   masks without core 0 : 2 threads 0x1400 (10,12)           5 threads 0x55400 (10,12,14,16,18)
#   1B Q4_K_M and 1B Q8_0; prefill 512 tokens and decode at 1k context; 5 repetitions each.
#   The clock of every P-core is logged during each run (% Processor Performance = % of base clock).
# Output: results\exp_pinning\summary.csv   (about 15 minutes)

. "$PSScriptRoot\settings.ps1"

$out = "$RESULTS\exp_pinning"
New-Item -ItemType Directory -Force $out | Out-Null

$cases = @(
    @{ name = "with_core0";    threads = 2; mask = 0x401 },
    @{ name = "without_core0"; threads = 2; mask = 0x1400 },
    @{ name = "with_core0";    threads = 5; mask = 0x15401 },
    @{ name = "without_core0"; threads = 5; mask = 0x55400 }
)
$models = "Llama-3.2-1B-Instruct-Q8_0", "Llama-3.2-1B-Instruct-Q4_K_M"
$tests = [ordered]@{ prefill512 = "-p 512 -n 0"; decode1k = "-p 0 -n 64 -d 1024" }
$counters = 0, 10, 12, 14, 16, 18 | ForEach-Object { "\Processor Information(0,$_)\% Processor Performance" }

"model,test,case,threads,mask,mean_ts,min_ts,max_ts,samples,clock_pct_by_proc" | Set-Content "$out\summary.csv"

foreach ($m in $models) { foreach ($test in $tests.Keys) { foreach ($c in $cases) {
    $tag = "${m}_${test}_t$($c.threads)_$($c.name)"
    $clockFile = "$out\$tag.freq.txt"
    Remove-Item $clockFile -ErrorAction SilentlyContinue

    # log the clock of each P-core once a second in the background
    $job = Start-Job -ArgumentList $counters, $clockFile -ScriptBlock {
        param($list, $file)
        while ($true) {
            foreach ($s in (Get-Counter -Counter $list).CounterSamples) {
                "{0},{1:F1}" -f ($s.Path -replace '.*\(0,(\d+)\).*', '$1'), $s.CookedValue | Add-Content $file
            }
        }
    }
    Start-Sleep 3

    Run-Pinned "$LLAMA_BIN\llama-bench.exe" "-m `"$MODELS\$m.gguf`" -t $($c.threads) -r 5 -o json $($tests[$test])" `
               $c.mask "$out\$tag.json" "$out\$tag.err"
    Stop-Job $job; Remove-Job $job

    $speeds = @((Get-Content "$out\$tag.json" -Raw | ConvertFrom-Json).samples_ts)
    $stats = $speeds | Measure-Object -Average -Minimum -Maximum
    $clock = (Import-Csv $clockFile -Header proc, pct | Group-Object proc | Sort-Object { [int]$_.Name } |
              ForEach-Object { "p{0}={1:F0}" -f $_.Name, ($_.Group | Measure-Object pct -Average).Average }) -join " "
    "{0},{1},{2},{3},0x{4:X},{5:F2},{6:F2},{7:F2},{8},{9}" -f $m, $test, $c.name, $c.threads, $c.mask,
        $stats.Average, $stats.Minimum, $stats.Maximum, ($speeds -join " "), $clock | Add-Content "$out\summary.csv"
    Write-Host "$(Get-Date -Format HH:mm:ss) done $tag"
    Start-Sleep 20      # let the laptop settle between cases
}}}
Write-Host "Saved $out\summary.csv"
