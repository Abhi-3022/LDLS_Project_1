# Step 3: perplexity of each model on the fixed text sample (data\sample.txt), 512-token chunks, 6 P-cores.
# Output: results\perplexity.txt  (one "Final estimate" line per model; lower is better)

. "$PSScriptRoot\settings.ps1"

$sample = "$REPO\data\sample.txt"
$out = "$RESULTS\perplexity.txt"
"model,final_estimate" | Set-Content $out

foreach ($f in $MODEL_FILES) {
    $log = "$RESULTS\perplexity_$($f -replace '\.gguf','').log"
    Run-Pinned "$LLAMA_BIN\llama-perplexity.exe" "-m `"$MODELS\$f`" -f `"$sample`" -c 512 -t 6" `
               $PCORE_MASK[6] "$log.out" $log
    $line = Select-String -Path $log, "$log.out" -Pattern "Final estimate" | Select-Object -First 1
    Write-Host "$f : $($line.Line)"
    "$f,$($line.Line.Trim())" | Add-Content $out
}
Write-Host "Saved $out"
