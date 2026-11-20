# Step 2: run one prompt end to end on each model (pinned to the 6 P-cores).
# llama-cli opens a chat after answering: read the answer, then type /exit to go to the next model.

. "$PSScriptRoot\settings.ps1"

foreach ($f in $MODEL_FILES) {
    Write-Host "`n===== $f =====" -ForegroundColor Cyan
    $p = Start-Process "$LLAMA_BIN\llama-cli.exe" -NoNewWindow -PassThru `
         -ArgumentList "-m `"$MODELS\$f`" -p `"Explain what a CPU cache is.`" -n 100 -t 6"
    $p.ProcessorAffinity = [IntPtr]$PCORE_MASK[6]
    $p.WaitForExit()
}
