# Step 6: per-operator profile with Intel VTune (hotspots), 1B Q4_K_M, 6 P-cores.
#   prefill_4k : 4,096-token prompt
#   decode_1k  : 2,048 tokens generated at 1k context  (collection starts after 20 s, skipping the context fill)
#   decode_8k  : 1,024 tokens generated at 8k context  (collection starts after 80 s)
# Needs the RelWithDebInfo build (function names) and VTune installed.
# Output: results\vtune\<profile>.csv  (CPU time per function)

. "$PSScriptRoot\settings.ps1"

$vtune = "C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
$bench = "$LLAMA_DBG\llama-bench.exe"
$model = "$MODELS\Llama-3.2-1B-Instruct-Q4_K_M.gguf"
$out = "$RESULTS\vtune"
New-Item -ItemType Directory -Force $out | Out-Null

$profiles = [ordered]@{
    "prefill_4k" = "-collect hotspots -result-dir `"$out\prefill_4k`" -- `"$bench`" -m `"$model`" -t 6 -r 1 -p 4096 -n 0"
    "decode_1k"  = "-collect hotspots -start-paused -resume-after=20 -result-dir `"$out\decode_1k`" -- `"$bench`" -m `"$model`" -t 6 -r 1 -p 0 -n 2048 -d 1024"
    "decode_8k"  = "-collect hotspots -start-paused -resume-after=80 -result-dir `"$out\decode_8k`" -- `"$bench`" -m `"$model`" -t 6 -r 1 -p 0 -n 1024 -d 8192"
}

foreach ($name in $profiles.Keys) {
    Write-Host "profiling $name ..."
    # Child processes inherit the affinity, so pinning VTune pins llama-bench too.
    Run-Pinned $vtune $profiles[$name] $PCORE_MASK[6] "$out\$name.log" "$out\$name.err"
    & $vtune -report hotspots -result-dir "$out\$name" -group-by function -format csv -csv-delimiter comma `
             -report-output "$out\$name.csv"
}
Write-Host "Saved CSVs in $out"
