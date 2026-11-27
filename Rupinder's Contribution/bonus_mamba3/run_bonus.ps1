# Mamba-3 bonus, end to end on Windows (PowerShell). Run from the bonus_mamba3 folder:
#
#   .\run_bonus.ps1                 # everything: reference, build, verify (16 and 64 heads), bench, plot, ggml op
#   .\run_bonus.ps1 -SkipBench      # skip the timing sweep (the slow part)
#   .\run_bonus.ps1 -SkipGgml       # skip the ggml stretch goal
#
# Every step writes its output to results\ as plain text, so the files read the same on GitHub.
param(
    [string]$MambaRepo = "$HOME\mamba-official",    # state-spaces/mamba at e9594ce (reference only)
    [string]$LlamaRepo = "$HOME\llama.cpp",         # llama.cpp at a946eb541 (b12431): its ggml is used for the op
    [string]$WinLibs   = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin",
    [switch]$ForceReference,
    [switch]$SkipBench,
    [switch]$SkipGgml
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force data, build, results | Out-Null
if (Test-Path $WinLibs) { $env:PATH = "$WinLibs;$env:PATH" }

function Step($name) { Write-Host "`n== $name ==" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
# run a program, show its output and save it as plain ASCII text (Tee-Object would write UTF-16)
function RunLogged($exe, $arguments, $log) {
    $out = & $exe @arguments
    $code = $LASTEXITCODE
    $out | Out-File -Encoding ascii $log
    $out | Write-Host
    $global:LASTEXITCODE = $code
}
# run a program pinned to the logical processors in $mask with $threads OpenMP threads; stdout goes to $out
function RunPinned($exe, $arguments, $threads, $mask, $out) {
    $env:OMP_NUM_THREADS = $threads
    $sp = @{ FilePath = $exe; NoNewWindow = $true; PassThru = $true; RedirectStandardOutput = $out }
    if ($arguments) { $sp.ArgumentList = $arguments }
    $p = Start-Process @sp
    $null = $p.Handle                  # keep the handle, or ExitCode can come back empty in Windows PowerShell
    try { $p.ProcessorAffinity = [IntPtr]$mask } catch { if (-not $p.HasExited) { throw } }  # a very short run may already be done
    $p.WaitForExit()
    Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
    $global:LASTEXITCODE = $p.ExitCode
}

$gcc = (Get-Command g++ -ErrorAction SilentlyContinue)
if (-not $gcc) { throw "g++ not found: install WinLibs (see README) or pass -WinLibs <path to mingw64\bin>" }
Write-Host "compiler: $((& g++ --version)[0])"

Step "1. reference (official PyTorch reference, fp64)"
if ($ForceReference -or -not (Test-Path data\case.bin)) {
    python ref\mamba3_reference.py --repo $MambaRepo --out data\case.bin; Check "reference"
} else { Write-Host "data\case.bin exists (use -ForceReference to rebuild it)" }

Step "2. build kernels, verify and bench"
g++ -O2 -std=c++17 -fopenmp -fno-tree-vectorize -fno-tree-slp-vectorize -c src/plain.cpp -o build/plain.o; Check "plain.cpp"
g++ -O3 -std=c++17 -fopenmp -mavx2 -mfma -c src/avx2.cpp -o build/avx2.o; Check "avx2.cpp"
g++ -O2 -std=c++17 -fopenmp -static src/verify.cpp build/plain.o build/avx2.o -o build/verify.exe; Check "verify"
g++ -O2 -std=c++17 -fopenmp -static src/bench.cpp build/plain.o build/avx2.o -o build/bench.exe; Check "bench"

Step "3. verify all 6 kernels over 1,000 steps"
RunLogged .\build\verify.exe @("data\case.bin", "1e-4") results\verify_output.txt; Check "verification"

Step "3b. verify again at the benchmark shape (64 heads)"
if ($ForceReference -or -not (Test-Path data\case_h64.bin)) {
    python ref\mamba3_reference.py --repo $MambaRepo --heads 64 --out data\case_h64.bin; Check "reference (64 heads)"
}
RunLogged .\build\verify.exe @("data\case_h64.bin", "1e-4") results\verify_output_h64.txt; Check "verification (64 heads)"

if (-not $SkipBench) {
    Step "4. benchmark pinned to P-cores (1 thread on processor 10, 6 threads one per P-core)"
    RunPinned .\build\bench.exe $null 1 0x400   results\bench_t1.csv; Check "bench (1 thread)"
    RunPinned .\build\bench.exe $null 6 0x55401 results\bench_t6.csv; Check "bench (6 threads)"
    Step "5. plot"
    python plot_bench.py results\bench_t1.csv results\bench_t6.csv --out results\mamba3_bench.png; Check "plot"
}

if (-not $SkipGgml) {
    Step "6. stretch goal: kernels as a ggml operator in a small ggml graph"
    if (-not (Test-Path "$LlamaRepo\ggml\include\ggml.h")) {
        Write-Warning "llama.cpp not found at $LlamaRepo, skipping (pass -LlamaRepo <path>)"
    } else {
        # A separate MinGW build of the same ggml: the MSVC build used for the main results stays untouched,
        # and MSVC libraries cannot be linked with g++ objects.
        $gb = "$LlamaRepo\build-mingw"
        $lib = "$gb\ggml\src"
        if (-not (Test-Path "$lib\ggml-cpu.a")) {
            cmake -S $LlamaRepo -B $gb -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ `
                -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON `
                -DGGML_F16C=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
                -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF; Check "cmake configure (ggml)"
            cmake --build $gb --target ggml -j 8; Check "ggml build"
        }
        g++ -O2 -std=c++17 -fopenmp -static "-I$LlamaRepo\ggml\include" src/ggml_m3op.cpp build/plain.o build/avx2.o `
            "$lib\ggml.a" "$lib\ggml-cpu.a" "$lib\ggml-base.a" -o build/ggml_m3op.exe; Check "ggml_m3op"
        # pinned like the 1-thread benchmark: one P-core (logical processor 10), one OpenMP thread, one ggml thread
        RunPinned .\build\ggml_m3op.exe @("data\case.bin", "1e-4", "1") 1 0x400 results\ggml_m3op_output.txt
        $code = $LASTEXITCODE
        Get-Content results\ggml_m3op_output.txt | Write-Host
        $global:LASTEXITCODE = $code; Check "ggml operator test"
    }
}
Write-Host "`nDone. Outputs are in results\" -ForegroundColor Green
