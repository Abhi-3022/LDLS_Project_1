# Shared settings for all PowerShell scripts. Edit the paths if your folders differ.

$LLAMA_BIN  = "$HOME\llama.cpp\build\bin\Release"         # llama.cpp CPU build
$LLAMA_DBG  = "$HOME\llama.cpp\build\bin\RelWithDebInfo"  # same code with debug symbols (for VTune)
$MODELS     = "$HOME\models"                              # the three GGUF files
$REPO       = Split-Path $PSScriptRoot -Parent
$RESULTS    = "$REPO\results"
$WINLIBS    = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"

$MODEL_FILES = @(
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)

# One thread per P-core. Coreinfo: logical 0-1 and 10-19 are the six P-cores (2-9 E-cores, 20-21 LP E-cores).
# We use logical 0, 10, 12, 14, 16, 18 (one per physical P-core).
$PCORE_MASK = @{ 1 = 0x1; 2 = 0x401; 3 = 0x1401; 4 = 0x5401; 5 = 0x15401; 6 = 0x55401 }

New-Item -ItemType Directory -Force $RESULTS | Out-Null

# Start a program pinned to the given cores and wait for it.
# (llama.cpp's own -C option had no effect on Windows, so we pin from outside.)
function Run-Pinned($exe, $arguments, $mask, $stdoutFile, $stderrFile) {
    $p = Start-Process $exe -ArgumentList $arguments -NoNewWindow -PassThru `
         -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $p.ProcessorAffinity = [IntPtr]$mask
    $p.WaitForExit()
}
