# Step 1: measure memory bandwidth with STREAM (6 threads on the 6 P-cores, run 3 times).
# Output: results\stream.txt   (use the mean "Triad" Best Rate MB/s as the bandwidth)

. "$PSScriptRoot\settings.ps1"

$env:PATH = "$WINLIBS;$env:PATH"     # 64-bit GCC (the old C:\MinGW gcc 6.3 is 32-bit and too old)
$dir = "$HOME\stream"
New-Item -ItemType Directory -Force $dir | Out-Null
if (!(Test-Path "$dir\stream.c")) {
    curl.exe -s -o "$dir\stream.c" https://www.cs.virginia.edu/stream/FTP/Code/stream.c
}

# 40M doubles per array = 320 MB per array, far larger than the 24 MB L3 cache
gcc -O3 -march=native -fopenmp -static -DSTREAM_ARRAY_SIZE=40000000 -DNTIMES=20 "$dir\stream.c" -o "$dir\stream.exe"

$env:OMP_NUM_THREADS = 6
$out = "$RESULTS\stream.txt"
"STREAM, 6 threads, affinity 0x55401" | Set-Content $out
foreach ($i in 1..3) {
    $text = cmd /c "start /b /wait /affinity 0x55401 $dir\stream.exe"
    "--- run $i ---" | Add-Content $out
    $text | Add-Content $out
    $text | Select-String "Triad"
}
Write-Host "Saved $out"
