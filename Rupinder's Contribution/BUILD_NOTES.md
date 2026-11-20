# Build notes

Everything was built and run on Windows 11 Home (build 26200), PowerShell, on the laptop described in
[hardware.md](hardware.md). All commands are PowerShell.

## 1. Tools

| Tool | Used for | Install |
|---|---|---|
| Git | getting llama.cpp | `winget install Git.Git` |
| CMake | building llama.cpp | `winget install Kitware.CMake` |
| Visual Studio 2022 Build Tools, "Desktop development with C++" | MSVC compiler for llama.cpp | Visual Studio Installer (the `winget` install of BuildTools failed for us; the installer GUI worked) |
| Python 3 + packages in `requirements.txt` | prediction, analysis, plots | `pip install -r requirements.txt` |
| WinLibs GCC 64-bit | STREAM and the Mamba-3 bonus (needs OpenMP + `-static`) | `winget install BrechtSanders.WinLibs.POSIX.UCRT` |
| Intel VTune Profiler | per-operator profile (step 6) | Intel oneAPI installer, VTune only |
| Sysinternals Coreinfo | which logical processors are P-cores | download from Microsoft Sysinternals |

An old 32-bit `C:\MinGW` gcc 6.3 was on the PATH and could not build STREAM; `settings.ps1` puts the
WinLibs `bin` folder first in the PATH for the scripts that need gcc.

## 2. llama.cpp (CPU only)

```powershell
git clone https://github.com/ggml-org/llama.cpp $HOME\llama.cpp
cd $HOME\llama.cpp
git checkout a946eb541          # build b12431, the version used for every result in this repository

# Release build: used for all benchmarks
cmake -B build -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=ON
cmake --build build --config Release -j

# Same code with debug symbols: used only for VTune (function names in the profile)
cmake --build build --config RelWithDebInfo -j
```

Binaries end up in `build\bin\Release` and `build\bin\RelWithDebInfo` (paths set in `scripts\settings.ps1`).

**Check the build.** CMake detects the CPU's instruction sets itself (`GGML_NATIVE` is on by default).
`llama-cli` prints a `system_info` line at start-up; it must show `AVX2 = 1`, `FMA = 1`, `F16C = 1`,
`OPENMP = 1` and `REPACK = 1`. The 155H has no AVX-512, so `AVX512 = 0` is expected.

```powershell
& "$HOME\llama.cpp\build\bin\Release\llama-cli.exe" --version     # shows build 12431 (a946eb541)
```

## 3. Models

Bartowski GGUF conversions of Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct, saved in `$HOME\models`:

```powershell
mkdir $HOME\models -Force
cd $HOME\models
$base1 = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main"
$base3 = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"
curl.exe -L -o Llama-3.2-1B-Instruct-Q4_K_M.gguf "$base1/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
curl.exe -L -o Llama-3.2-1B-Instruct-Q8_0.gguf   "$base1/Llama-3.2-1B-Instruct-Q8_0.gguf"
curl.exe -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf "$base3/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

Per the project rules: Q4_K_M for both models, Q8_0 only for the 1B, no F16.
Weight tensor bytes (read by `scripts/step4_predict_decode.py`): 799,862,912 / 1,313,251,456 / 2,011,539,712.

## 4. STREAM

`scripts\step1_stream.ps1` downloads `stream.c` and compiles it with:

```
gcc -O3 -march=native -fopenmp -static -DSTREAM_ARRAY_SIZE=40000000 -DNTIMES=20 stream.c -o stream.exe
```

40 M doubles = 320 MB per array, far larger than the 24 MB L3. Run with `OMP_NUM_THREADS=6` and
affinity `0x55401` (one thread per P-core).

## 5. Things that did not work as expected (and what we did)

| Problem | What we found | What we did |
|---|---|---|
| `llama-bench -C <mask>` / `--cpu-mask` | Had no effect on Windows: threads still ran on E-cores | Pin from outside: `Start-Process` + `$p.ProcessorAffinity = mask` (function `Run-Pinned` in `settings.ps1`) |
| P-core numbering | We first assumed P-cores were logical 0-11 (mask `0x555`); wrong | Per-processor speed test + Coreinfo: P-cores are logical 0-1 and 10-19 → mask `0x55401` |
| `--no-mmap`, `-mmp 0` | Not valid options in build b12431 | The option is `-lm none` / `--load-mode none` |
| 2-thread runs noisy (up to 19% spread) | Per-core clock changes under turbo boost, not core 0 (own experiment, part A) | Reported as a limitation; part B measures the clock effect with boost off |

## 6. Power settings used for every benchmark

Charger connected, Windows power mode "Best performance", sleep disabled
(`powercfg /change standby-timeout-ac 0`), other applications closed. Turbo boost was **on** for all runs
except the boost-off experiment (`scripts\step9_experiment_boost_off.ps1`, which switches it back on at the end).
