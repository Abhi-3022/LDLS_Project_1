# Data

`sample.txt`: the fixed text sample used for the perplexity check (`scripts/step3_perplexity.ps1`),
about 1,400 words of plain English text, evaluated in 512-token chunks.

Every team that reruns the project must use the same file, because perplexity values are only
comparable on identical text. The models themselves are not stored here (see `BUILD_NOTES.md`, section 3).
