"""Load the official pure-PyTorch reference functions from state-spaces/mamba.

The `mamba_ssm` package cannot be imported on a CPU-only machine (its __init__
pulls in CUDA extensions and the Mamba-3 module imports Triton/TileLang/CuTe
kernels). The decode-step *reference* implementations, however, are plain
PyTorch. This module extracts those function definitions verbatim from the
pinned checkout with `ast` and executes them in a namespace containing only
torch/einops, so the code that runs is exactly the upstream code.
"""
import ast
import math
import pathlib
import subprocess
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

MAMBA_DIR = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "mamba"

# function name -> file inside the official repo
SOURCES = {
    "heavy_tail_activation": "mamba_ssm/modules/mamba3.py",
    "_segsum": "tests/ops/triton/test_mamba3_siso.py",
    "mamba3_siso_step_ref": "tests/ops/triton/test_mamba3_siso.py",
    "mamba3_siso_fwd_ref": "tests/ops/triton/test_mamba3_siso.py",
    "mamba3_MIMO_step_ref": "tests/ops/tilelang/test_mamba3_mimo.py",
    "apply_rotary_qk_inference_reference": "mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py",
    "selective_state_update_ref": "mamba_ssm/ops/triton/selective_state_update.py",
}


def commit() -> str:
    return subprocess.run(["git", "-C", str(MAMBA_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def load() -> SimpleNamespace:
    ns = dict(torch=torch, F=F, math=math, rearrange=rearrange, repeat=repeat,
              Optional=Optional, Tuple=Tuple, Tensor=torch.Tensor)
    for name, rel in SOURCES.items():
        src = (MAMBA_DIR / rel).read_text()
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        code = ast.get_source_segment(src, node)
        exec(compile(code, f"{MAMBA_DIR.name}/{rel}::{name}", "exec"), ns)
    return SimpleNamespace(**{k: ns[k] for k in SOURCES})


if __name__ == "__main__":
    ref = load()
    print("state-spaces/mamba @", commit())
    for k in SOURCES:
        print(f"  loaded {k:40s} from {SOURCES[k]}")
