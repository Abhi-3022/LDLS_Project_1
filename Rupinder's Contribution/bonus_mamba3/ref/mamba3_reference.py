"""Official Mamba-3 / Mamba-2 decode-step references, run on CPU in float64.

The three reference functions are loaded verbatim from the official repository
(state-spaces/mamba) by parsing the source files, so the GPU-only imports in those
files (cutlass, triton, tilelang) are never executed:

  mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py :: apply_rotary_qk_inference_reference
  mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py            :: selective_state_update_fused_ref_v2
  mamba_ssm/ops/triton/selective_state_update.py         :: selective_state_update_ref   (Mamba-2)

One Mamba-3 decode step = the sequence used in Mamba3.step() (mamba_ssm/modules/mamba3.py):
  1. rotate B and C by the accumulated data-dependent angle, after adding B/C biases
  2. exponential-trapezoidal state update using the current and the previous (B, x)
  3. output y = state . C + D x, gated by silu(z), summed over the MIMO rank
in_proj, out_proj and the B/C RMSNorm are ordinary matmuls/norms and are not part of
the kernel; the kernel starts from their outputs.

Usage:
  python mamba3_reference.py --repo <path to state-spaces/mamba> --out case.bin
         [--heads 16 --headdim 64 --dstate 128 --rank 4 --steps 1000 --seed 0]
"""
import argparse
import ast
import math
import struct
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

REF_FUNCS = [
    ("mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py", "apply_rotary_qk_inference_reference"),
    ("mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py", "selective_state_update_fused_ref_v2"),
    ("mamba_ssm/ops/triton/selective_state_update.py", "selective_state_update_ref"),
]


def load_official(repo: Path) -> dict:
    ns = {"torch": torch, "F": F, "math": math, "Optional": Optional,
          "rearrange": rearrange, "repeat": repeat}
    for rel, name in REF_FUNCS:
        src = (repo / rel).read_text(encoding="utf-8")
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(repo / rel), "exec"), ns)
    return ns


def heavy_tail_activation(x):  # copied from mamba_ssm/modules/mamba3.py
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", default="case.bin")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--dstate", type=int, default=128)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ref = load_official(a.repo)
    rot_ref = ref["apply_rotary_qk_inference_reference"]
    m3_ref = ref["selective_state_update_fused_ref_v2"]
    m2_ref = ref["selective_state_update_ref"]

    H, P, N, R, T = a.heads, a.headdim, a.dstate, a.rank, a.steps
    S = (N // 2) // 2  # rope_fraction = 0.5 (module default): num_rope_angles = d_state/4
    g = torch.Generator().manual_seed(a.seed)
    rn = lambda *s: torch.randn(*s, generator=g)

    # ---- layer parameters (float32, as stored in a checkpoint) ----
    D = rn(H)
    B_bias = 1 + 0.1 * rn(H, R, N)
    C_bias = 1 + 0.1 * rn(H, R, N)
    xproj = 1.0 / R + 0.1 * rn(R, H, P)
    zproj = 1.0 + 0.1 * rn(R, H, P)
    outproj = 1.0 / R + 0.1 * rn(R, H, P)
    _dt = torch.exp(torch.rand(H, generator=g) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
    dt_bias = _dt + torch.log(-torch.expm1(-_dt))

    # ---- per-step inputs: the outputs of in_proj (+ activations), float32 ----
    def rms(v):
        return v / v.pow(2).mean(-1, keepdim=True).sqrt()
    steps = []
    for _ in range(T):
        steps.append(dict(
            x=rn(H, P), z=rn(H, P),
            B=rms(rn(R, N)), C=rms(rn(R, N)),               # after B_norm / C_norm, ngroups = 1
            dt=F.softplus(rn(H) + dt_bias),
            A=torch.clamp(-heavy_tail_activation(rn(H)), max=-1e-4),
            trap=torch.sigmoid(rn(H)),
            ang=rn(S),
        ))

    f64 = lambda t: t.to(torch.float64)

    def run_mamba3(rank, rotate_pairwise, Bb, Cb, xp, zp, op):
        state = torch.zeros(1, H, P, N, dtype=torch.float64)
        angle = torch.zeros(1, H, S, dtype=torch.float64)
        k_state = torch.zeros(1, rank, H, N, dtype=torch.float64)
        v_state = torch.zeros(1, H, P, dtype=torch.float64)
        ys = []
        for s in steps:
            Bh = f64(s["B"][:rank])[None, :, None, :].expand(1, rank, H, N)
            Ch = f64(s["C"][:rank])[None, :, None, :].expand(1, rank, H, N)
            dt, A, trap = f64(s["dt"])[None], f64(s["A"])[None], f64(s["trap"])[None]
            ang = f64(s["ang"])[None, None, :].expand(1, H, S)
            Crot, Brot, angle = rot_ref(q=Ch, k=Bh, angle_state=angle, angle_proj=ang, dt=dt,
                                        bias_q=f64(Cb), bias_k=f64(Bb), conjugate=False,
                                        rotate_pairwise=rotate_pairwise)
            x, z = f64(s["x"])[None], f64(s["z"])[None]
            y, state = m3_ref(state, A, Brot, Crot, f64(xp), x, f64(zp), z, dt, k_state, v_state,
                              trap, f64(D), f64(op), compute_dtype=torch.float64)
            k_state, v_state = Brot, x
            ys.append(y[0])
        return torch.stack(ys), state[0]

    # SISO: rank 1, interleaved RoPE pairs; Mamba3.step uses all-ones x/z/out projections
    ones = torch.ones(1, H, P)
    y_siso, s_siso = run_mamba3(1, True, rearrange(B_bias[:, :1], "h r n -> r h n"),
                                rearrange(C_bias[:, :1], "h r n -> r h n"), ones, ones, ones)
    # MIMO: rank R, halved RoPE pairs (i, i + N/2), as in Mamba3.step for is_mimo=True
    y_mimo, s_mimo = run_mamba3(R, False, rearrange(B_bias, "h r n -> r h n"),
                                rearrange(C_bias, "h r n -> r h n"), xproj, zproj, outproj)

    # Mamba-2 at the same state size: official selective_state_update_ref, per-head A and dt
    state = torch.zeros(1, H, P, N, dtype=torch.float64)
    ys = []
    for s in steps:
        out = m2_ref(state, f64(s["x"])[None], f64(s["dt"])[None, :, None].expand(1, H, P),
                     f64(s["A"])[:, None, None].expand(H, P, N), f64(s["B"][0])[None, None],
                     f64(s["C"][0])[None, None], D=f64(D)[:, None].expand(H, P), z=f64(s["z"])[None])
        ys.append(out[0])
    y_m2, s_m2 = torch.stack(ys), state[0]

    # ---- write the case file (read back by load_case() in src/verify.cpp and src/ggml_m3op.cpp) ----
    with open(a.out, "wb") as f:
        f.write(b"M3REF001")
        f.write(struct.pack("<6i", H, P, N, R, S, T))
        for t in (D, B_bias, C_bias, xproj, zproj, outproj):
            f.write(t.to(torch.float32).contiguous().numpy().tobytes())
        for s in steps:
            for k in ("x", "z", "B", "C", "dt", "A", "trap", "ang"):
                f.write(s[k].to(torch.float32).contiguous().numpy().tobytes())
        for y, st in ((y_siso, s_siso), (y_mimo, s_mimo), (y_m2, s_m2)):
            f.write(y.contiguous().numpy().tobytes())
            f.write(st.contiguous().numpy().tobytes())

    for name, y, st in (("SISO", y_siso, s_siso), ("MIMO", y_mimo, s_mimo), ("Mamba-2", y_m2, s_m2)):
        print(f"{name:8s} |y| max {y.abs().max():.3g}  mean {y.abs().mean():.3g}   |state| max {st.abs().max():.3g}")
    print(f"wrote {a.out}: H={H} P={P} N={N} R={R} rope_angles={S} steps={T}")


if __name__ == "__main__":
    main()
