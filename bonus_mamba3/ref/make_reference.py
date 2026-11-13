"""Build reference data for the C++ decode-step kernels.

1. Draws a random-weight Mamba-3 layer with the parameterisation of
   mamba_ssm/modules/mamba3.py (in_proj split, dt_bias, heavy-tail A, sigmoid trap,
   RMS-normed B/C, B/C biases, RoPE angles, MIMO x/z/o projections).
2. Feeds T random hidden states u_t through in_proj and the Mamba3.step()
   pre-processing to get the per-step kernel inputs.
3. Runs the *official* step references (loaded verbatim by official_ref.py) in
   float64 -> ground truth, and in float32 -> "what PyTorch fp32 gets".
4. Cross-checks: SISO step ref vs the official chunked forward ref (mamba3_siso_fwd_ref),
   and MIMO ref at rank 1 with unit projections vs the SISO ref.
5. Same for a Mamba-2 layer of identical state size via selective_state_update_ref.

Everything is written to data/<case>/ as raw little-endian arrays + meta.json.
Usage: python make_reference.py [--steps 1000] [--rank 4]
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

import official_ref

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def cfg_from_args(a):
    d_inner = a.expand * a.d_model
    H = d_inner // a.headdim
    split = int(a.d_state * 0.5)          # rope_fraction = 0.5
    split -= split % 2
    return dict(d_model=a.d_model, d_inner=d_inner, H=H, P=a.headdim, N=a.d_state,
                S=split // 2, steps=a.steps, seed=a.seed)


def rmsnorm(x, eps=1e-5):
    # B_norm / C_norm in Mamba3 are RMSNormGated(d_state) with weight initialised to 1
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def make_inputs(c, R, g):
    """Per-step kernel inputs for a rank-R layer (R = 1 is SISO)."""
    H, P, N, S, T, dm = c["H"], c["P"], c["N"], c["S"], c["steps"], c["d_model"]
    d_in_proj = 2 * c["d_inner"] + 2 * N * R + 3 * H + S
    W = torch.randn(d_in_proj, dm, generator=g) / math.sqrt(dm)
    u = torch.randn(T, dm, generator=g)
    zxBCdtAtrap = u @ W.T
    z, x, B, C, dd_dt, dd_A, trap, ang = torch.split(
        zxBCdtAtrap, [c["d_inner"], c["d_inner"], N * R, N * R, H, H, H, S], dim=-1)

    # dt_bias init exactly as Mamba3.__init__ (dt in [1e-3, 1e-1], log-uniform)
    _dt = torch.exp(torch.rand(H, generator=g) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    _dt = torch.clamp(_dt, min=1e-4)
    dt_bias = _dt + torch.log(-torch.expm1(-_dt))

    ref = official_ref.load()
    A = torch.clamp(-ref.heavy_tail_activation(dd_A), max=-1e-4)
    DT = F.softplus(dd_dt + dt_bias)
    inp = dict(
        q=rmsnorm(C.reshape(T, R, N)), k=rmsnorm(B.reshape(T, R, N)),
        v=x.reshape(T, H, P), z=z.reshape(T, H, P),
        adt=A * DT, dt=DT, trap=trap, ang=ang,
    )
    # parameters: biases init to 1 and MIMO projections init as in Mamba3.__init__,
    # each perturbed so that no rank/head is degenerate
    par = dict(
        D=1.0 + 0.1 * torch.randn(H, generator=g),
        qb=1.0 + 0.1 * torch.randn(H, R, N, generator=g),
        kb=1.0 + 0.1 * torch.randn(H, R, N, generator=g),
        mx=1.0 / R + 0.1 * torch.randn(H, R, P, generator=g),
        mz=1.0 + 0.1 * torch.randn(H, R, P, generator=g),
        mo=1.0 / R + 0.1 * torch.randn(H, R, P, generator=g),
    )
    return {k: v.float().contiguous() for k, v in inp.items()}, {k: v.float().contiguous() for k, v in par.items()}


def make_inputs_real(ckpt, token_ids, layer=0):
    """Kernel inputs from a released Mamba-3 checkpoint: layer `layer` of state-spaces/mamba3-*-1.5b,
    driven by real token embeddings (RMS-normed with the layer's pre-norm), and that layer's own
    in_proj / dt_bias / B,C norms and biases / D / MIMO projections."""
    import json as _json
    cfg = _json.loads((ckpt / "config.json").read_text())
    sc = cfg["ssm_cfg"]
    sd = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    pre = f"backbone.layers.{layer}."
    R = sc.get("mimo_rank", 4) if sc.get("is_mimo") else 1
    dm, N, P = cfg["d_model"], sc["d_state"], sc["headdim"]
    d_inner = sc["expand"] * dm
    H = d_inner // P
    S = int(N * sc["rope_fraction"]) // 2
    c = dict(d_model=dm, d_inner=d_inner, H=H, P=P, N=N, S=S, steps=len(token_ids), seed=-1,
             checkpoint=ckpt.name, layer=layer)
    f32 = lambda k: sd[pre + k].float()
    emb = sd["backbone.embedding.weight"].float()[torch.tensor(token_ids)]
    u = rmsnorm(emb) * f32("norm.weight")
    zxBCdtAtrap = u @ f32("mixer.in_proj.weight").T
    z, x, B, C, dd_dt, dd_A, trap, ang = torch.split(
        zxBCdtAtrap, [d_inner, d_inner, N * R, N * R, H, H, H, S], dim=-1)
    T = len(token_ids)
    ref = official_ref.load()
    A = torch.clamp(-ref.heavy_tail_activation(dd_A), max=-sc["A_floor"])
    DT = F.softplus(dd_dt + f32("mixer.dt_bias"))
    inp = dict(
        q=rmsnorm(C.reshape(T, R, N)) * f32("mixer.C_norm.weight"),
        k=rmsnorm(B.reshape(T, R, N)) * f32("mixer.B_norm.weight"),
        v=x.reshape(T, H, P), z=z.reshape(T, H, P), adt=A * DT, dt=DT, trap=trap, ang=ang)
    ones = torch.ones(H, R, P)
    par = dict(D=f32("mixer.D"), qb=f32("mixer.C_bias"), kb=f32("mixer.B_bias"),
               mx=f32("mixer.mimo_x") if R > 1 else ones, mz=f32("mixer.mimo_z") if R > 1 else ones,
               mo=f32("mixer.mimo_o") if R > 1 else ones)
    return c, R, {k: v.float().contiguous() for k, v in inp.items()}, {k: v.float().contiguous() for k, v in par.items()}


def real_token_ids(n):
    """first n Llama-3 token ids of the WikiText-2 sample (the Mamba-3 1.5B models use the Llama-3 vocab)"""
    import subprocess
    proj = HERE.parents[1]
    out = subprocess.run([str(proj / "third_party/llama.cpp/build-cpu/bin/llama-tokenize"),
                          "-m", str(proj / "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
                          "-f", str(proj / "data/wikitext2_test_100k.txt"), "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True).stdout
    ids = json.loads(out.strip().splitlines()[-1])
    return ids[:n]


def run_siso(ref, inp, par, dtype):
    T = inp["v"].shape[0]
    cast = lambda t: t.to(dtype)
    H = inp["v"].shape[1]
    y, st = ref.mamba3_siso_step_ref(
        Q=cast(inp["q"][None, :, :1]),            # (1, T, 1, N): one B/C group
        K=cast(inp["k"][None, :, :1]),
        V=cast(inp["v"][None]),
        ADT=cast(inp["adt"].T[None]), DT=cast(inp["dt"].T[None]), Trap=cast(inp["trap"].T[None]),
        Q_bias=cast(par["qb"][:, 0]), K_bias=cast(par["kb"][:, 0]),
        Angles=cast(inp["ang"][None, :, None, :].expand(1, T, H, -1)),
        D=cast(par["D"]), Z=cast(inp["z"][None]))
    return y[0], [s[0] for s in st]


def run_mimo(ref, inp, par, dtype):
    T, H = inp["v"].shape[:2]
    cast = lambda t: t.to(dtype)
    y, st = ref.mamba3_MIMO_step_ref(
        Q=cast(inp["q"][None, :, :, None]),       # (1, T, R, 1, N)
        K=cast(inp["k"][None, :, :, None]),
        V=cast(inp["v"][None]),
        ADT=cast(inp["adt"].T[None]), DT=cast(inp["dt"].T[None]), Trap=cast(inp["trap"].T[None]),
        Q_bias=cast(par["qb"]), K_bias=cast(par["kb"]),
        Angles=cast(inp["ang"][None, :, None, :].expand(1, T, H, -1)),
        MIMO_V=cast(par["mx"]), MIMO_O=cast(par["mo"]),
        D=cast(par["D"]), Z=cast(inp["z"][None]), MIMO_Z=cast(par["mz"]))
    return y[0], [s[0] for s in st]


def make_mamba2(c, g):
    H, P, N, T, dm = c["H"], c["P"], c["N"], c["steps"], c["d_model"]
    # Mamba-2 in_proj order [z, x, B, C, dt]; conv1d is not part of the SSM step
    W = torch.randn(2 * c["d_inner"] + 2 * N + H, dm, generator=g) / math.sqrt(dm)
    zxBCdt = torch.randn(T, dm, generator=g) @ W.T
    z, x, B, C, dt = torch.split(zxBCdt, [c["d_inner"], c["d_inner"], N, N, H], dim=-1)
    _dt = torch.exp(torch.rand(H, generator=g) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    dt_bias = _dt + torch.log(-torch.expm1(-_dt))
    A = -torch.empty(H).uniform_(1, 16, generator=g)  # A_log init range of Mamba2
    inp = dict(v=x.reshape(T, H, P), z=z.reshape(T, H, P), B=F.silu(B), C=F.silu(C), dt=dt)
    par = dict(A=A, D=1.0 + 0.1 * torch.randn(H, generator=g), dt_bias=dt_bias)
    return {k: v.float().contiguous() for k, v in inp.items()}, {k: v.float().contiguous() for k, v in par.items()}


def run_mamba2(ref, inp, par, dtype, with_dz=True):
    T, H, P = inp["v"].shape
    N = inp["B"].shape[1]
    cast = lambda t: t.to(dtype)
    state = torch.zeros(1, H, P, N, dtype=dtype)
    A = cast(par["A"])[:, None, None].expand(H, P, N)
    ys = []
    for t in range(T):
        x = cast(inp["v"][t][None])
        out = ref.selective_state_update_ref(
            state, x, cast(inp["dt"][t])[None, :, None].expand(1, H, P), A,
            cast(inp["B"][t])[None, None], cast(inp["C"][t])[None, None],
            D=cast(par["D"])[:, None].expand(H, P) if with_dz else None,
            z=cast(inp["z"][t][None]) if with_dz else None,
            dt_bias=cast(par["dt_bias"])[:, None].expand(H, P), dt_softplus=True)
        ys.append(out[0])
    return torch.stack(ys), [state[0]]


def dump(case, c, inp, par, outs, extra_meta):
    d = DATA / case
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(config=c, mamba_commit=official_ref.commit(), arrays={}, **extra_meta)
    for group in (inp, par, outs):
        for k, v in group.items():
            arr = v.detach().cpu().numpy()
            arr = arr.astype(np.float64 if arr.dtype == np.float64 else np.float32)
            arr.tofile(d / f"{k}.bin")
            meta["arrays"][k] = dict(shape=list(arr.shape), dtype=str(arr.dtype))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    # plain-text header for the C++ side: H P N S R T
    (d / "config.txt").write_text(f'{c["H"]} {c["P"]} {c["N"]} {c["S"]} {extra_meta["rank"]} {c["steps"]}\n')
    print(f"wrote {d}")


def relerr(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=2048)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", action="store_true",
                    help="only build the real-weight cases from models/mamba3/mamba3-{siso,mimo}-1.5b")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ref = official_ref.load()

    if a.real:
        ids = real_token_ids(a.steps)
        for name, case in [("mamba3-siso-1.5b", "real_siso"), ("mamba3-mimo-1.5b", "real_mimo4")]:
            ckpt = HERE.parents[1] / "models" / "mamba3" / name
            c, R, inp, par = make_inputs_real(ckpt, ids)
            y64, st64 = run_mimo(ref, inp, par, torch.float64)
            y32, st32 = run_mimo(ref, inp, par, torch.float32)
            checks = dict(torch_fp32_vs_fp64_rel=relerr(y32, y64))
            if R == 1:
                y_s, _ = run_siso(ref, inp, par, torch.float64)
                checks["siso_ref_vs_mimo_ref_r1"] = relerr(y_s, y64)
            print(name, "config:", c, "checks:", checks)
            dump(case, c, inp, par, dict(y_ref=y64, ang_ref=st64[0], ssm_ref=st64[1], y_torch32=y32, ssm_torch32=st32[1]),
                 dict(rank=R, checks=checks))
        return

    c = cfg_from_args(a)
    print("config:", c)

    for R in a.ranks:
        g = torch.Generator().manual_seed(a.seed + R)
        inp, par = make_inputs(c, R, g)
        t0 = time.time()
        y64, st64 = run_mimo(ref, inp, par, torch.float64)
        t_ref = (time.time() - t0) / c["steps"]
        y32, st32 = run_mimo(ref, inp, par, torch.float32)
        checks = dict(torch_fp32_vs_fp64_rel=relerr(y32, y64), ref_us_per_step_fp64=t_ref * 1e6)
        outs = dict(y_ref=y64, ang_ref=st64[0], ssm_ref=st64[1], y_torch32=y32, ssm_torch32=st32[1])
        if R == 1:
            # SISO: the dedicated SISO reference must agree with the MIMO ref at rank 1
            # when the rank-1 projections are all ones
            ones = {**par, "mx": torch.ones_like(par["mx"]), "mz": torch.ones_like(par["mz"]),
                    "mo": torch.ones_like(par["mo"])}
            y_s, st_s = run_siso(ref, inp, par, torch.float64)
            y_m1, _ = run_mimo(ref, inp, ones, torch.float64)
            checks["siso_ref_vs_mimo_ref_r1"] = relerr(y_s, y_m1)
            # and the step recurrence must match the official chunked (training) forward
            yf, _ = ref.mamba3_siso_fwd_ref(
                Q=inp["q"][None, :, :1].double(), K=inp["k"][None, :, :1].double(), V=inp["v"][None].double(),
                ADT=inp["adt"].T[None].double(), DT=inp["dt"].T[None].double(), Trap=inp["trap"].T[None].double(),
                Q_bias=par["qb"][:, 0].double(), K_bias=par["kb"][:, 0].double(),
                Angles=inp["ang"][None, :, None, :].expand(1, c["steps"], c["H"], -1).double(),
                D=par["D"].double(), Z=inp["z"][None].double(), dtype=torch.float64)
            checks["siso_step_ref_vs_chunked_fwd_ref"] = relerr(y_s, yf[0])
            y_s32, st_s32 = run_siso(ref, inp, par, torch.float32)
            dump("siso", c, inp, {**par, "mx": ones["mx"], "mz": ones["mz"], "mo": ones["mo"]},
                 dict(y_ref=y_s, ang_ref=st_s[0], ssm_ref=st_s[1], y_torch32=y_s32, ssm_torch32=st_s32[1]),
                 dict(rank=1, checks=dict(checks, torch_fp32_vs_fp64_rel=relerr(y_s32, y_s))))
            print("SISO checks:", checks)
        print(f"MIMO R={R} checks:", checks)
        dump(f"mimo{R}", c, inp, par, outs, dict(rank=R, checks=checks))

    g = torch.Generator().manual_seed(a.seed + 100)
    inp, par = make_mamba2(c, g)
    y64, st64 = run_mamba2(ref, inp, par, torch.float64)
    y64_core, st64_core = run_mamba2(ref, inp, par, torch.float64, with_dz=False)
    y32, st32 = run_mamba2(ref, inp, par, torch.float32)
    dump("mamba2", c, inp, par, dict(y_ref=y64, ssm_ref=st64[0], y_core_ref=y64_core, y_torch32=y32, ssm_torch32=st32[0]),
         dict(rank=0, checks=dict(torch_fp32_vs_fp64_rel=relerr(y32, y64))))


if __name__ == "__main__":
    main()
