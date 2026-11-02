# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""FP16 RoPE fusion for the bundled MiniMax H3 Video VAE."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fp16(
    T,
    COS,
    SIN,
    OUT,
    COUNT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C3: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = linear < COUNT
    dim = linear % D
    head = linear // D % H
    token = linear // (D * H) % N
    batch = linear // (D * H * N)
    base = batch * S0 + token * S1 + head * S2
    value = tl.load(T + base + dim * S3, valid, other=0).to(tl.float32)
    rotating = valid & (dim < ROT)
    partner_dim = tl.where(dim < ROT // 2, dim + ROT // 2, dim - ROT // 2)
    partner = tl.load(T + base + partner_dim * S3, rotating, other=0).to(tl.float32)
    partner = tl.where(dim < ROT // 2, -partner, partner)
    cos = (
        tl.load(COS + batch * C0 + token * C1 + dim * C3, rotating, other=1)
        .to(tl.float16)
        .to(tl.float32)
    )
    sin = (
        tl.load(SIN + batch * V0 + token * V1 + dim * V3, rotating, other=0)
        .to(tl.float16)
        .to(tl.float32)
    )
    first = (value * cos).to(tl.float16).to(tl.float32)
    second = (partner * sin).to(tl.float16).to(tl.float32)
    result = (first + second).to(tl.float16)
    result = tl.where(dim < ROT, result, value.to(tl.float16))
    tl.store(OUT + linear, result, valid)


def fused_vae_rope(tensor, rotary_pos_emb):
    (cos, sin) = rotary_pos_emb
    output = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    (_, tokens, heads, dim) = tensor.shape
    _rope_fp16[triton.cdiv(tensor.numel(), 1024),](
        tensor,
        cos,
        sin,
        output,
        tensor.numel(),
        N=tokens,
        H=heads,
        D=dim,
        ROT=cos.shape[-1],
        S0=tensor.stride(0),
        S1=tensor.stride(1),
        S2=tensor.stride(2),
        S3=tensor.stride(3),
        C0=0 if cos.shape[0] == 1 else cos.stride(0),
        C1=cos.stride(1),
        C3=cos.stride(3),
        V0=0 if sin.shape[0] == 1 else sin.stride(0),
        V1=sin.stride(1),
        V3=sin.stride(3),
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
