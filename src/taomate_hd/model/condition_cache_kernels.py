# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Exact row restoration for the LOW condition-cache Q/K/V tensors."""

import triton
import triton.language as tl


@triton.jit
def _restore_condition_qkv(
    Q,
    K,
    V,
    ACTIVE_ROWS,
    C0_ROWS,
    C0_K,
    C0_V,
    C1_ROWS,
    C1_K,
    C1_V,
    OUT,
    N_ACTIVE,
    N_C0,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    c0_k_row_stride,
    c0_v_row_stride,
    c1_k_row_stride,
    c1_v_row_stride,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles: tl.constexpr = (WIDTH + BLOCK - 1) // BLOCK
    row = tl.program_id(0) // tiles
    col = tl.program_id(0) % tiles * BLOCK + tl.arange(0, BLOCK)
    mask = col < WIDTH
    if row < N_ACTIVE:
        destination = tl.load(ACTIVE_ROWS + row)
        q_value = tl.load(Q + row * q_row_stride + col, mask, other=0)
        k_value = tl.load(K + row * k_row_stride + col, mask, other=0)
        v_value = tl.load(V + row * v_row_stride + col, mask, other=0)
    else:
        q_value = tl.full((BLOCK,), 0, Q.dtype.element_ty)
        if row < N_ACTIVE + N_C0:
            source = row - N_ACTIVE
            destination = tl.load(C0_ROWS + source)
            k_value = tl.load(C0_K + source * c0_k_row_stride + col, mask, other=0)
            v_value = tl.load(C0_V + source * c0_v_row_stride + col, mask, other=0)
        else:
            source = row - N_ACTIVE - N_C0
            destination = tl.load(C1_ROWS + source)
            k_value = tl.load(C1_K + source * c1_k_row_stride + col, mask, other=0)
            v_value = tl.load(C1_V + source * c1_v_row_stride + col, mask, other=0)
    destination = destination * (3 * WIDTH) + col
    tl.store(OUT + destination, q_value, mask)
    tl.store(OUT + destination + WIDTH, k_value, mask)
    tl.store(OUT + destination + 2 * WIDTH, v_value, mask)


def fused_restore_condition_qkv(q, k, v, *, active_rows, total_rows, cached_parts):
    """Match full-QKV index_copy/index_fill for one or two cached documents.

    Active rows and cached document rows partition the full sequence. The
    attention projections and cached index_select results have contiguous
    [heads, dim] tails; their row strides can differ (notably split-QKV V).
    The output allocation, interleaved layout, and BF16 values are unchanged.
    """
    if len(cached_parts) not in (1, 2):
        raise ValueError("LOW QKV restore requires one or two cached documents")
    (rows0, key0, value0) = cached_parts[0]
    (rows1, key1, value1) = cached_parts[-1]
    width = int(q.shape[1]) * int(q.shape[2])
    full = q.new_empty((total_rows, 3, *q.shape[1:]))
    _restore_condition_qkv[total_rows * triton.cdiv(width, 1024),](
        q,
        k,
        v,
        active_rows,
        rows0,
        key0,
        value0,
        rows1,
        key1,
        value1,
        full,
        int(active_rows.numel()),
        int(rows0.numel()),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        key0.stride(0),
        value0.stride(0),
        key1.stride(0),
        value1.stride(0),
        WIDTH=width,
        BLOCK=1024,
        num_warps=4,
    )
    return full.unbind(dim=1)
