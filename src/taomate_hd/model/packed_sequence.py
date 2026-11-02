# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Native MiniMax H3 T2VA/FL2VA packed-sequence builder.

The row layout and fp64 position-grid math are kept byte-for-byte compatible
with the pinned Apache-2.0 MiniMax H3 reference behavior.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import torch
from .architecture import MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT

MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = ((0,), (-1,), (0, -1))

_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _keyframe_cond_frame_indices(
    *, include_keyframe_cond: bool, keyframe_frame_indices: list[int] | tuple[int, ...] | None
) -> list[int]:
    if not include_keyframe_cond:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted when keyframe cond is not included"
            )
        return []
    if keyframe_frame_indices is None:
        raise ValueError("strict fl2va packed layout requires keyframe_frame_indices")
    if any(
        (isinstance(value, bool) or not isinstance(value, int) for value in keyframe_frame_indices)
    ):
        raise ValueError("strict fl2va packed layout requires integer keyframe_frame_indices")
    out = list(keyframe_frame_indices)
    if tuple(out) not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            f"strict fl2va packed layout requires keyframe_frame_indices in {MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {out!r}"
        )
    return out


def _resolve_keyframe_frame_indices(
    frame_indices: Sequence[int], *, frame_count: int | None
) -> list[int]:
    if frame_indices and frame_count is None:
        raise ValueError("frame_count is required when keyframe_frame_indices are provided")
    if frame_count is None:
        return []
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    seen: dict[int, int] = {}
    resolved: list[int] = []
    for block_index, semantic_index in enumerate(frame_indices):
        if semantic_index == -1:
            resolved_index = frame_count - 1
        elif 0 <= semantic_index < frame_count:
            resolved_index = semantic_index
        else:
            raise ValueError(
                f"keyframe frame index {semantic_index} must be -1 or in [0, {frame_count})"
            )
        previous = seen.get(resolved_index)
        if previous is not None:
            raise ValueError(
                f"keyframe frame index at block {block_index} resolves to {resolved_index}, already bound by block {previous}"
            )
        seen[resolved_index] = block_index
        resolved.append(resolved_index)
    return resolved


def _temporal_position_span(temporal_length: int) -> float:
    spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
    for token_index in range(_T_GROUP):
        spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)], dtype=torch.float64
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    audio_is_condition: bool = False,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Build one positive-branch packed layout: text/cond/audio/video/pad."""
    if min(text_len, latent_t, latent_h, latent_w, audio_t, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 packed-sequence dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")
    (ph, pw) = (latent_h // _PATCH_H, latent_w // _PATCH_W)
    frame_rows = ph * pw
    cond_frame_indices = _keyframe_cond_frame_indices(
        include_keyframe_cond=include_keyframe_cond, keyframe_frame_indices=keyframe_frame_indices
    )
    resolved_cond_frame_indices = _resolve_keyframe_frame_indices(
        cond_frame_indices, frame_count=frame_count
    )
    cond_rows = len(cond_frame_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = (
        torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_img_pos])
        if cond_rows
        else target_img_pos
    )
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    audio_update_mask = torch.full((audio_rows,), not audio_is_condition, dtype=torch.bool)
    text_pos = torch.arange(0, text_len)
    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    (hh, ww) = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_grid = grid[video_sl].view(latent_t, frame_rows, 3)
    video_grid[:, :, 0] = t_grid[:, None]
    video_grid[:, :, 1:] = frame[None]
    for block_index, pixel_index in enumerate(resolved_cond_frame_indices):
        cond_block = slice(
            cond_sl.start + block_index * frame_rows, cond_sl.start + (block_index + 1) * frame_rows
        )
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = float(text_len) + _temporal_position_span(latent_t) - _FRAME_RESCALE
        else:
            raise ValueError(
                f"fl2va packed layout only supports first/last keyframe anchors, got resolved frame index {pixel_index}"
            )
        grid[cond_block, 0] = cond_t
        grid[cond_block, 1:] = frame
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    token_tags[img_pos] = 0
    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "audio_update_mask": audio_update_mask,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
    }
