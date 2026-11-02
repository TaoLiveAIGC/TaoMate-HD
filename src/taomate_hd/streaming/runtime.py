# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from fractions import Fraction
import os
from typing import Any, Iterator
from .cache import CleanAVKVCache, kv_contract_from_model
from .geometry import (
    AUDIO_LATENT_RATE,
    StreamPhase,
    StreamPlan,
    VIDEO_FPS,
    canonical_continuation_plan,
    plan_native_stream,
    video_temporal_position,
    video_temporal_positions,
)
from .policy import StreamingInferencePolicy
from .session import H3StreamingSession

DENOISE_LOOP_MODULE = "taomate_hd.model.denoise"
PACKED_SEQUENCE_MODULE = "taomate_hd.model.packed_sequence"
VIDEO_POSITION_POLICY = "request_aligned_native_cycle"
CANONICAL_VIDEO_POSITION_POLICY = "global_canonical_codec_stream"
PROMPT_POSITION_POLICY = "current_prompt_precedes_request_media"
STEP_KV_PROMPT_POSITION_POLICY = "prompt_epoch_precedes_first_request_media"


@dataclass(frozen=True)
class StreamingPhaseExecution:
    phase_index: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int
    denoise_forwards: int
    clean_forwards: int
    committed_history_tokens: int
    global_block_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StreamingExecution:
    native_frame_count: int
    phase_executions: tuple[StreamingPhaseExecution, ...]
    request_task: str = "t2va"
    persistent_kv_modality: str = "audio_video"
    noisy_kv_writes: int = 0
    decode_mode: str = "official_full_clip_noncausal_video_vae"
    video_latent_offset: int = 0
    audio_latent_offset: int = 0
    starting_history_tokens: int = 0
    retained_history_tokens: int = 0
    cross_request_kv_reused: bool = False
    cross_request_kv_boundary_reset: bool = False
    cross_request_kv_boundary_reset_tokens: int = 0
    cross_request_clean_video_latent_handoff: bool = False
    streaming_policy: dict[str, object] | None = None
    request_index: int = 0
    current_prompt_rope_start: float = 0.0
    starting_history_audio_tokens: int = 0
    starting_history_video_tokens: int = 0
    retained_history_audio_tokens: int = 0
    retained_history_video_tokens: int = 0
    canonical_continuation_geometry: bool = False
    native_frame_offset: int = 0
    published_native_frames: int = 0
    video_transport_prefix_latents: int = 0
    audio_transport_prefix_latents_per_channel: int = 0
    published_video_latents: int = 0
    published_audio_latents_per_channel: int = 0
    current_text_token_count: int = 0
    text_time_start: float = 0.0
    text_time_stop: float = 0.0
    media_time_start: float = 0.0
    request_video_time_shift: float = 0.0
    audio_origin_skew: float = 0.0
    fixed_media_time_origin: int = 0
    reuse_across_requests: bool = False
    position_capture: dict[str, object] | None = None
    progressive_pipeline: dict[str, object] | None = None
    attention_backend_receipt: dict[str, object] | None = None

    @property
    def phase_count(self) -> int:
        return len(self.phase_executions)

    @property
    def denoise_forwards(self) -> int:
        return sum((item.denoise_forwards for item in self.phase_executions))

    @property
    def clean_forwards(self) -> int:
        return sum((item.clean_forwards for item in self.phase_executions))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["phases"] = payload.pop("phase_executions")
        payload.update(
            phase_count=self.phase_count,
            denoise_forwards=self.denoise_forwards,
            clean_forwards=self.clean_forwards,
            video_position_policy=CANONICAL_VIDEO_POSITION_POLICY,
            prompt_position_policy=PROMPT_POSITION_POLICY,
        )
        return payload


@dataclass(frozen=True)
class _RequestTemporalAlignment:
    request_index: int
    current_text_token_count: int
    text_time_start: Fraction
    text_time_stop: Fraction
    media_time_start: Fraction
    request_video_time_shift: Fraction
    audio_origin_skew: Fraction


def _request_temporal_alignment(
    *,
    current_text_len: int,
    fixed_media_origin: int,
    total_frame_count: int,
    total_video_latents: int,
    video_latent_offset: int,
    audio_latent_offset: int,
) -> _RequestTemporalAlignment:
    """Right-align the current text to a fixed, video-derived media origin."""
    if current_text_len <= 0:
        raise ValueError("current text must contain at least one token")
    if total_frame_count <= 0 or total_video_latents <= 0:
        raise ValueError("streaming request geometry must be positive")
    if video_latent_offset < 0 or audio_latent_offset < 0:
        raise ValueError("streaming media offsets must be non-negative")
    (request_index, remainder) = divmod(video_latent_offset, total_video_latents)
    if remainder:
        raise ValueError("video_latent_offset must start at a complete request boundary")
    total_audio_latents = round(Fraction(total_frame_count * AUDIO_LATENT_RATE, VIDEO_FPS))
    expected_audio_offset = request_index * total_audio_latents
    if audio_latent_offset != expected_audio_offset:
        raise ValueError(
            f"audio_latent_offset does not match the video request boundary: {audio_latent_offset} != {expected_audio_offset}"
        )
    request_video_time_shift = request_index * video_temporal_position(total_video_latents)
    media_time_start = Fraction(fixed_media_origin) + request_video_time_shift
    return _RequestTemporalAlignment(
        request_index=request_index,
        current_text_token_count=current_text_len,
        text_time_start=media_time_start - current_text_len,
        text_time_stop=media_time_start,
        media_time_start=media_time_start,
        request_video_time_shift=request_video_time_shift,
        audio_origin_skew=Fraction(audio_latent_offset) - request_video_time_shift,
    )


def _required_tensor(value: Any, name: str) -> Any:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _resolve_spatial_latent_shape(positive: Any, *, frame_rows: int) -> tuple[int, int]:
    """Recover the exact target latent canvas from official absolute positions."""
    import torch

    static = positive.static_kwargs
    position_ids = _required_tensor(
        static.get("img_position_ids"), "positive.static_kwargs['img_position_ids']"
    )
    target_info = static.get("img_pos_for_infer_output_info")
    if not isinstance(target_info, dict):
        raise TypeError("img_pos_for_infer_output_info must be a mapping")
    target_ids = _required_tensor(
        target_info.get("position_ids"), "img_pos_for_infer_output_info.position_ids"
    ).view(-1)
    if position_ids.ndim != 3 or int(position_ids.shape[0]) != 1:
        raise ValueError("official H3 img_position_ids must be [1, sequence, 3]")
    with torch.no_grad():
        target_grid = position_ids[0].index_select(
            0, target_ids.to(device=position_ids.device, dtype=torch.long)
        )
        latent_h = int(torch.unique(target_grid[:, 1]).numel()) * 2
        latent_w = int(torch.unique(target_grid[:, 2]).numel()) * 2
    if latent_h // 2 * (latent_w // 2) != frame_rows:
        raise RuntimeError("official H3 spatial RoPE grid and target packed-row count disagree")
    return (latent_h, latent_w)


def _current_prompt_rope_start(
    *,
    text_len: int,
    media_time_origin: int,
    video_latent_offset: int,
    total_video_latents: int,
    canonical_continuation_geometry: bool = False,
) -> float:
    """Place the current prompt immediately before its request-local media."""
    if canonical_continuation_geometry:
        return float(media_time_origin + video_temporal_position(video_latent_offset) - text_len)
    request_index = video_latent_offset // total_video_latents
    request_time_origin = request_index * video_temporal_position(total_video_latents)
    return float(media_time_origin + request_time_origin - text_len)


def _streaming_prompt_rope_start(
    *,
    text_len: int,
    media_time_origin: int,
    video_latent_offset: int,
    total_video_latents: int,
    prompt_video_latent_start: int | None,
    canonical_continuation_geometry: bool = False,
) -> float:
    """Place text immediately before its prompt segment in one long stream."""
    if prompt_video_latent_start is None:
        return _current_prompt_rope_start(
            text_len=text_len,
            media_time_origin=media_time_origin,
            video_latent_offset=video_latent_offset,
            total_video_latents=total_video_latents,
            canonical_continuation_geometry=canonical_continuation_geometry,
        )
    if prompt_video_latent_start < 0:
        raise ValueError("prompt video latent start must be non-negative")
    return float(media_time_origin + video_temporal_position(prompt_video_latent_start) - text_len)


def _first_frame_condition_position(
    *,
    canonical_continuation_geometry: bool,
    request_index: int,
    video_latent_offset: int,
    total_video_latents: int,
) -> Fraction:
    """Place a continuation anchor on the prior clean video latent."""
    if canonical_continuation_geometry and video_latent_offset > 0:
        return video_temporal_position(video_latent_offset - 1)
    if request_index == 0:
        return video_temporal_position(0)
    return (request_index - 1) * video_temporal_position(
        total_video_latents
    ) + video_temporal_position(total_video_latents - 1)


def _uses_bounded_rope_exact_layout(
    *,
    phase: StreamPhase,
    total_frame_count: int,
    total_video_latents: int,
    canonical_continuation_geometry: bool,
) -> bool:
    return (
        not canonical_continuation_geometry
        and total_frame_count == 243
        and (total_video_latents == 72)
        and (phase.frame_start == 0)
        and (phase.frame_stop == total_frame_count)
        and (phase.video_latent_start == 0)
        and (phase.video_latent_stop == total_video_latents)
        and (phase.audio_latent_start == 0)
        and (phase.audio_latent_stop == 405)
    )


def _compact_packed_grid_position_capture(
    *,
    grid: Any,
    text_rows: list[int],
    video_rows: list[int],
    audio_rows: list[list[int]],
    phase: StreamPhase,
) -> dict[str, object]:
    """Summarize values read from an already-written packed position grid."""

    def temporal(row: int) -> float:
        try:
            value = grid[row, 0]
        except TypeError:
            value = grid[row][0]
        item = getattr(value, "item", None)
        return float(item() if callable(item) else value)

    text = [temporal(row) for row in text_rows]
    video = [temporal(row) for row in video_rows]
    audio = [[temporal(row) for row in channel] for channel in audio_rows]
    return {
        "source": "actual_packed_grid",
        "text": {
            "count": len(text),
            "first": text[0],
            "last": text[-1],
            "stop_exclusive": text[-1] + 1.0,
            "unit_step": all(
                (
                    abs(current - previous - 1.0) < 0.0001
                    for (previous, current) in zip(text, text[1:])
                )
            ),
        },
        "video": {
            "unique_count": len({round(value, 5) for value in video}),
            "first": video[0],
            "last": video[-1],
        },
        "audio": {
            "channel_count": len(audio),
            "per_channel_count": len(audio[0]),
            "first": audio[0][0],
            "last": audio[0][-1],
            "channels_equal": all(
                (abs(left - right) < 0.0001 for (left, right) in zip(audio[0], audio[1]))
            ),
        },
        "phase_boundaries": [
            {
                "phase_index": phase.index,
                "frame_start": phase.frame_start,
                "frame_stop": phase.frame_stop,
                "video_latent_start": phase.video_latent_start,
                "video_latent_stop": phase.video_latent_stop,
                "audio_latent_start": phase.audio_latent_start,
                "audio_latent_stop": phase.audio_latent_stop,
            }
        ],
    }


def _merge_packed_grid_position_captures(
    captures: list[dict[str, object]],
) -> dict[str, object] | None:
    """Combine the recorded packed-position summaries."""
    if not captures:
        return None
    merged = dict(captures[0])
    if len(captures) > 1:
        merged["phases"] = [dict(capture) for capture in captures]
    return merged


def build_streaming_phase_packed_layout(
    *,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    text_token_tags: Any,
    keyframe_frame_indices: tuple[int, ...],
    total_frame_count: int,
    total_video_latents: int,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
    prompt_video_latent_start: int | None = None,
    prompt_rope_start: float | None = None,
    phase_tail_condition: bool = False,
    canonical_continuation_geometry: bool = False,
) -> dict[str, Any]:
    """Build one future-free target phase on the native H3 video cadence."""
    import importlib
    import torch

    packed_module = importlib.import_module(PACKED_SEQUENCE_MODULE)
    build = getattr(packed_module, "minimax_h3_packed_sequence", None)
    if not callable(build):
        raise RuntimeError("pinned H3 packed-sequence builder is unavailable")
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"), "positive.static_kwargs['prompt_embeds']"
    )
    text_len = int(prompt_embeds.shape[0])
    packed = build(
        text_len=text_len,
        latent_t=phase.video_latent_count,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=phase.audio_latent_count,
        audio_channel=2,
        include_keyframe_cond=bool(keyframe_frame_indices),
        keyframe_frame_indices=keyframe_frame_indices or None,
        frame_count=total_frame_count if keyframe_frame_indices else None,
    )
    text_pos = _required_tensor(packed.get("text_pos"), "text_pos").view(-1)
    tags = _required_tensor(packed.get("token_tags"), "token_tags")
    if list(text_token_tags.shape) != [text_len]:
        raise ValueError("official H3 text token tags do not match prompt embeddings")
    tags[text_pos] = text_token_tags.to(device=tags.device, dtype=tags.dtype)
    grid = _required_tensor(packed.get("img_position_ids"), "img_position_ids")
    img_pos = _required_tensor(packed.get("img_pos"), "img_pos").view(-1)
    update_mask = _required_tensor(packed.get("update_mask"), "update_mask").view(-1)
    target_pos = img_pos[update_mask]
    condition_pos = img_pos[~update_mask]
    frame_rows = latent_h // 2 * (latent_w // 2)
    if int(target_pos.numel()) != phase.video_latent_count * frame_rows:
        raise RuntimeError("phase video packed rows do not match its latent geometry")
    exact_alignment = _uses_bounded_rope_exact_layout(
        phase=phase,
        total_frame_count=total_frame_count,
        total_video_latents=total_video_latents,
        canonical_continuation_geometry=canonical_continuation_geometry,
    )
    temporal_alignment = (
        _request_temporal_alignment(
            current_text_len=text_len,
            fixed_media_origin=media_time_origin,
            total_frame_count=total_frame_count,
            total_video_latents=total_video_latents,
            video_latent_offset=video_latent_offset,
            audio_latent_offset=audio_latent_offset,
        )
        if exact_alignment
        else None
    )
    if temporal_alignment is None:
        (request_index, request_local_start) = divmod(
            video_latent_offset + phase.video_latent_start, total_video_latents
        )
        request_time_origin = request_index * video_temporal_position(total_video_latents)
        prompt_time_origin = (
            prompt_rope_start
            if prompt_rope_start is not None
            else _streaming_prompt_rope_start(
                text_len=text_len,
                media_time_origin=media_time_origin,
                video_latent_offset=video_latent_offset,
                total_video_latents=total_video_latents,
                prompt_video_latent_start=prompt_video_latent_start,
                canonical_continuation_geometry=canonical_continuation_geometry,
            )
        )
    else:
        request_index = temporal_alignment.request_index
        request_local_start = phase.video_latent_start
        request_time_origin = temporal_alignment.request_video_time_shift
        prompt_time_origin = float(temporal_alignment.text_time_start)
    grid[text_pos, 0] = prompt_time_origin + torch.arange(
        text_len, dtype=grid.dtype, device=grid.device
    )
    if canonical_continuation_geometry:
        video_positions = video_temporal_positions(
            video_latent_offset + phase.video_latent_start, phase.video_latent_count
        )
    else:
        video_positions = tuple(
            (
                request_time_origin + position
                for position in video_temporal_positions(
                    request_local_start, phase.video_latent_count
                )
            )
        )
    video_times = torch.tensor(
        [float(media_time_origin + position) for position in video_positions], dtype=grid.dtype
    )
    grid[target_pos, 0] = video_times.repeat_interleave(frame_rows)
    if keyframe_frame_indices:
        if int(condition_pos.numel()) != len(keyframe_frame_indices) * frame_rows:
            raise RuntimeError("FL2VA phase has invalid condition rows")
        for condition_index, frame_index in enumerate(keyframe_frame_indices):
            rows = condition_pos[condition_index * frame_rows : (condition_index + 1) * frame_rows]
            if phase_tail_condition:
                condition_position = (
                    video_temporal_position(video_latent_offset + phase.video_latent_start - 1)
                    if canonical_continuation_geometry
                    else request_time_origin + video_temporal_position(phase.video_latent_start - 1)
                )
            elif frame_index == 0:
                condition_position = _first_frame_condition_position(
                    canonical_continuation_geometry=canonical_continuation_geometry,
                    request_index=request_index,
                    video_latent_offset=video_latent_offset,
                    total_video_latents=total_video_latents,
                )
            else:
                condition_position = (
                    request_time_origin
                    + video_temporal_positions(total_video_latents, 1)[0]
                    - video_temporal_position(1)
                )
            condition_time = float(media_time_origin + condition_position)
            grid[rows, 0] = condition_time
    audio_pos = _required_tensor(packed.get("audio_pos"), "audio_pos").view(
        2, phase.audio_latent_count
    )
    native_audio_times = torch.arange(
        audio_latent_offset + phase.audio_latent_start,
        audio_latent_offset + phase.audio_latent_stop,
        dtype=grid.dtype,
    )
    audio_times = media_time_origin + native_audio_times
    grid[audio_pos[0], 0] = audio_times
    grid[audio_pos[1], 0] = audio_times
    if exact_alignment or (phase.index == 0 and canonical_continuation_geometry):
        packed["_taomate_hd_position_capture"] = _compact_packed_grid_position_capture(
            grid=grid,
            text_rows=[int(value) for value in text_pos.tolist()],
            video_rows=[
                int(value)
                for value in target_pos.view(phase.video_latent_count, frame_rows)[:, 0].tolist()
            ],
            audio_rows=[[int(value) for value in channel] for channel in audio_pos.tolist()],
            phase=phase,
        )
    return packed


def _build_phase_branch(
    *,
    loop_module: Any,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    model: Any,
    device: Any,
    text_token_tags: Any,
    keyframe_frame_indices: tuple[int, ...],
    total_frame_count: int,
    total_video_latents: int,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
    prompt_video_latent_start: int | None = None,
    prompt_rope_start: float | None = None,
    phase_tail_condition: bool = False,
    canonical_continuation_geometry: bool = False,
    reusable_branch: Any | None = None,
    static_runtime_plans: bool = False,
    static_layout_token: Any | None = None,
) -> Any:
    import torch

    branch_type = getattr(loop_module, "MiniMaxH3DenoiseBranch", None)
    if branch_type is None:
        raise RuntimeError("pinned H3 MiniMaxH3DenoiseBranch is unavailable")
    packed = build_streaming_phase_packed_layout(
        positive=positive,
        phase=phase,
        latent_h=latent_h,
        latent_w=latent_w,
        text_token_tags=text_token_tags,
        keyframe_frame_indices=keyframe_frame_indices,
        total_frame_count=total_frame_count,
        total_video_latents=total_video_latents,
        media_time_origin=media_time_origin,
        video_latent_offset=video_latent_offset,
        audio_latent_offset=audio_latent_offset,
        prompt_video_latent_start=prompt_video_latent_start,
        prompt_rope_start=prompt_rope_start,
        phase_tail_condition=phase_tail_condition,
        canonical_continuation_geometry=canonical_continuation_geometry,
    )
    position_capture = packed.pop("_taomate_hd_position_capture", None)
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"), "positive.static_kwargs['prompt_embeds']"
    )
    if reusable_branch is None:
        branch = branch_type(
            packed=packed,
            text_embeddings=prompt_embeds,
            token_tags=packed["token_tags"],
            device=device,
            parallel_context=model.parallel_context,
        )
        branch._taomate_hd_static_layout_token = static_layout_token
        branch._taomate_hd_static_workspace_reuses = 0
        if static_runtime_plans:
            branch.enable_inference_static_runtime_plans()
    else:
        branch = reusable_branch
        if not static_runtime_plans:
            raise RuntimeError("phase branch reuse requires static runtime plans")
        if branch._taomate_hd_static_layout_token != static_layout_token:
            raise RuntimeError("phase branch static layout token differs")
        position_ids = packed["img_position_ids"][None].to(device=device, dtype=torch.float32)
        destination = branch.static_kwargs["img_position_ids"]
        if tuple(destination.shape) != tuple(position_ids.shape):
            raise RuntimeError("phase branch reusable position shape differs")
        destination.copy_(position_ids)
        branch.img_position_ids_host = (
            packed["img_position_ids"].detach().to(dtype=torch.float32, device="cpu")
        )
        branch._x_buffer_primed = False
        branch._taomate_hd_static_workspace_reuses += 1
    refined_length = positive.static_kwargs.get("refined_prompt_embeds_length")
    if refined_length is not None:
        branch.static_kwargs["refined_prompt_embeds_length"] = int(refined_length)
    build_rope = getattr(model, "build_rope_cache", None)
    if not callable(build_rope):
        raise RuntimeError("pinned H3 model does not expose build_rope_cache")
    with torch.no_grad():
        branch.static_kwargs["rope_cache"] = build_rope(
            branch.static_kwargs["img_position_ids"], device=device
        )
    branch._taomate_hd_position_capture = position_capture
    return branch


def _resolve_global_text_token_tags(positive: Any) -> Any:
    """Recover official text/vision AdaLN tags before rebuilding phase layouts."""
    import torch

    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"), "positive.static_kwargs['prompt_embeds']"
    )
    text_len = int(prompt_embeds.shape[0])
    local_tags = _required_tensor(
        positive.static_kwargs.get("block_token_tags"), "positive.static_kwargs['block_token_tags']"
    ).view(-1)
    seq_len = int(getattr(positive, "seq_len", 0))
    if seq_len <= 0:
        raise ValueError("official H3 branch sequence length is invalid")
    if int(local_tags.numel()) == seq_len:
        global_tags = local_tags
    else:
        from taomate_hd.distributed import ulysses_all_gather_rows

        global_tags = ulysses_all_gather_rows(
            local_tags.view(-1, 1), context=positive.parallel_context
        ).view(-1)
    if int(global_tags.numel()) != seq_len or text_len > seq_len:
        raise RuntimeError("official H3 global token-tag reconstruction differs")
    tags = global_tags[:text_len].detach().to(device="cpu", dtype=torch.long)
    if bool(((tags != 0) & (tags != 1)).any().item()):
        raise ValueError("H3 prompt rows contain unsupported streaming token tags")
    return tags


@contextmanager
def _native_attention_metadata(
    branch: Any, *, live_documents_hint: tuple[int, ...] | None = None, mlp_video_stride: int = 1
) -> Iterator[None]:
    """Bind full-sequence modality tags and target-media commit rows."""
    import torch
    from taomate_hd.distributed import ulysses_all_gather_rows
    from taomate_hd.model.layers import native_stream_attention_metadata

    resolved_live_documents_hint = live_documents_hint
    if resolved_live_documents_hint is None:
        packed = branch.static_kwargs.get("packed_seq_params")
        boundaries_value = packed.get("cu_seqlens_q_host") if isinstance(packed, dict) else None
        if boundaries_value is not None:
            boundaries = tuple((int(value) for value in boundaries_value))
            if (
                len(boundaries) >= 2
                and boundaries[0] == 0
                and (boundaries[-1] == int(branch.seq_len))
                and all((left <= right for (left, right) in zip(boundaries, boundaries[1:])))
            ):
                host_target_rows: list[int] = []
                host_layouts = (
                    (getattr(branch, "img_pos", None), getattr(branch, "update_mask", None)),
                    (
                        getattr(branch, "audio_pos", None),
                        getattr(branch, "audio_update_mask", None),
                    ),
                )
                for positions, update_mask in host_layouts:
                    if positions is None or update_mask is None:
                        host_target_rows = []
                        break
                    positions = positions.view(-1)
                    update_mask = update_mask.view(-1)
                    if (
                        int(positions.numel()) != int(update_mask.numel())
                        or positions.device.type != "cpu"
                        or update_mask.device.type != "cpu"
                    ):
                        host_target_rows = []
                        break
                    host_target_rows.extend(
                        (int(value) for value in positions[update_mask].tolist())
                    )
                if host_target_rows:
                    resolved_live_documents_hint = tuple(
                        (
                            index
                            for (index, (start, stop)) in enumerate(zip(boundaries, boundaries[1:]))
                            if any((start <= row < stop for row in host_target_rows))
                        )
                    )
    metadata_key = (resolved_live_documents_hint, int(mlp_video_stride))
    metadata_cache = (
        branch._inference_native_attention_metadata_cache
        if getattr(branch, "_inference_static_runtime_plans_enabled", False)
        else None
    )
    cached = (
        metadata_cache.get(metadata_key)
        if metadata_cache is not None
        else getattr(branch, "_taomate_hd_native_attention_metadata", None)
    )
    if cached is None:
        tags = _required_tensor(
            branch.static_kwargs.get("block_token_tags"), "block_token_tags"
        ).view(-1)
        if int(tags.numel()) != int(branch.seq_len):
            tags = ulysses_all_gather_rows(tags.view(-1, 1), context=branch.parallel_context).view(
                -1
            )
        if int(tags.numel()) != int(branch.seq_len):
            raise RuntimeError("native H3 streaming token tags do not cover the sequence")
        mask = torch.zeros(int(branch.seq_len), dtype=torch.bool, device=tags.device)
        mask.index_fill_(0, branch.img_target_seq_idx.to(tags.device), True)
        mask.index_fill_(0, branch.audio_target_seq_idx.to(tags.device), True)
        if metadata_cache is not None:
            metadata_cache[metadata_key] = (tags, mask)
            branch._inference_native_attention_metadata_misses += 1
        else:
            setattr(branch, "_taomate_hd_native_attention_metadata", (tags, mask))
    else:
        (tags, mask) = cached
        if metadata_cache is not None:
            branch._inference_native_attention_metadata_hits += 1
    with native_stream_attention_metadata(
        tags,
        mask,
        live_documents_hint=resolved_live_documents_hint,
        mlp_video_stride=mlp_video_stride,
    ):
        yield


@dataclass(frozen=True)
class _RequestVideoState:
    task: str
    target_initial_rows: Any
    condition_initial_rows: Any | None
    keyframe_condition_rows: Any | None
    keyframe_frame_indices: tuple[int, ...]


def _resolve_request_video_state(
    *,
    positive: Any,
    initial_video: Any,
    keyframe_cond_rows: Any,
    frame_rows: int,
    total_video_latents: int,
) -> _RequestVideoState:
    """Admit T2VA, one first-frame anchor, or first-plus-last anchors."""
    import torch

    condition_rows = int(positive.video_target_start)
    target_rows = total_video_latents * frame_rows
    if list(initial_video.shape) != [condition_rows + target_rows, 96]:
        raise ValueError("request video rows do not match condition plus native stream geometry")
    if condition_rows == 0:
        if keyframe_cond_rows is not None:
            raise ValueError("T2VA streaming request unexpectedly carries keyframe rows")
        return _RequestVideoState(
            task="t2va",
            target_initial_rows=initial_video,
            condition_initial_rows=None,
            keyframe_condition_rows=None,
            keyframe_frame_indices=(),
        )
    if condition_rows not in {frame_rows, 2 * frame_rows} or not torch.is_tensor(
        keyframe_cond_rows
    ):
        raise NotImplementedError(
            "streaming admits one first-frame or two first/last FL2VA keyframes"
        )
    if list(keyframe_cond_rows.shape) != [condition_rows, 96]:
        raise ValueError("FL2VA keyframe rows do not match the target latent canvas")
    position_ids = _required_tensor(
        positive.static_kwargs.get("img_position_ids"), "positive.static_kwargs['img_position_ids']"
    )
    img_pos = _required_tensor(getattr(positive, "img_pos", None), "positive.img_pos")
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"), "positive.static_kwargs['prompt_embeds']"
    )
    condition_positions = position_ids[0].index_select(
        0, img_pos[:condition_rows].to(device=position_ids.device, dtype=torch.long)
    )
    expected_time = float(prompt_embeds.shape[0])
    if not bool((condition_positions[:frame_rows, 0] == expected_time).all().item()):
        raise NotImplementedError("streaming FL2VA requires the first condition at frame zero")
    if condition_rows == 2 * frame_rows:
        expected_last_time = float(
            prompt_embeds.shape[0] + video_temporal_positions(total_video_latents, 1)[0] - 5 / 3
        )
        last_times = condition_positions[frame_rows:, 0]
        if not torch.allclose(
            last_times, torch.full_like(last_times, expected_last_time), rtol=0.0, atol=1e-05
        ):
            raise NotImplementedError(
                "II2AV streaming requires the second condition at the last frame"
            )
    frame_indices = (0,) if condition_rows == frame_rows else (0, -1)
    return _RequestVideoState(
        task="fl2va_first_frame" if len(frame_indices) == 1 else "fl2va_first_last",
        target_initial_rows=initial_video[condition_rows:],
        condition_initial_rows=initial_video[:condition_rows],
        keyframe_condition_rows=keyframe_cond_rows,
        keyframe_frame_indices=frame_indices,
    )


def _phase_initial_rows(
    *,
    initial_video_rows: Any,
    initial_audio_rows: Any,
    phase: StreamPhase,
    frame_rows: int,
    total_audio_latents: int,
) -> tuple[Any, Any]:
    video = initial_video_rows[
        phase.video_latent_start * frame_rows : phase.video_latent_stop * frame_rows
    ].contiguous()
    audio = initial_audio_rows.view(2, total_audio_latents, -1)[
        :, phase.audio_latent_start : phase.audio_latent_stop
    ].contiguous()
    return (video, audio.view(-1, int(audio.shape[-1])))


def _join_phase_audio(rows: list[Any], phases: tuple[StreamPhase, ...]) -> Any:
    import torch

    by_channel: list[list[Any]] = [[], []]
    for value, phase in zip(rows, phases):
        current = value.view(2, phase.audio_latent_count, -1)
        by_channel[0].append(current[0])
        by_channel[1].append(current[1])
    return torch.cat((torch.cat(by_channel[0], dim=0), torch.cat(by_channel[1], dim=0)), dim=0)


class H3StreamingRuntimeOverlay:
    """Phase scheduler around the pinned H3 denoise loop.

    The official request validation, encoders, sigma schedules, output
    publication, and full-clip decoders remain unchanged.  This overlay only
    replaces the full target denoise loop with future-free native AV phases.
    T2VA and one explicit first-frame FL2VA input condition are admitted. The
    FL2VA condition is present only in phase zero; later phases read the clean
    AV-KV committed inside this request. Persistent long-video continuation
    starts the next request as T2VA and applies the configured bounded clean
    KV policy, without injecting the previous terminal latent a second time.
    """

    def __init__(
        self,
        *,
        plan: StreamPlan,
        server_args: Any,
        reuse_across_requests: bool = False,
        policy: StreamingInferencePolicy | None = None,
        attention_backend: str = "sdpa",
    ) -> None:
        self.plan = plan
        self.server_args = server_args
        self.reuse_across_requests = bool(reuse_across_requests)
        self.policy = policy or StreamingInferencePolicy()
        self.attention_backend = attention_backend
        if not self.reuse_across_requests:
            raise ValueError("canonical continuation geometry requires cross-request KV reuse")
        self.last_execution: StreamingExecution | None = None
        self.executions: list[StreamingExecution] = []
        self._running = False
        self._cache: CleanAVKVCache | None = None
        self._progressive_predicted_history: Any | None = None
        self._media_time_origin: int | None = None
        self._request_index = 0
        self._native_frame_offset = 0
        self._video_latent_offset = 0
        self._audio_latent_offset = 0
        self._clean_video_condition_rows: Any | None = None
        self._canonical_transport_video_rows: Any | None = None
        self._canonical_transport_audio_rows: Any | None = None

    @property
    def clean_video_condition_rows(self) -> Any | None:
        """Previous request's terminal clean video latent on CPU."""
        return self._clean_video_condition_rows

    def release_retained_state(self) -> None:
        """Release cross-request CUDA state after the final generated chunk."""
        if self._running:
            raise RuntimeError("cannot release streaming state during generation")
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        if self._progressive_predicted_history is not None:
            self._progressive_predicted_history.release()
            self._progressive_predicted_history = None
        self._clean_video_condition_rows = None
        self._canonical_transport_video_rows = None
        self._canonical_transport_audio_rows = None

    def _apply_cross_chunk_cache_policy(self, cache: CleanAVKVCache) -> None:
        recent = self.policy.recent_prefix_blocks
        cache.retain_sink_and_recent_commits(recent)

    def _run_progressive(
        self, original_loop: Any, loop_module: Any, result_type: Any, **kwargs: Any
    ) -> Any:
        """Run all three resolution transitions inside one cache transaction."""
        import torch
        from taomate_hd.progressive_resolution import lift_video_state_rows, resize_video_state_rows

        if self._running:
            raise RuntimeError("one H3 streaming overlay cannot run concurrent requests")
        self._running = True
        try:
            model = kwargs["model"]
            low_positive = kwargs["low_positive"]
            high_positive = kwargs["high_positive"]
            initial_low_video = _required_tensor(
                kwargs["initial_low_video_rows"], "initial_low_video_rows"
            )
            initial_audio = _required_tensor(kwargs["initial_audio_rows"], "initial_audio_rows")
            plan = kwargs["plan"]
            switch_position = int(kwargs["switch_position"])
            request_index = self._request_index
            canonical_continuation = request_index > 0
            continuation_groups = os.environ.get("H3_PROGRESSIVE_CONTINUATION_GROUPS")
            continuation_base_plan = (
                plan_native_stream(
                    self.plan.native_frame_count, groups_per_phase=int(continuation_groups)
                )
                if canonical_continuation and continuation_groups
                else self.plan
            )
            active_plan = (
                canonical_continuation_plan(continuation_base_plan, request_index=request_index)
                if canonical_continuation
                else self.plan
            )
            official_video_latents = self.plan.phases[-1].video_latent_stop
            official_audio_latents = self.plan.phases[-1].audio_latent_stop
            total_video_latents = active_plan.phases[-1].video_latent_stop
            total_audio_latents = active_plan.phases[-1].audio_latent_stop
            video_transport_prefix_latents = official_video_latents - total_video_latents
            audio_transport_prefix_latents = official_audio_latents - total_audio_latents
            low_frame_rows = int(plan.low.spatial_tokens_per_video_latent)
            high_frame_rows = int(plan.high.spatial_tokens_per_video_latent)
            (low_latent_h, low_latent_w) = _resolve_spatial_latent_shape(
                low_positive, frame_rows=low_frame_rows
            )
            (high_latent_h, high_latent_w) = _resolve_spatial_latent_shape(
                high_positive, frame_rows=high_frame_rows
            )
            if (low_latent_h, low_latent_w) != (
                int(plan.low.latent_height),
                int(plan.low.latent_width),
            ) or (high_latent_h, high_latent_w) != (
                int(plan.high.latent_height),
                int(plan.high.latent_width),
            ):
                raise RuntimeError(
                    "progressive streaming branch canvas differs from resolution plan"
                )
            low_request_video = _resolve_request_video_state(
                positive=low_positive,
                initial_video=initial_low_video,
                keyframe_cond_rows=kwargs.get("low_keyframe_cond_rows"),
                frame_rows=low_frame_rows,
                total_video_latents=official_video_latents,
            )
            high_template = torch.zeros(
                int(high_positive.img_pos.numel()), 96, dtype=torch.float32, device="cpu"
            )
            high_request_video = _resolve_request_video_state(
                positive=high_positive,
                initial_video=high_template,
                keyframe_cond_rows=kwargs.get("high_keyframe_cond_rows"),
                frame_rows=high_frame_rows,
                total_video_latents=official_video_latents,
            )
            if (
                low_request_video.task != high_request_video.task
                or low_request_video.keyframe_frame_indices
                != high_request_video.keyframe_frame_indices
            ):
                raise RuntimeError("progressive low/high streaming conditioning contracts differ")
            if list(initial_audio.shape) != [2 * official_audio_latents, 32]:
                raise ValueError(
                    "request audio rows do not match cumulative 40 Hz phase boundaries"
                )
            if initial_low_video.device.type != "cpu":
                raise ValueError("official H3 initial noise must enter denoise from CPU")
            transport_video_rows = None
            transport_audio_rows = None
            if canonical_continuation:
                transport_video_rows = self._canonical_transport_video_rows
                transport_audio_rows = self._canonical_transport_audio_rows
                if transport_video_rows is None or list(transport_video_rows.shape) != [
                    video_transport_prefix_latents * high_frame_rows,
                    96,
                ]:
                    raise RuntimeError(
                        "canonical continuation has no exact prior progressive video transport prefix"
                    )
                if transport_audio_rows is None:
                    raise RuntimeError("canonical continuation has no prior audio transport prefix")
                previous_audio_t = int(transport_audio_rows.shape[0]) // 2
                if (
                    list(transport_audio_rows.shape) != [2 * previous_audio_t, 32]
                    or previous_audio_t < audio_transport_prefix_latents
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior audio transport prefix"
                    )
                low_request_video = _RequestVideoState(
                    task=low_request_video.task,
                    target_initial_rows=low_request_video.target_initial_rows[
                        video_transport_prefix_latents * low_frame_rows :
                    ].contiguous(),
                    condition_initial_rows=low_request_video.condition_initial_rows,
                    keyframe_condition_rows=low_request_video.keyframe_condition_rows,
                    keyframe_frame_indices=low_request_video.keyframe_frame_indices,
                )
                initial_audio = (
                    initial_audio.view(2, official_audio_latents, -1)[
                        :, audio_transport_prefix_latents:
                    ]
                    .contiguous()
                    .view(-1, int(initial_audio.shape[-1]))
                )
            sigmas_video = list(kwargs["sigmas_video"])
            sigmas_audio = list(kwargs["sigmas_audio"])
            if (
                len(sigmas_video) != len(sigmas_audio)
                or not 0 < switch_position < len(sigmas_video) - 1
            ):
                raise ValueError("progressive streaming sigma schedule is invalid")
            low_sigmas_video = sigmas_video[: switch_position + 1]
            low_sigmas_audio = sigmas_audio[: switch_position + 1]
            high_sigmas_video = sigmas_video[switch_position:]
            high_sigmas_audio = sigmas_audio[switch_position:]
            phases = active_plan.phases
            callbacks = (
                kwargs.get("on_low_step"),
                kwargs.get("on_resolution_lift"),
                kwargs.get("on_high_step"),
            )
            if len(phases) > 1 and any((callback is not None for callback in callbacks)):
                raise NotImplementedError(
                    "progressive trajectory observation runs on the native full clip, not the temporal streaming overlay"
                )
            low_text_token_tags = _resolve_global_text_token_tags(low_positive)
            high_text_token_tags = _resolve_global_text_token_tags(high_positive)
            if not torch.equal(low_text_token_tags, high_text_token_tags):
                raise RuntimeError("progressive low/high text token tags differ")
            text_len = int(
                _required_tensor(
                    high_positive.static_kwargs.get("prompt_embeds"),
                    "high_positive.static_kwargs['prompt_embeds']",
                ).shape[0]
            )
            media_time_origin = (
                text_len
                if self._media_time_origin is None or not self.reuse_across_requests
                else self._media_time_origin
            )
            cuda_device = torch.device("cuda")
            base_model_forward = kwargs.get("model_forward")
            pp = None
            if os.environ.get("H3_PROGRESSIVE_PP_STAGE") is not None:
                if os.environ.get("H3_PIPELINE_PREWARM") == "1" and request_index == 0:
                    from .pipeline_warmup import warm_stage

                    warm_stage_index = int(os.environ["H3_PROGRESSIVE_PP_STAGE"])
                    warm_high = warm_stage_index == 2
                    warm_positive = high_positive if warm_high else low_positive
                    (warm_h, warm_w) = (
                        (high_latent_h, high_latent_w)
                        if warm_high
                        else (low_latent_h, low_latent_w)
                    )
                    warm_rows = high_frame_rows if warm_high else low_frame_rows
                    warm_branch = _build_phase_branch(
                        loop_module=loop_module,
                        positive=warm_positive,
                        phase=phases[0],
                        latent_h=warm_h,
                        latent_w=warm_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=high_text_token_tags if warm_high else low_text_token_tags,
                        keyframe_frame_indices=(),
                        total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents,
                        media_time_origin=media_time_origin,
                        video_latent_offset=0,
                        audio_latent_offset=0,
                        phase_tail_condition=False,
                        canonical_continuation_geometry=True,
                    )
                    (warm_video, warm_audio) = _phase_initial_rows(
                        initial_video_rows=high_template
                        if warm_high
                        else low_request_video.target_initial_rows,
                        initial_audio_rows=initial_audio,
                        phase=phases[0],
                        frame_rows=warm_rows,
                        total_audio_latents=total_audio_latents,
                    )
                    warm_stage(
                        owner=self,
                        model=model,
                        branch=warm_branch,
                        phase=phases[0],
                        active_plan=active_plan,
                        video=warm_video,
                        audio=warm_audio,
                        video_sigmas=sigmas_video[warm_stage_index : warm_stage_index + 2],
                        audio_sigmas=sigmas_audio[warm_stage_index : warm_stage_index + 2],
                        original_loop=original_loop,
                        loop_module=loop_module,
                        model_forward=base_model_forward,
                        directory=os.environ["H3_PIPELINE_WARMUP_DIR"],
                    )
                from .progressive_pipeline import ProgressivePipelineStage

                pp = getattr(self, "_progressive_pipeline", None)
                if pp is None:
                    pp = self._progressive_pipeline = ProgressivePipelineStage(
                        cross_request_visual_handoff=False
                    )
                if os.environ.get("H3_PROGRESSIVE_STAGE_LOCAL_PREDICTED_KV") != "1":
                    raise ValueError(
                        "Progressive PP requires stage-local low KV and no full-prefix; an enabled teacher must be external"
                    )
                original_model_forward = base_model_forward

                def traced_model_forward(active_model, call_kwargs, step_index):
                    clean_forward = (
                        bool(call_kwargs.get("streaming_cache_only"))
                        or step_index == len(sigmas_video) - 1
                    )
                    label = "clean" if clean_forward else f"dit_step_{step_index}"
                    fused_low_clean = bool(
                        not clean_forward
                        and predicted_history is not None
                        and (step_index in predicted_history.deferred)
                    )
                    with pp.compute(
                        label,
                        request=int(request_index),
                        phase=int(phase.index),
                        fused_low_clean=fused_low_clean,
                    ):
                        if original_model_forward is None:
                            return active_model(**call_kwargs)
                        return original_model_forward(active_model, call_kwargs, step_index)

                base_model_forward = traced_model_forward
            stage_predicted_kv = True
            predicted_history = None
            if stage_predicted_kv:
                from .progressive_predicted_history import ProgressivePredictedHistory

                if self._progressive_predicted_history is None:
                    self._progressive_predicted_history = ProgressivePredictedHistory(
                        model,
                        owned_stage=None if pp is None else pp.stage,
                        reuse_across_requests=self.reuse_across_requests,
                    )
                predicted_history = self._progressive_predicted_history
                if pp is None:
                    predicted_history.fused = False
                    predicted_history.early_send = False
                predicted_history.begin_request(
                    owner=self,
                    plan=active_plan,
                    model=model,
                )
            low_early_send = predicted_history is not None and predicted_history.early_send
            if pp is None or pp.stage == 2:
                pass
            cache = self._cache
            if cache is None:
                cache = CleanAVKVCache(
                    kv_contract_from_model(model),
                )
                if self.reuse_across_requests:
                    self._cache = cache
            starting_history_tokens = cache.history_tokens
            starting_history_audio_tokens = cache.history_audio_tokens
            starting_history_video_tokens = cache.history_video_tokens
            if stage_predicted_kv:
                starting_history_tokens += sum(
                    (item.history_tokens for item in predicted_history.caches)
                )
                starting_history_audio_tokens += sum(
                    (item.history_audio_tokens for item in predicted_history.caches)
                )
                starting_history_video_tokens += sum(
                    (item.history_video_tokens for item in predicted_history.caches)
                )
            reused_kv = starting_history_tokens > 0
            current_prompt_rope_start = _current_prompt_rope_start(
                text_len=text_len,
                media_time_origin=media_time_origin,
                video_latent_offset=self._video_latent_offset,
                total_video_latents=total_video_latents,
                canonical_continuation_geometry=True,
            )
            direct_high_condition_rows = None
            direct_low_condition_rows = None
            reused_clean_video = False
            if (
                self.reuse_across_requests
                and self._clean_video_condition_rows is not None
                and (high_request_video.task == "fl2va_first_frame")
            ):
                direct_high_condition_rows = self._clean_video_condition_rows
                if list(direct_high_condition_rows.shape) != [high_frame_rows, 96]:
                    raise RuntimeError(
                        "cross-request clean video latent changed high-resolution geometry"
                    )
                direct_low_condition_rows = resize_video_state_rows(
                    direct_high_condition_rows,
                    temporal_length=1,
                    source_canvas=plan.high,
                    target_canvas=plan.low,
                )
                reused_clean_video = True

            high_video_outputs: list[Any] = []
            high_audio_outputs: list[Any] = []
            lifted_outputs: list[Any] = []
            phase_records: list[StreamingPhaseExecution] = []
            position_captures: list[dict[str, object]] = []

            def finish_low_pipeline_phase(phase, student_audio):
                (clean_video, clean_audio) = pp.finish_low_phase(
                    phase=phase,
                    request=request_index,
                    last_phase=len(phases) - 1,
                    needs_next_tail=False,
                    frame_rows=high_frame_rows,
                    student_audio=student_audio,
                )
                high_video_outputs.append(clean_video)
                high_audio_outputs.append(clean_audio)
                lifted_outputs.append(clean_video)
                phase_records.append(
                    StreamingPhaseExecution(
                        phase_index=phase.index,
                        frame_start=phase.frame_start,
                        frame_stop=phase.frame_stop,
                        video_latent_start=phase.video_latent_start,
                        video_latent_stop=phase.video_latent_stop,
                        audio_latent_start=phase.audio_latent_start,
                        audio_latent_stop=phase.audio_latent_stop,
                        denoise_forwards=1,
                        clean_forwards=predicted_history.separate_clean_count(
                            phase.index, len(phases) - 1
                        ),
                        committed_history_tokens=predicted_history.caches[pp.stage].history_tokens,
                    )
                )

            session_context = H3StreamingSession(
                active_plan,
                model=model,
                cache=cache,
                attention_backend=os.environ.get(
                    "H3_PROGRESSIVE_HIGH_ATTENTION_BACKEND", self.attention_backend
                ),
                fused_history_use_sink=True,
                fused_history_recent_blocks=self.policy.cross_chunk_kv_recent_blocks,
            )
            low49_renoise_phases: list[int] = []
            with session_context as session:
                for phase in phases:
                    skip_unused_low_prep = (
                        pp is not None
                        and pp.stage == 2
                        and (os.environ.get("H3_HIGH_SKIP_UNUSED_LOW_PREP") == "1")
                    )
                    phase_keyframe_indices = (
                        high_request_video.keyframe_frame_indices
                        if high_request_video.task != "t2va" and phase.index == 0
                        else ()
                    )
                    low_branch = (
                        None
                        if skip_unused_low_prep
                        else _build_phase_branch(
                            loop_module=loop_module,
                            positive=low_positive,
                            phase=phase,
                            latent_h=low_latent_h,
                            latent_w=low_latent_w,
                            model=model,
                            device=cuda_device,
                            text_token_tags=low_text_token_tags,
                            keyframe_frame_indices=phase_keyframe_indices,
                            total_frame_count=active_plan.native_frame_count,
                            total_video_latents=total_video_latents,
                            media_time_origin=media_time_origin,
                            video_latent_offset=self._video_latent_offset,
                            audio_latent_offset=self._audio_latent_offset,
                            phase_tail_condition=False,
                            canonical_continuation_geometry=True,
                        )
                    )
                    (low_video, phase_audio) = _phase_initial_rows(
                        initial_video_rows=low_request_video.target_initial_rows,
                        initial_audio_rows=initial_audio,
                        phase=phase,
                        frame_rows=low_frame_rows,
                        total_audio_latents=total_audio_latents,
                    )
                    low_keyframe_rows = None
                    if phase_keyframe_indices and (not skip_unused_low_prep):
                        low_condition_initial = low_request_video.condition_initial_rows
                        if low_condition_initial is None:
                            raise RuntimeError("progressive phase latent-tail condition is absent")
                        low_condition_initial = low_condition_initial.to(
                            device=low_video.device, dtype=low_video.dtype
                        )
                        low_video = torch.cat((low_condition_initial, low_video), dim=0)
                        low_keyframe_rows = (
                            direct_low_condition_rows
                            if direct_low_condition_rows is not None
                            else low_request_video.keyframe_condition_rows
                        )

                    def low_noisy_forward(
                        active_model: Any, call_kwargs: dict[str, Any], step_index: int
                    ) -> tuple[Any, Any]:
                        if pp is not None:
                            step_index += pp.stage
                        if predicted_history is not None:
                            return predicted_history.forward(
                                step_index,
                                phase.index,
                                owner=self,
                                branch=low_branch,
                                model=active_model,
                                call_kwargs=call_kwargs,
                                model_forward=base_model_forward,
                                metadata=_native_attention_metadata,
                                restore_hook=session.hook,
                                video_sigmas=low_sigmas_video,
                                audio_sigmas=low_sigmas_audio,
                                condition_timestep=1.0 if reused_clean_video or False else 0.999,
                            )
                        history_context = session.noisy_step(
                            phase.index, denoise_step_index=step_index
                        )
                        with nullcontext(), history_context, _native_attention_metadata(low_branch):
                            if base_model_forward is None:
                                return active_model(**call_kwargs)
                            return base_model_forward(active_model, call_kwargs, step_index)

                    low_on_step = None if skip_unused_low_prep else kwargs.get("on_low_step")
                    low_capture_callbacks: dict[str, Any] = {}
                    if predicted_history is not None and (not skip_unused_low_prep):

                        def capture_low_prediction(step, vv, av):
                            predicted_history.capture(
                                step,
                                vv,
                                av,
                                branch=low_branch,
                                video_sigmas=low_sigmas_video,
                                audio_sigmas=low_sigmas_audio,
                                clean_audio_callback=None,
                                step_offset=0 if pp is None else pp.stage,
                            )

                        low_capture_callbacks = {
                            "on_initial_state": predicted_history.initial,
                            "on_velocity": capture_low_prediction,
                        }
                    if pp is not None and pp.stage == 2:
                        lifted_target = pp.receive(
                            "input",
                            request_index,
                            phase.index,
                            (phase.video_latent_count * high_frame_rows, 96),
                        )
                        generated_low_audio = pp.receive(
                            "input", request_index, [phase.index, "audio"], tuple(phase_audio.shape)
                        )
                    else:
                        if pp is not None and pp.stage == 1:
                            low_video = pp.receive(
                                "input", request_index, phase.index, tuple(low_video.shape)
                            )
                            phase_audio = pp.receive(
                                "input",
                                request_index,
                                [phase.index, "audio"],
                                tuple(phase_audio.shape),
                            )
                        owned_slice = slice(None) if pp is None else slice(pp.stage, pp.stage + 2)
                        (generated_low_video, generated_low_audio) = original_loop(
                            model=model,
                            model_forward=low_noisy_forward,
                            positive=low_branch,
                            initial_video_rows=low_video,
                            initial_audio_rows=phase_audio,
                            keyframe_cond_rows=low_keyframe_rows,
                            audio_ref_rows=None,
                            sigmas_video=low_sigmas_video[owned_slice],
                            sigmas_audio=low_sigmas_audio[owned_slice],
                            device=cuda_device,
                            imgvid_cond_noise_aug_for_inference=1.0
                            if reused_clean_video or False
                            else 0.999,
                            audio_cond_noise_aug_for_inference=1.0,
                            on_step=low_on_step,
                            **low_capture_callbacks,
                        )
                        if predicted_history is not None:
                            low_history_commit = dict(
                                owner=self,
                                phase=phase.index,
                                branch=low_branch,
                                model=model,
                                restore_hook=session.hook,
                                condition_timestep=1.0 if reused_clean_video or False else 0.999,
                                model_forward=base_model_forward,
                                metadata=_native_attention_metadata,
                            )
                            if not low_early_send:
                                predicted_history.commit(**low_history_commit)
                        if pp is not None and (pp.stage == 0 or (pp.stage == 1 and pp.rank != 0)):
                            pp.send("next", generated_low_video, request_index, phase.index)
                            pp.send(
                                "next", generated_low_audio, request_index, [phase.index, "audio"]
                            )
                            if low_early_send:
                                predicted_history.commit(**low_history_commit)
                            finish_low_pipeline_phase(phase, generated_low_audio)
                            continue
                        import time as lift_clock

                        lift_started = lift_clock.time()
                        lift_function = lift_video_state_rows
                        lift_extra = {}
                        lift_args = dict(
                            temporal_length=phase.video_latent_count,
                            temporal_offset=video_transport_prefix_latents
                            + phase.video_latent_start,
                            source_canvas=plan.low,
                            target_canvas=plan.high,
                            video_sigma=float(sigmas_video[switch_position]),
                            noise_seed=int(kwargs["lift_noise_seed"]),
                            **lift_extra,
                        )
                        (lifted_target, lift_receipt) = lift_function(
                            generated_low_video[low_branch.video_target_slice], **lift_args
                        )
                    if pp is not None and pp.stage == 1:
                        pp.records.append(
                            {
                                "kind": "host_lift",
                                "key": [request_index, phase.index],
                                "start": lift_started,
                                "end": lift_clock.time(),
                                "cpu_noise_seconds": lift_receipt.get("cpu_noise_seconds"),
                                "cpu_threads": torch.get_num_threads(),
                            }
                        )
                        pp.send("next", lifted_target, request_index, phase.index)
                        pp.send("next", generated_low_audio, request_index, [phase.index, "audio"])
                        if low_early_send:
                            predicted_history.commit(**low_history_commit)
                        finish_low_pipeline_phase(phase, generated_low_audio)
                        continue
                    lifted_outputs.append(lifted_target)
                    if pp is None and kwargs.get("on_resolution_lift") is not None:
                        kwargs["on_resolution_lift"](
                            lifted_target, generated_low_audio, lift_receipt
                        )
                    high_branch = _build_phase_branch(
                        loop_module=loop_module,
                        positive=high_positive,
                        phase=phase,
                        latent_h=high_latent_h,
                        latent_w=high_latent_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=high_text_token_tags,
                        keyframe_frame_indices=phase_keyframe_indices,
                        total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents,
                        media_time_origin=media_time_origin,
                        video_latent_offset=self._video_latent_offset,
                        audio_latent_offset=self._audio_latent_offset,
                        phase_tail_condition=False,
                        canonical_continuation_geometry=True,
                    )
                    branch_capture = getattr(high_branch, "_taomate_hd_position_capture", None)
                    if branch_capture is not None:
                        if position_captures:
                            raise RuntimeError(
                                "bounded RoPE exact progressive request produced multiple high-resolution captures"
                            )
                        position_captures.append(dict(branch_capture))
                    high_video = lifted_target
                    high_keyframe_rows = None
                    if phase_keyframe_indices:
                        high_condition_initial = high_request_video.condition_initial_rows
                        if high_condition_initial is None:
                            raise RuntimeError("progressive high-resolution latent tail is absent")
                        high_condition_initial = high_condition_initial.to(
                            device=high_video.device, dtype=high_video.dtype
                        )
                        high_video = torch.cat((high_condition_initial, high_video), dim=0)
                        high_keyframe_rows = (
                            direct_high_condition_rows
                            if direct_high_condition_rows is not None
                            else high_request_video.keyframe_condition_rows
                        )

                    def high_noisy_forward(
                        active_model: Any, call_kwargs: dict[str, Any], step_index: int
                    ) -> tuple[Any, Any]:
                        history_context = session.noisy_step(
                            phase.index,
                            denoise_step_index=switch_position + step_index,
                        )
                        with history_context, _native_attention_metadata(high_branch):
                            if base_model_forward is None:
                                return active_model(**call_kwargs)
                            return base_model_forward(
                                active_model, call_kwargs, switch_position + step_index
                            )

                    high_on_step = kwargs.get("on_high_step")
                    (generated_high_video, generated_high_audio) = original_loop(
                        model=model,
                        model_forward=high_noisy_forward,
                        positive=high_branch,
                        initial_video_rows=high_video,
                        initial_audio_rows=generated_low_audio,
                        keyframe_cond_rows=high_keyframe_rows,
                        audio_ref_rows=None,
                        sigmas_video=high_sigmas_video,
                        sigmas_audio=high_sigmas_audio,
                        device=cuda_device,
                        imgvid_cond_noise_aug_for_inference=1.0
                        if reused_clean_video or False
                        else 0.999,
                        audio_cond_noise_aug_for_inference=1.0,
                        on_step=high_on_step,
                    )
                    clean_phase_video = generated_high_video[high_branch.video_target_slice]
                    if pp is not None:
                        pp.feedback(
                            clean_phase_video,
                            request=request_index,
                            phase=phase.index,
                            last_phase=len(phases) - 1,
                            frame_rows=high_frame_rows,
                            needs_next_tail=phase.index + 1 < len(phases) and False,
                        )
                        pp.publish_phase(
                            clean_phase_video, generated_high_audio, request_index, phase.index
                        )
                    history_before_commit = session.cache.history_tokens
                    current_commit_tokens = (
                        phase.video_latent_count * high_frame_rows + 2 * phase.audio_latent_count
                    )
                    expected_history_tokens = history_before_commit + current_commit_tokens
                    clean_timestep = high_branch.prepare_timestep_plan(
                        video_timesteps=[1.0],
                        audio_timesteps=[1.0],
                        imgvid_cond_noise_aug=1.0 if reused_clean_video or False else 0.999,
                        audio_ref_cond_noise_aug=1.0,
                    )[0]
                    clean_kwargs = high_branch.forward_kwargs(
                        video_rows=generated_high_video,
                        audio_rows=generated_high_audio,
                        step_timesteps=clean_timestep,
                    )
                    with (
                        session.clean_commit(phase.index),
                        _native_attention_metadata(high_branch),
                        torch.no_grad(),
                    ):
                        if base_model_forward is None:
                            model(**clean_kwargs)
                        else:
                            base_model_forward(model, clean_kwargs, len(sigmas_video) - 1)
                    if session.cache.history_tokens != expected_history_tokens:
                        raise RuntimeError(
                            "progressive clean H3 KV history differs from the high-resolution phase boundary"
                        )
                    self._apply_cross_chunk_cache_policy(session.cache)
                    high_video_outputs.append(clean_phase_video)
                    high_audio_outputs.append(generated_high_audio)
                    phase_records.append(
                        StreamingPhaseExecution(
                            phase_index=phase.index,
                            frame_start=phase.frame_start,
                            frame_stop=phase.frame_stop,
                            video_latent_start=phase.video_latent_start,
                            video_latent_stop=phase.video_latent_stop,
                            audio_latent_start=phase.audio_latent_start,
                            audio_latent_stop=phase.audio_latent_stop,
                            denoise_forwards=1
                            if pp is not None
                            else len(low_sigmas_video) - 1 + len(high_sigmas_video) - 1,
                            clean_forwards=int(not False)
                            + (
                                predicted_history.separate_clean_count(phase.index, len(phases) - 1)
                                if predicted_history is not None
                                else 0
                            ),
                            committed_history_tokens=session.cache.history_tokens + 0,
                            global_block_index=None,
                        )
                    )
            joined_video = torch.cat(high_video_outputs, dim=0)
            joined_audio = _join_phase_audio(high_audio_outputs, phases)
            joined_lifted = torch.cat(lifted_outputs, dim=0)
            if list(joined_video.shape) != [total_video_latents * high_frame_rows, 96] or list(
                joined_audio.shape
            ) != [2 * total_audio_latents, 32]:
                raise RuntimeError("reassembled progressive streaming output changed high geometry")
            returned_video = joined_video
            returned_audio = joined_audio
            if canonical_continuation:
                returned_video = torch.cat(
                    (
                        transport_video_rows.to(
                            device=joined_video.device, dtype=joined_video.dtype
                        ),
                        joined_video,
                    ),
                    dim=0,
                )
                previous_audio = transport_audio_rows.view(2, -1, int(joined_audio.shape[-1]))[
                    :, -audio_transport_prefix_latents:
                ]
                returned_audio = (
                    torch.cat(
                        (
                            previous_audio.to(device=joined_audio.device, dtype=joined_audio.dtype),
                            joined_audio.view(2, total_audio_latents, int(joined_audio.shape[-1])),
                        ),
                        dim=1,
                    )
                    .contiguous()
                    .view(-1, int(joined_audio.shape[-1]))
                )
            if self.reuse_across_requests:
                self._clean_video_condition_rows = (
                    joined_video[-high_frame_rows:]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                self._canonical_transport_video_rows = (
                    joined_video[-2 * high_frame_rows :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                audio_by_channel = joined_audio.view(
                    2, total_audio_latents, int(joined_audio.shape[-1])
                )
                self._canonical_transport_audio_rows = (
                    audio_by_channel[:, -min(9, total_audio_latents) :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .view(-1, int(joined_audio.shape[-1]))
                )
                retained_history_tokens = cache.history_tokens
                retained_history_audio_tokens = cache.history_audio_tokens
                retained_history_video_tokens = cache.history_video_tokens
                if stage_predicted_kv:
                    retained_history_tokens += sum(
                        (item.history_tokens for item in predicted_history.caches)
                    )
                    retained_history_audio_tokens += sum(
                        (item.history_audio_tokens for item in predicted_history.caches)
                    )
                    retained_history_video_tokens += sum(
                        (item.history_video_tokens for item in predicted_history.caches)
                    )
                self._media_time_origin = media_time_origin
            else:
                retained_history_tokens = 0
                retained_history_audio_tokens = 0
                retained_history_video_tokens = 0
            progressive_pipeline_receipt = None
            if stage_predicted_kv:
                progressive_pipeline_receipt = {
                    "history_mode": "stage_local_predicted_clean"
                    + ("_pipeline" if pp is not None else "_serial_reference"),
                    "stage_history_canvas": ["low", "low", "high"],
                    "low_stage_history_sources": ["predicted_clean", "predicted_clean"],
                    "low_stage_audio_history_source": "predicted_clean",
                    "high_history_source": "native_final_clean",
                    "progressive_history_kv_mode": "stage_local_clean",
                    "high_committed_blocks": cache.committed_blocks,
                    "low_retained_history_tokens": [
                        item.history_tokens for item in predicted_history.caches
                    ],
                    "high_retained_history_tokens": cache.history_tokens,
                    "low_committed_blocks": [
                        item.committed_blocks for item in predicted_history.caches
                    ],
                    "extra_low_clean_forwards_per_chunk": None
                    if predicted_history.fused
                    else len(predicted_history.owned_stages),
                    "separate_low_clean_forwards_total": predicted_history.separate_clean_forwards,
                    "fused_low_commit_denoise_forwards_total": predicted_history.fused_forwards,
                    "parallel_pipeline": pp is not None,
                    "low_tail_condition_source": "disabled",
                    "low_tail_requires_high_feedback": False,
                    "owned_stage": None if pp is None else pp.stage,
                    "owned_low_stages": list(predicted_history.owned_stages),
                    "fused_commit_denoise": predicted_history.fused,
                    "fused_static_runtime_plans": predicted_history.fused
                    and os.environ.get("H3_PROGRESSIVE_FUSED_STATIC_PLANS") == "1",
                }
            execution = StreamingExecution(
                native_frame_count=active_plan.native_frame_count,
                attention_backend_receipt=session.hook.attention_backend_receipt()
                if session is not None and self.attention_backend == "sage_attn"
                else None,
                phase_executions=tuple(phase_records),
                request_task=high_request_video.task,
                noisy_kv_writes=0,
                video_latent_offset=self._video_latent_offset,
                audio_latent_offset=self._audio_latent_offset,
                starting_history_tokens=starting_history_tokens,
                retained_history_tokens=retained_history_tokens,
                cross_request_kv_reused=reused_kv,
                cross_request_kv_boundary_reset=False,
                cross_request_kv_boundary_reset_tokens=0,
                cross_request_clean_video_latent_handoff=reused_clean_video,
                streaming_policy=self.policy.to_dict(),
                request_index=request_index,
                current_prompt_rope_start=current_prompt_rope_start,
                starting_history_audio_tokens=starting_history_audio_tokens,
                starting_history_video_tokens=starting_history_video_tokens,
                retained_history_audio_tokens=retained_history_audio_tokens,
                retained_history_video_tokens=retained_history_video_tokens,
                canonical_continuation_geometry=True,
                native_frame_offset=self._native_frame_offset,
                published_native_frames=active_plan.native_frame_count,
                video_transport_prefix_latents=video_transport_prefix_latents,
                audio_transport_prefix_latents_per_channel=audio_transport_prefix_latents,
                published_video_latents=total_video_latents,
                published_audio_latents_per_channel=total_audio_latents,
                text_time_start=current_prompt_rope_start,
                fixed_media_time_origin=media_time_origin,
                reuse_across_requests=self.reuse_across_requests,
                position_capture=_merge_packed_grid_position_captures(position_captures),
                progressive_pipeline=progressive_pipeline_receipt,
            )
            self.last_execution = execution
            self.executions.append(execution)
            if self.reuse_across_requests:
                self._video_latent_offset += total_video_latents
                self._audio_latent_offset += total_audio_latents
                self._native_frame_offset += active_plan.native_frame_count
                self._request_index += 1
            if high_request_video.condition_initial_rows is not None:
                returned_condition_rows = (
                    direct_high_condition_rows
                    if direct_high_condition_rows is not None
                    else high_request_video.keyframe_condition_rows
                )
                returned_video = torch.cat(
                    (
                        returned_condition_rows.to(
                            device=returned_video.device, dtype=returned_video.dtype
                        ),
                        returned_video,
                    ),
                    dim=0,
                )
            if list(returned_video.shape) != list(high_template.shape):
                raise RuntimeError(
                    "progressive streaming video output no longer matches raw request ABI"
                )
            if list(returned_audio.shape) != [2 * official_audio_latents, 32]:
                raise RuntimeError(
                    "progressive streaming audio output no longer matches raw request ABI"
                )
            return result_type(
                video_rows=returned_video,
                audio_rows=returned_audio,
                lifted_video_target_rows=joined_lifted,
                lift_receipt={
                    "schema": "taomate_hd.progressive_resolution_lift.streaming.v1",
                    "phase_count": len(phases),
                    "video_sigma": float(sigmas_video[switch_position]),
                    "noise_seed": int(kwargs["lift_noise_seed"]),
                    "temporal_offsets": [
                        video_transport_prefix_latents + phase.video_latent_start
                        for phase in phases
                    ],
                    "low49_renoise33_phase_indices": low49_renoise_phases,
                },
                high_target_source="continued_from_lifted_low_state",
                low_model_forwards=switch_position,
                high_model_forwards=len(sigmas_video) - switch_position - 1,
            )
        finally:
            self._running = False

    @contextmanager
    def install(self):
        import taomate_hd.model.pipeline as pipeline
        import taomate_hd.model.denoise as loop
        from taomate_hd.model.progressive_denoise import H3ProgressiveDenoiseResult

        original = pipeline.minimax_h3_progressive_denoise_loop

        def progressive(**kwargs):
            return self._run_progressive(
                loop.minimax_h3_denoise_loop, loop, H3ProgressiveDenoiseResult, **kwargs
            )

        pipeline.minimax_h3_progressive_denoise_loop = progressive
        try:
            yield self
        finally:
            pipeline.minimax_h3_progressive_denoise_loop = original


@contextmanager
def install_streaming_runtime(
    *,
    plan: StreamPlan,
    server_args: Any,
    reuse_across_requests: bool = False,
    policy: StreamingInferencePolicy | None = None,
    attention_backend: str = "sdpa",
) -> Iterator[H3StreamingRuntimeOverlay]:
    overlay = H3StreamingRuntimeOverlay(
        plan=plan,
        server_args=server_args,
        reuse_across_requests=reuse_across_requests,
        policy=policy,
        attention_backend=attention_backend,
    )
    with overlay.install():
        yield overlay
