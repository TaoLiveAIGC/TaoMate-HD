# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from __future__ import annotations
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
from .cache import CleanAVKVCache


class HookMode(str, Enum):
    IDLE = "idle"
    NOISY = "noisy"
    CLEAN_COMMIT = "clean_commit"
    FUSED_PROGRESS_DENOISE = "fused_progress_denoise"


@dataclass(frozen=True)
class _KVPart:
    key: Any
    value: Any
    cached_pair: Any | None = None
    transfer_tracker: Any | None = None


@dataclass(frozen=True)
class _LiveDocumentKVPlan:
    condition: tuple[_KVPart, ...]
    media: tuple[_KVPart, ...]


def _plan_live_document_kv(
    *,
    condition_key: Any,
    condition_value: Any,
    allowed_prefixes: tuple[Any, ...],
    current_media_key: Any,
    current_media_value: Any,
    previous_media_key: Any | None = None,
    previous_media_value: Any | None = None,
) -> _LiveDocumentKVPlan:
    """Plan the same condition/media visibility used by Stage2/3 training."""
    condition = _KVPart(condition_key, condition_value)
    media = [
        condition,
        *(
            pair if isinstance(pair, _KVPart) else _KVPart(pair.key, pair.value, cached_pair=pair)
            for pair in allowed_prefixes
        ),
    ]
    if previous_media_key is not None:
        media.append(_KVPart(previous_media_key, previous_media_value))
    media.append(_KVPart(current_media_key, current_media_value))
    return _LiveDocumentKVPlan(condition=(condition,), media=tuple(media))


def _materialize_dense_kv_parts(
    torch: Any, *, parts: tuple[_KVPart, ...], device: Any
) -> tuple[Any, Any]:
    """Materialize one final dense K/V buffer directly from ordered parts."""
    if device is None:
        return (
            parts[0].key
            if len(parts) == 1
            else torch.cat(tuple((part.key for part in parts)), dim=0),
            parts[0].value
            if len(parts) == 1
            else torch.cat(tuple((part.value for part in parts)), dim=0),
        )
    if len(parts) == 1 and parts[0].key.device == device:
        return (parts[0].key, parts[0].value)
    total_tokens = sum((int(part.key.shape[0]) for part in parts))
    key = torch.empty(
        (total_tokens, *parts[0].key.shape[1:]), dtype=parts[0].key.dtype, device=device
    )
    value = torch.empty_like(key)
    offset = 0
    trackers: dict[int, Any] = {}
    for part in parts:
        count = int(part.key.shape[0])
        if part.key.device.type == "cuda" and part.key.device != device:
            raise RuntimeError("streaming K/V part is resident on another CUDA rank")
        key.narrow(0, offset, count).copy_(part.key)
        value.narrow(0, offset, count).copy_(part.value)
        if part.transfer_tracker is not None:
            trackers[id(part.transfer_tracker)] = part.transfer_tracker
            if part.key.device.type == "cpu":
                part.transfer_tracker.record_h2d_bytes(
                    int(part.key.numel()) * int(part.key.element_size())
                    + int(part.value.numel()) * int(part.value.element_size())
                )
        offset += count
    materialized_bytes = int(key.numel()) * int(key.element_size()) + int(value.numel()) * int(
        value.element_size()
    )
    for tracker in trackers.values():
        tracker.record_dense_attention_temp_bytes(materialized_bytes)
    return (key, value)


def _sdpa_from_kv_parts(
    torch: Any, functional: Any, *, query: Any, parts: tuple[_KVPart, ...], scale: float
) -> Any:
    """Execute exact SDPA over a reusable ordered K/V plan."""
    (key, value) = _materialize_dense_kv_parts(
        torch, parts=parts, device=getattr(query, "device", None)
    )
    output = functional.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)


def _live_documents_from_mask_or_hint(
    *, boundaries: tuple[int, ...], commit_mask: Any, live_documents_hint: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Resolve target documents without reading CUDA state when given a hint."""
    if live_documents_hint is None:
        return tuple(
            (
                index
                for (index, (start, stop)) in enumerate(zip(boundaries, boundaries[1:]))
                if bool(commit_mask[start:stop].any().item())
            )
        )
    if not isinstance(live_documents_hint, tuple):
        raise TypeError("live document hint must be a tuple")
    document_count = len(boundaries) - 1
    if any(
        (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or (index >= document_count)
            for index in live_documents_hint
        )
    ):
        raise ValueError("live document hint contains an invalid document index")
    if tuple(sorted(set(live_documents_hint))) != live_documents_hint:
        raise ValueError("live document hint must be sorted and unique")
    if any((boundaries[index] == boundaries[index + 1] for index in live_documents_hint)):
        raise ValueError("live document hint references an empty document")
    return live_documents_hint


class H3StreamingAttentionHook:
    """Consumes official post-QK-norm/post-RoPE/post-Ulysses Q/K/V tensors."""

    def __init__(self, cache: CleanAVKVCache, *, attention_backend: str = "sdpa") -> None:
        self.cache = cache
        self.attention_backend = attention_backend
        self._sage_attention_sm90 = None
        if attention_backend == "sage_attn":
            from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90

            self._sage_attention_sm90 = sageattn_qk_int8_pv_fp8_cuda_sm90
        self.mode = HookMode.IDLE
        self.block_index: int | None = None
        self.fused_current_use_sink: bool | None = None
        self.fused_current_recent_blocks: int | None = None
        self._denoise_step_index: int | None = None
        self._live_documents: tuple[int, ...] | None = None
        self._live_document_boundaries: tuple[int, ...] | None = None
        self._fused_previous_media_indices: Any | None = None
        self._live_document_rows: dict[int, tuple[Any, Any]] = {}
        self.sage_attn_calls = 0
        self.sage_attn_query_tokens = 0
        self.sage_attn_key_tokens = 0

    def sage_attn_receipt(self) -> dict[str, Any] | None:
        if self.attention_backend != "sage_attn":
            return None
        receipt = {
            "backend": "sageattn_qk_int8_pv_fp8_cuda_sm90",
            "tensor_layout": "NHD",
            "is_causal": False,
            "qk_quant_granularity": "per_thread",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": True,
            "kernel_calls": self.sage_attn_calls,
            "query_tokens": self.sage_attn_query_tokens,
            "key_tokens": self.sage_attn_key_tokens,
        }
        return receipt

    def attention_backend_receipt(self) -> dict[str, Any] | None:
        return self.sage_attn_receipt()

    def activate(
        self,
        *,
        mode: HookMode,
        block_index: int,
        fused_current_use_sink: bool | None = None,
        fused_current_recent_blocks: int | None = None,
        denoise_step_index: int | None = None,
    ) -> None:
        if self.mode is not HookMode.IDLE:
            raise RuntimeError("H3 streaming attention hook is already active")
        if block_index != self.cache.committed_blocks:
            raise ValueError("streaming attention block differs from clean history")
        if (
            fused_current_use_sink is not None or fused_current_recent_blocks is not None
        ) and mode is not HookMode.FUSED_PROGRESS_DENOISE:
            raise ValueError("fused current history policy requires fused progress-denoise mode")
        if denoise_step_index is not None and mode not in {
            HookMode.NOISY,
            HookMode.FUSED_PROGRESS_DENOISE,
        }:
            raise ValueError("denoise step index requires noisy or fused progress-denoise mode")
        self.mode = mode
        self.block_index = block_index
        self.fused_current_use_sink = fused_current_use_sink
        self.fused_current_recent_blocks = fused_current_recent_blocks
        self._denoise_step_index = denoise_step_index
        self._live_documents = None
        self._live_document_boundaries = None
        self._fused_previous_media_indices = None
        self._live_document_rows = {}

    def deactivate(self) -> None:
        self.mode = HookMode.IDLE
        self.block_index = None
        self.fused_current_use_sink = None
        self.fused_current_recent_blocks = None
        self._denoise_step_index = None
        self._live_documents = None
        self._live_document_boundaries = None
        self._fused_previous_media_indices = None
        self._live_document_rows = {}

    def __call__(
        self,
        *,
        attention: Any,
        layer_name: str,
        query: Any,
        key: Any,
        value: Any,
        token_tags: Any,
        commit_mask: Any,
        cu_seqlens: Any,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        live_documents_hint: tuple[int, ...] | None = None,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        del max_seqlen
        if self.mode is HookMode.IDLE:
            raise RuntimeError("patched H3 attention called outside a streaming step")
        if layer_name not in self.cache.contract.layer_names:
            raise ValueError(f"streaming hook reached a non-main attention layer: {layer_name}")
        if any(
            (not torch.is_tensor(item) for item in (query, key, value, token_tags, commit_mask))
        ):
            raise TypeError("H3 streaming Q/K/V and token metadata must be tensors")
        expected_tail = (self.cache.contract.local_heads, self.cache.contract.head_dim)
        if (
            query.ndim != 3
            or key.shape != query.shape
            or value.shape != query.shape
            or (tuple(query.shape[1:]) != expected_tail)
        ):
            raise ValueError(
                f"post-Ulysses Q/K/V must be [tokens, {expected_tail[0]}, {expected_tail[1]}]"
            )
        if query.dtype != torch.bfloat16 or not query.is_cuda:
            raise ValueError("official H3 streaming attention requires CUDA bfloat16 Q/K/V")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("H3 streaming Q/K/V dtypes differ")
        if key.device != query.device or value.device != query.device:
            raise ValueError("H3 streaming Q/K/V devices differ")
        if token_tags.ndim != 1 or token_tags.shape[0] != query.shape[0]:
            raise ValueError("post-Ulysses token tags do not cover Q/K/V rows")
        if (
            commit_mask.ndim != 1
            or commit_mask.dtype != torch.bool
            or commit_mask.shape[0] != query.shape[0]
        ):
            raise ValueError("post-Ulysses clean commit mask does not cover Q/K/V rows")
        if cu_seqlens_host is None:
            boundaries = tuple((int(item) for item in cu_seqlens.tolist()))
        else:
            boundaries = tuple((int(item) for item in cu_seqlens_host))
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != query.shape[0]
            or any((left > right for (left, right) in zip(boundaries, boundaries[1:])))
        ):
            raise ValueError("invalid packed H3 sequence boundaries")
        if self._live_documents is None:
            live_documents = _live_documents_from_mask_or_hint(
                boundaries=boundaries,
                commit_mask=commit_mask,
                live_documents_hint=live_documents_hint,
            )
            self._live_documents = live_documents
            self._live_document_boundaries = boundaries
        else:
            if boundaries != self._live_document_boundaries:
                raise ValueError("packed H3 boundaries changed within one streaming step")
            live_documents = self._live_documents
            if live_documents_hint is not None and live_documents_hint != live_documents:
                raise ValueError("live document hint changed within one streaming step")
        fused = self.mode is HookMode.FUSED_PROGRESS_DENOISE
        expected_live_documents = 2 if fused else 1
        if len(live_documents) != expected_live_documents:
            raise ValueError(
                f"streaming session target-document count differs from hook mode: expected={expected_live_documents}, actual={len(live_documents)}"
            )
        live_document = live_documents[-1]
        previous_live_document = live_documents[0] if fused else None
        persistent = self.cache.history(layer_name)
        outputs: list[Any] = []
        from taomate_hd.model.condition_cache import compact_condition_attention_output

        compact_output = compact_condition_attention_output(layer_name, query)

        def append_document(start: int, stop: int, output: Any) -> None:
            if compact_output is None:
                outputs.append(output)
            else:
                compact_output.write_document(start, stop, output)

        scale = float(attention.softmax_scale)

        def exact_attention(q_segment: Any, parts: tuple[_KVPart, ...]) -> Any:
            if self.attention_backend == "sage_attn":
                key_segment = (
                    parts[0].key
                    if len(parts) == 1
                    else torch.cat(tuple(part.key for part in parts), dim=0)
                )
                value_segment = (
                    parts[0].value
                    if len(parts) == 1
                    else torch.cat(tuple(part.value for part in parts), dim=0)
                )
                if os.environ.get("H3_SAGE_FUSED_VALUE_PADDING") == "1":
                    from ..inference.sage_value_padding import sage_with_fused_value_padding

                    output = sage_with_fused_value_padding(
                        q_segment.unsqueeze(0),
                        key_segment.unsqueeze(0),
                        value_segment.unsqueeze(0),
                        sm_scale=scale,
                    )
                else:
                    output = self._sage_attention_sm90(
                        q_segment.unsqueeze(0),
                        key_segment.unsqueeze(0),
                        value_segment.unsqueeze(0),
                        tensor_layout="NHD",
                        is_causal=False,
                        qk_quant_gran="per_thread",
                        sm_scale=scale,
                        pv_accum_dtype="fp32+fp32",
                        smooth_k=True,
                        return_lse=False,
                    )
                self.sage_attn_calls += 1
                self.sage_attn_query_tokens += int(q_segment.shape[0])
                self.sage_attn_key_tokens += int(key_segment.shape[0])
                return output.squeeze(0)
            return _sdpa_from_kv_parts(torch, functional, query=q_segment, parts=parts, scale=scale)

        reuse_previous_media = (
            fused and os.environ.get("H3_PROGRESSIVE_LOW_REUSE_PREVIOUS_KV") == "1"
        )
        previous_document_media: tuple[Any, Any] | None = None
        for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            if start == stop:
                continue
            q_segment = query[start:stop]
            segment_key = key[start:stop]
            segment_value = value[start:stop]
            if index not in live_documents:
                append_document(
                    start,
                    stop,
                    exact_attention(q_segment, (_KVPart(segment_key, segment_value),)),
                )
                continue
            rows = self._live_document_rows.get(index)
            if rows is None:
                commit_mask_segment = commit_mask[start:stop]
                condition_indices = torch.nonzero(~commit_mask_segment, as_tuple=False).view(-1)
                media_indices = torch.nonzero(commit_mask_segment, as_tuple=False).view(-1)
                rows = (condition_indices, media_indices)
                self._live_document_rows[index] = rows
            (condition_indices, media_indices) = rows
            current_media_indices = media_indices
            condition_query = q_segment.index_select(0, condition_indices)
            condition_key = segment_key.index_select(0, condition_indices)
            condition_value = segment_value.index_select(0, condition_indices)
            media_query = q_segment.index_select(0, media_indices)
            current_media_key = segment_key.index_select(0, current_media_indices)
            current_media_value = segment_value.index_select(0, current_media_indices)
            if reuse_previous_media and index == previous_live_document:
                previous_document_media = (current_media_key, current_media_value)
            previous_media_key = None
            previous_media_value = None
            if fused and index == live_document:
                assert previous_live_document is not None
                current_persistent = self.cache.history_for_fused_current(
                    layer_name,
                    **{}
                    if self.fused_current_recent_blocks is None
                    else {
                        "use_sink": bool(self.fused_current_use_sink),
                        "recent_blocks": self.fused_current_recent_blocks,
                    },
                )
                prefixes = (*(() if current_persistent is None else (current_persistent,)),)
                previous_start = boundaries[previous_live_document]
                previous_stop = boundaries[previous_live_document + 1]
                if self._fused_previous_media_indices is None:
                    previous_commit_mask = commit_mask[previous_start:previous_stop]
                    previous_media = torch.nonzero(previous_commit_mask, as_tuple=False).view(-1)
                    self._fused_previous_media_indices = previous_media.add(previous_start)
                previous_media_indices = self._fused_previous_media_indices
                if previous_document_media is not None:
                    (previous_media_key, previous_media_value) = previous_document_media
                else:
                    previous_media_key = key.index_select(0, previous_media_indices)
                    previous_media_value = value.index_select(0, previous_media_indices)
            else:
                prefixes = (persistent,) if persistent is not None else ()
            plan = _plan_live_document_kv(
                condition_key=condition_key,
                condition_value=condition_value,
                allowed_prefixes=prefixes,
                previous_media_key=previous_media_key,
                previous_media_value=previous_media_value,
                current_media_key=current_media_key,
                current_media_value=current_media_value,
            )
            from taomate_hd.model.condition_cache import condition_attention_cached

            condition_cached = condition_attention_cached(layer_name, index)
            if condition_cached:
                condition_output = (
                    None if compact_output is not None else torch.zeros_like(condition_query)
                )
            else:
                condition_output = exact_attention(condition_query, plan.condition)
            media_output = exact_attention(media_query, plan.media)
            if compact_output is None:
                document_output = torch.zeros_like(q_segment)
                document_output.index_copy_(0, condition_indices, condition_output)
                document_output.index_copy_(0, media_indices, media_output)
                outputs.append(document_output)
            else:
                if not condition_cached:
                    compact_output.write_rows(
                        index, "condition", start, condition_indices, condition_output
                    )
                compact_output.write_rows(index, "media", start, media_indices, media_output)
        merged = torch.cat(outputs, dim=0) if compact_output is None else compact_output.output
        if compact_output is None and merged.shape != query.shape:
            raise RuntimeError("streaming attention output shape differs from query")
        if self.mode is HookMode.CLEAN_COMMIT:
            self.cache.stage(layer_name, key, value, token_tags, commit_mask)
        elif fused:
            assert previous_live_document is not None
            if self._fused_previous_media_indices is None:
                raise RuntimeError("fused H3 media indices were not resolved")
            self.cache.stage(
                layer_name,
                key,
                value,
                token_tags,
                commit_mask,
                commit_indices=self._fused_previous_media_indices,
            )
        return merged
