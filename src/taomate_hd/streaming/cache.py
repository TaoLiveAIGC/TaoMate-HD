# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

MAIN_LAYER_NAMES = tuple((f"blocks.{index}.attn" for index in range(50)))
VIDEO_TOKEN_TAG = 0
AUDIO_TOKEN_TAG = 2
AV_TOKEN_TAGS = frozenset({VIDEO_TOKEN_TAG, AUDIO_TOKEN_TAG})


@dataclass(frozen=True)
class KVContract:
    layer_names: tuple[str, ...] = MAIN_LAYER_NAMES
    local_heads: int = 14
    head_dim: int = 128
    dtype: str = "torch.bfloat16"


def kv_contract_from_model(model: Any) -> KVContract:
    """Resolve post-Ulysses local K/V heads from the live H3 topology."""
    arch = getattr(model, "arch", None)
    context = getattr(model, "parallel_context", None)
    total_heads = int(getattr(arch, "num_attention_heads", 0))
    head_dim = int(getattr(arch, "attention_head_dim", 0))
    tp_size = int(getattr(context, "tp_world_size", 0))
    ulysses_size = int(getattr(context, "ulysses_world_size", 0))
    parallel_size = tp_size * ulysses_size
    if total_heads <= 0 or head_dim <= 0 or parallel_size <= 0 or total_heads % tp_size:
        raise ValueError("live H3 attention topology cannot resolve local K/V heads")
    return KVContract(
        local_heads=(total_heads // tp_size + ulysses_size - 1) // ulysses_size, head_dim=head_dim
    )


@dataclass(frozen=True)
class AVKV:
    key: Any
    value: Any
    modality: str = "audio_video"


def _validate_av_pair(pair: AVKV, contract: KVContract, *, layer: str) -> None:
    import torch

    if not isinstance(pair, AVKV) or pair.modality != "audio_video":
        raise TypeError(f"{layer} persistent prefix must be an audio-video AVKV")
    (key, value) = (pair.key, pair.value)
    if not torch.is_tensor(key) or not torch.is_tensor(value):
        raise TypeError(f"{layer} K/V must be tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError(f"{layer} K/V must have the same [tokens, heads, dim] shape")
    if tuple(key.shape[1:]) != (contract.local_heads, contract.head_dim):
        raise ValueError(
            f"{layer} K/V local shape must be [{contract.local_heads}, {contract.head_dim}]"
        )
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError(f"{layer} K/V must be bfloat16")
    if not key.is_cuda or not value.is_cuda or key.device != value.device:
        raise ValueError(f"{layer} K/V must share one CUDA device")


def _validate_committed_media_tags(tags: tuple[int, ...]) -> None:
    if not tags or frozenset(tags) != AV_TOKEN_TAGS:
        raise ValueError("clean H3 commit must contain both video and audio rows")


class CleanAVKVCache:
    """Persistent post-RoPE media history committed only from clean AV states."""

    def __init__(
        self,
        contract: KVContract | None = None,
    ) -> None:
        self.contract = contract or KVContract()
        self._history: dict[str, AVKV] = {}
        self._staged: dict[str, AVKV] | None = None
        self._staged_commit_indices: Any | None = None
        self._staged_commit_tags: tuple[int, ...] | None = None
        self._block_index = 0
        self._commit_token_counts: list[int] = []
        self._commit_token_tags: list[tuple[int, ...]] = []
        self._sink_video_only = False

    @property
    def committed_blocks(self) -> int:
        return self._block_index

    @property
    def history_tokens(self) -> int:
        return sum(self._commit_token_counts)

    @property
    def retained_blocks(self) -> int:
        return len(self._commit_token_counts)

    @property
    def history_audio_tokens(self) -> int:
        return sum((tag == AUDIO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags))

    @property
    def history_video_tokens(self) -> int:
        return sum((tag == VIDEO_TOKEN_TAG for tags in self._commit_token_tags for tag in tags))

    @property
    def clean_commit_active(self) -> bool:
        return self._staged is not None

    def history(self, layer: str) -> AVKV | None:
        """Return the complete retained causal prefix for attention.

        The default long-video policy retains a video-only block-zero sink plus
        the two most recent clean AV commits. Newly completed phases enter the
        same bounded prefix, so history does not grow with request count.
        """
        if layer not in self.contract.layer_names:
            raise KeyError(f"unknown H3 attention layer: {layer}")
        pair = self._history.get(layer)
        if pair is None:
            return None
        if not self._commit_token_counts:
            raise RuntimeError("clean H3 history has no commit boundary ledger")
        return pair

    def recent_history(self, layer: str, count: int) -> AVKV | None:
        """Return a suffix view containing at most ``count`` committed blocks."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("recent H3 history count must be non-negative")
        pair = self.history(layer)
        if pair is None or count == 0:
            return None
        if count >= len(self._commit_token_counts):
            return pair
        tokens = sum(self._commit_token_counts[-count:])
        return AVKV(key=pair.key[-tokens:], value=pair.value[-tokens:])

    def sink_and_recent_history(self, layer: str, recent_count: int) -> AVKV | None:
        """Return block-zero sink plus the newest non-sink commits."""
        import torch

        selected = self.sink_and_recent_history_blocks(layer, recent_count)
        if not selected:
            return None
        if len(selected) == 1:
            return selected[0]
        return AVKV(
            key=torch.cat(tuple((pair.key for pair in selected)), dim=0),
            value=torch.cat(tuple((pair.value for pair in selected)), dim=0),
        )

    def sink_and_recent_history_blocks(self, layer: str, recent_count: int) -> tuple[AVKV, ...]:
        """Return block-zero plus ``recent_count`` newest block views."""
        if isinstance(recent_count, bool) or not isinstance(recent_count, int) or recent_count < 0:
            raise ValueError("recent H3 history count must be non-negative")
        blocks = self.history_blocks(layer)
        if not blocks:
            return ()
        recent_start = max(1, len(blocks) - recent_count)
        return (blocks[0], *blocks[recent_start:])

    def history_blocks(self, layer: str) -> tuple[AVKV, ...]:
        """Return retained history as zero-copy per-block views."""
        if layer not in self.contract.layer_names:
            raise KeyError(f"unknown H3 attention layer: {layer}")
        pair = self.history(layer)
        if pair is None:
            return ()
        blocks: list[AVKV] = []
        offset = 0
        for count in self._commit_token_counts:
            blocks.append(
                AVKV(
                    key=pair.key.narrow(0, offset, count), value=pair.value.narrow(0, offset, count)
                )
            )
            offset += count
        return tuple(blocks)

    def history_for_fused_current(
        self, layer: str, *, use_sink: bool = False, recent_blocks: int | None = None
    ) -> AVKV | None:
        """Return the persistent part of a hypothetical post-commit view.

        The packed previous document supplies the newest clean block directly.
        This selector therefore reserves one recent slot without mutating the
        cache before the fused transformer forward succeeds.
        """
        if recent_blocks is None:
            return self.history(layer)
        persistent_recent = max(0, recent_blocks - 1)
        if use_sink:
            pair = self.sink_and_recent_history(layer, persistent_recent)
            if (
                pair is not None
                and self._sink_video_only
                and (self.retained_blocks + 1 > recent_blocks)
                and (AUDIO_TOKEN_TAG in self._commit_token_tags[0])
            ):
                import torch

                sink_tags = self._commit_token_tags[0]
                rows = [row for (row, tag) in enumerate(sink_tags) if tag == VIDEO_TOKEN_TAG]
                rows.extend(range(len(sink_tags), int(pair.key.shape[0])))
                indices = torch.tensor(rows, dtype=torch.long, device=pair.key.device)
                return self._select_pair_rows(pair, indices)
            return pair
        return self.recent_history(layer, persistent_recent)

    def begin_clean_commit(self, block_index: int) -> None:
        if block_index != self._block_index:
            raise ValueError(
                f"clean block order mismatch: expected {self._block_index}, got {block_index}"
            )
        if self._staged is not None:
            raise RuntimeError("a clean commit is already active")
        self._staged = {}
        self._staged_commit_indices = None
        self._staged_commit_tags = None

    def stage(
        self,
        layer: str,
        key: Any,
        value: Any,
        token_tags: Any,
        commit_mask: Any,
        *,
        commit_indices: Any | None = None,
    ) -> None:
        import torch

        if self._staged is None:
            raise RuntimeError("K/V may be staged only during a clean sigma=0 forward")
        if layer not in self.contract.layer_names or layer in self._staged:
            raise ValueError(f"invalid or duplicate clean K/V layer: {layer}")
        if (
            not torch.is_tensor(token_tags)
            or not torch.is_tensor(commit_mask)
            or token_tags.ndim != 1
            or (commit_mask.ndim != 1)
            or (commit_mask.dtype != torch.bool)
        ):
            raise TypeError("token_tags and commit_mask must be one-dimensional tensors")
        if (
            key.shape[0] != token_tags.shape[0]
            or value.shape[0] != token_tags.shape[0]
            or commit_mask.shape[0] != token_tags.shape[0]
        ):
            raise ValueError("token metadata and post-Ulysses K/V rows differ")
        if self._staged_commit_indices is None:
            if commit_indices is None:
                commit_indices = torch.nonzero(commit_mask, as_tuple=False).view(-1)
            if (
                not torch.is_tensor(commit_indices)
                or commit_indices.ndim != 1
                or commit_indices.dtype != torch.long
                or (commit_indices.device != key.device)
                or (int(commit_indices.numel()) == 0)
            ):
                raise ValueError("clean H3 block contains no valid audio/video rows")
            committed_tags = tuple(
                (
                    int(item)
                    for item in token_tags.index_select(0, commit_indices).detach().cpu().tolist()
                )
            )
            _validate_committed_media_tags(committed_tags)
            self._staged_commit_indices = commit_indices.detach()
            self._staged_commit_tags = committed_tags
        indices = self._staged_commit_indices
        if indices is None:
            raise RuntimeError("clean H3 commit indices were not initialized")
        selected_key = key.index_select(0, indices).detach()
        selected_value = value.index_select(0, indices).detach()
        pair = AVKV(key=selected_key, value=selected_value)
        _validate_av_pair(pair, self.contract, layer=layer)
        self._staged[layer] = pair

    def commit(self) -> None:
        import torch

        if self._staged is None:
            raise RuntimeError("no clean commit is active")
        missing = [name for name in self.contract.layer_names if name not in self._staged]
        if missing:
            raise RuntimeError(f"clean commit is missing attention layers: {missing}")
        for name in self.contract.layer_names:
            current = self._staged.pop(name)
            previous = self._history.pop(name, None)
            if previous is None:
                updated = current
            else:
                updated = AVKV(
                    key=torch.cat((previous.key, current.key), dim=0),
                    value=torch.cat((previous.value, current.value), dim=0),
                )
            self._history[name] = updated
        token_counts = {int(pair.key.shape[0]) for pair in self._history.values()}
        if len(token_counts) != 1:
            raise RuntimeError("per-layer H3 history token counts diverged")
        previous_tokens = self.history_tokens
        current_tokens = token_counts.pop() - previous_tokens
        if current_tokens <= 0:
            raise RuntimeError("clean H3 commit did not append media tokens")
        self._finish_commit(current_tokens)

    def _finish_commit(self, current_tokens: int) -> None:
        self._staged = None
        self._staged_commit_indices = None
        staged_tags = self._staged_commit_tags
        self._staged_commit_tags = None
        if staged_tags is None:
            raise RuntimeError("clean H3 commit tags were not initialized")
        self._block_index += 1
        self._commit_token_counts.append(current_tokens)
        self._commit_token_tags.append(staged_tags)

    def retain_sink_and_recent_commits(
        self, recent_count: int, *, sink_video_only: bool = True
    ) -> None:
        """Keep block-zero sink plus the newest complete AV blocks."""
        if isinstance(recent_count, bool) or not isinstance(recent_count, int) or recent_count < 0:
            raise ValueError("recent clean commit count must be non-negative")
        if self._staged is not None:
            raise RuntimeError("cannot evict H3 history during a clean commit")
        self._sink_video_only = bool(sink_video_only)
        block_count = len(self._commit_token_counts)
        if block_count == 0:
            return
        recent_start = max(0, block_count - recent_count)
        selections = [
            (0, bool(sink_video_only and recent_start > 0)),
            *((index, False) for index in range(max(1, recent_start), block_count)),
        ]
        if selections == [(index, False) for index in range(block_count)]:
            return
        self._retain_commit_rows(selections)

    @staticmethod
    def _select_pair_rows(pair: AVKV, indices: Any) -> AVKV:
        key = pair.key.index_select(0, indices)
        value = pair.value.index_select(0, indices)
        return AVKV(key=key, value=value)

    def _retain_commit_rows(self, selections: list[tuple[int, bool]]) -> None:
        """Retain selected commits, optionally filtering each to video rows."""
        import torch

        original_counts = self._commit_token_counts
        original_tags = self._commit_token_tags
        retained_tags = [
            tuple((tag for tag in original_tags[index] if tag == VIDEO_TOKEN_TAG))
            if video_only
            else original_tags[index]
            for (index, video_only) in selections
        ]
        retained_counts = [len(tags) for tags in retained_tags]
        offsets: list[int] = []
        cursor = 0
        for count in original_counts:
            offsets.append(cursor)
            cursor += count
        selected_rows = [
            offsets[index] + row
            for (index, video_only) in selections
            for (row, tag) in enumerate(original_tags[index])
            if not video_only or tag == VIDEO_TOKEN_TAG
        ]
        indices_by_device: dict[Any, Any] = {}
        for name in self.contract.layer_names:
            pair = self._history.pop(name)
            indices = indices_by_device.get(pair.key.device)
            if indices is None:
                indices = torch.tensor(selected_rows, dtype=torch.long, device=pair.key.device)
                indices_by_device[pair.key.device] = indices
            self._history[name] = self._select_pair_rows(pair, indices)
        self._commit_token_counts = retained_counts
        self._commit_token_tags = retained_tags

    def rollback(self) -> None:
        self._staged = None
        self._staged_commit_indices = None
        self._staged_commit_tags = None

    def clear(self) -> None:
        if self._staged is not None:
            raise RuntimeError("cannot clear H3 history during a clean commit")
        self._history.clear()
        self._block_index = 0
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._sink_video_only = False
