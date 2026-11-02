# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Local H3 streaming attention hook and capability information."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

STREAM_HOOK_ABI = 3
_OFFICIAL_ATTENTION_HOOK: Any | None = None


@dataclass(frozen=True)
class StreamingCapability:
    backend: str
    hook_abi: int
    hook_setter: str
    layer_count: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def streaming_capability(*, model: Any | None = None) -> StreamingCapability:
    from ..model.layers import set_native_stream_attention_hook

    return StreamingCapability(
        backend="taomate_hd_native_torch",
        hook_abi=STREAM_HOOK_ABI,
        hook_setter=f"{set_native_stream_attention_hook.__module__}.{set_native_stream_attention_hook.__name__}",
        layer_count=None if model is None else len(model.blocks),
    )


def set_official_attention_hook(hook: Any | None) -> None:
    global _OFFICIAL_ATTENTION_HOOK
    from ..model.layers import set_native_stream_attention_hook

    set_native_stream_attention_hook(hook)
    _OFFICIAL_ATTENTION_HOOK = hook


def official_attention_hook_mode() -> str | None:
    hook = _OFFICIAL_ATTENTION_HOOK
    if hook is None:
        return None
    mode = getattr(hook, "mode", None)
    return None if mode is None else str(getattr(mode, "value", mode))


def official_attention_denoise_step_index() -> int | None:
    return getattr(_OFFICIAL_ATTENTION_HOOK, "_denoise_step_index", None)
