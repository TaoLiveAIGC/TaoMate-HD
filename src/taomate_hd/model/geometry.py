# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""MiniMax H3 spatial and temporal request geometry."""

from __future__ import annotations
import math
from dataclasses import asdict, dataclass

FPS = 24
CANVAS_MULTIPLE = 32


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    width: int
    height: int
    fps: int
    frame_count: int
    duration_seconds: float
    video_latent_t: int
    video_latent_h: int
    video_latent_w: int
    audio_latent_t: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    return int(frame_count) + (5 - int(frame_count)) % 17


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return (int(frame_count) - 5) // 17 * 5 + 2


def frame_count_from_video_latent_t(value: int) -> int:
    if value == 1:
        return 1
    if value < 2 or (value - 2) % 5:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(value) - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    return int(round(float(duration_seconds) * 40.0))


def resolve_geometry(
    *,
    duration_seconds: float,
    width: int,
    height: int,
) -> MiniMaxH3Geometry:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if isinstance(width, bool) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    (width, height) = (int(width), int(height))
    if width <= 0 or height <= 0 or width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("explicit width and height must be positive multiples of 32")
    frames = align_frame_count(int(round(duration_seconds * FPS)))
    aligned_duration = frames / FPS
    return MiniMaxH3Geometry(
        width=width,
        height=height,
        fps=FPS,
        frame_count=frames,
        duration_seconds=aligned_duration,
        video_latent_t=video_latent_t(frames),
        video_latent_h=height // 16,
        video_latent_w=width // 16,
        audio_latent_t=audio_latent_t(aligned_duration),
    )
