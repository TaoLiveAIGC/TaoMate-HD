# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Media timing transforms shared by H3 long-video publication paths."""

from __future__ import annotations


def endpoint_preserving_video_filter(
    *, native_frames: int, published_frames: int, fps: int = 24
) -> str:
    """Compress the complete native request onto the published frame timeline."""
    if native_frames <= 1 or published_frames <= 1:
        raise ValueError("video timing requires at least two native and published frames")
    return f"trim=end_frame={native_frames},setpts=(PTS-STARTPTS)*{published_frames - 1}/{native_frames - 1},fps={fps}:round=near,tpad=stop_mode=clone:stop=1,trim=end_frame={published_frames}"


def exact_audio_delivery_filter(
    *, native_samples: int, published_samples: int, sample_rate: int = 32000
) -> str:
    """Retimestamp the complete native waveform to an exact delivery sample count."""
    if native_samples <= 0 or published_samples <= 0:
        raise ValueError("audio timing requires positive sample counts")
    return f"atempo={native_samples}/{published_samples},aresample={sample_rate},apad=whole_len={published_samples},atrim=end_sample={published_samples},asetpts=N/SR/TB"
