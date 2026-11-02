# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Decode native Progressive chunks online with exact temporal-window DP.

CPU Gloo prefetch is independent of each GPU's window work. Window boundaries,
halo, microbatch=2, normalization and RGB24 conversion match the Direct decoder.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import torch
from taomate_hd.streaming.progressive_pipeline import _Receiver
from taomate_hd.streaming.geometry import (
    plan_duration,
    plan_native_stream,
    canonical_continuation_plan,
)
from taomate_hd.model.components import partition_root
from taomate_hd.model.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
    minimax_h3_unpack_audio_tokens,
)
from taomate_hd.model.output import (
    H3LatentDecoder,
    _TemporalWindowDPDecode,
    _reverse_normalize,
    write_h3_rgb24_shard,
    save_h3_video_mp4_from_rgb24_shards,
    configure_bundled_video_vae_tile_parallel,
)
from taomate_hd.inference.media_timing import (
    endpoint_preserving_video_filter,
    exact_audio_delivery_filter,
)


class LatentPrefetch:
    def __init__(self, *, source, decoder, plans, rank, device, port):
        (self.source, self.decoder, self.plans) = (source, decoder, plans)
        (self.rank, self.device) = (rank, device)
        (self.audio_parts, self.audio_complete) = ([], False)
        self.receiver = _Receiver(port) if rank == 0 else None
        self.condition = threading.Condition()
        (self.available, self.error, self.ready_event) = (0, None, None)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(device=self.device)
            (height, width) = self.source.shape[-2:]
            for request, plan in enumerate(self.plans):
                for phase in plan.phases:
                    for modality in ("video", "audio"):
                        shape = (
                            (phase.video_latent_count * (height // 2) * (width // 2), 96)
                            if modality == "video"
                            else (2 * phase.audio_latent_count, 32)
                        )
                        rows = (
                            self.receiver.receive([request, [phase.index, modality]])
                            if self.rank == 0
                            else torch.empty(shape, dtype=torch.float32)
                        )
                        if modality == "audio":
                            if self.rank == 0:
                                self.audio_parts.append(
                                    minimax_h3_unpack_audio_tokens(
                                        rows, audio_t=2 * phase.audio_latent_count, audio_channel=2
                                    )
                                )
                            continue
                        torch.distributed.broadcast(rows, src=0)
                        if modality == "video":
                            with torch.cuda.stream(stream):
                                latent = minimax_h3_unpatchify_video_tokens(
                                    rows.to(self.device),
                                    latent_shape=(
                                        phase.video_latent_count,
                                        height // 2,
                                        width // 2,
                                        24,
                                    ),
                                    patch_size=(1, 2, 2),
                                )
                                normalized = _reverse_normalize(
                                    latent,
                                    mean_values=self.decoder.video_config["latents_mean"],
                                    std_values=self.decoder.video_config["latents_std"],
                                )
                                stop = self.available + phase.video_latent_count
                                self.source[:, :, self.available : stop].copy_(normalized)
                                event = torch.cuda.Event()
                                event.record(stream)
                            with self.condition:
                                (self.available, self.ready_event) = (stop, event)
                                self.condition.notify_all()
            with self.condition:
                self.audio_complete = True
                self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()
        finally:
            if self.receiver is not None:
                self.receiver.close()

    def wait(self, tokens):
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.available >= tokens or self.error is not None, timeout=300
            ):
                raise TimeoutError(f"VAE waiting for latent {tokens}; received {self.available}")
            if self.error is not None:
                raise RuntimeError("VAE latent receive failed") from self.error
            event = self.ready_event
        torch.cuda.current_stream(self.device).wait_event(event)

    def wait_audio(self):
        with self.condition:
            self.condition.wait_for(lambda: self.audio_complete or self.error is not None)
            if self.error is not None:
                raise RuntimeError("Audio VAE latent receive failed") from self.error
        return self.audio_parts


class OnlineTemporalDP(_TemporalWindowDPDecode):
    def __init__(self, *, prefetch, **kwargs):
        super().__init__(**kwargs)
        self.prefetch = prefetch
        self.ready_parts = []

    def _window_input(self, chunk_index):
        body_tokens = int(self.source.shape[2]) - self.head_tokens - self.tail_tokens
        start = chunk_index * int(self.active_vae.tokens_chunk_size)
        stop = start + int(self.active_vae.tokens_chunk_size) + int(self.active_vae.token_overlap)
        required = self.head_tokens + min(stop, body_tokens)
        if chunk_index == self.num_chunks - 1:
            required = int(self.source.shape[2])
        self.prefetch.wait(required)
        clip = self.source[
            :, :, self.head_tokens + start : self.head_tokens + min(stop, body_tokens)
        ]
        padding = min(stop, body_tokens + self.pad_tokens) - max(start, body_tokens)
        if padding > 0:
            last = self.source[
                :, :, self.head_tokens + body_tokens - 1 : self.head_tokens + body_tokens
            ]
            clip = torch.cat((clip, last.repeat(1, 1, padding, 1, 1)), dim=2)
        if chunk_index == 0 and self.head_tokens:
            clip = torch.cat((self.source[:, :, :1], clip), dim=2)
        if chunk_index == self.num_chunks - 1 and self.tail_tokens:
            clip = torch.cat((clip, self.source[:, :, -1:]), dim=2)
        return clip

    def _write_part(self, part):
        before = self.write_pos
        super()._write_part(part)
        if self.write_pos > before:
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.copy_stream)
            self.ready_parts.append(
                (self.retained_frame_start + before, self.write_pos - before, event)
            )


class SpatialHeadTemporalDP(OnlineTemporalDP):
    """All decoder GPUs split each original window's spatial tiles and reuse its halo."""

    def __init__(self, *, head_state, **kwargs):
        super().__init__(**kwargs)
        self.head_state = head_state
        self.boundary_halo_window_count = 0
        self.defer_output_sync = True
        self.window_events = []

    def _write_part(self, part):
        if (
            torch.distributed.get_rank()
            != self.video_parallel_rank % torch.distributed.get_world_size()
        ):
            self.write_pos += min(int(part.shape[2]), self.retained_output_frames - self.write_pos)
            if self.decoded_cpu is None:
                self.decoded_cpu = torch.empty(0, dtype=torch.uint8)
            return
        super()._write_part(part)

    def _decode_windows(self, chunk_indices, clip_inputs):
        (begin, end) = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        (head, primary, overlap, tail) = super()._decode_windows(chunk_indices, clip_inputs)[0]
        end.record()
        self.window_events.append((chunk_indices[0], begin, end))
        if self.head_state.get("head_overlap") is not None:
            primary = self.active_vae.blend(
                self.head_state["head_overlap"], primary, int(self.active_vae.frame_overlap), dim=-3
            )
        self.head_state["head_overlap"] = overlap
        return [(head, primary, overlap, tail)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default="case_001")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1376)
    parser.add_argument("--port", type=int, default=29770)
    args = parser.parse_args()
    (rank, world) = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group("gloo")
    spatial_group = torch.distributed.new_group(backend="nccl")
    root = Path(os.environ["H3_OUTPUT_ROOT"]) / "inference" / args.run_id
    case = root / args.case_id
    profile = root / "pipeline_profile" / args.case_id
    scratch = Path(os.environ.get("H3_RGB_TEMP_ROOT", "/dev/shm")) / args.run_id / args.case_id
    for directory in (case, profile, scratch):
        directory.mkdir(parents=True, exist_ok=True)
    component_root = partition_root(Path(args.model_root))
    decoder = H3LatentDecoder(
        video_vae_root=component_root / "video_vae",
        audio_vae_root=component_root / "audio_vae",
        load_audio=rank == 0,
    )
    decoder.enable_video_vae_exact_fusions()
    if rank == 0:
        decoder.audio_vae.to(device).eval()
    vae = decoder.video_vae.to(device).eval()
    vae.model.stack_tiling = True
    from taomate_hd.inference.video_vae_exact_fusions import install_h3_video_vae_exact_fusions_

    install_h3_video_vae_exact_fusions_(vae.model.decoder, device=device)
    configure_bundled_video_vae_tile_parallel(vae, group=spatial_group, rank=rank, world_size=world)
    prepare = getattr(vae, "prepare_decoder_autocast_weights", None)
    if callable(prepare):
        prepare(torch.float16)
    base_plan = plan_duration(5)
    continuation = plan_native_stream(base_plan.native_frame_count, groups_per_phase=2)
    plans = [
        base_plan if i == 0 else canonical_continuation_plan(continuation, request_index=i)
        for i in range(args.requests)
    ]
    tokens = sum((p.phases[-1].video_latent_stop for p in plans))
    source = torch.zeros(
        (1, 24, tokens, args.height // 16, args.width // 16), device=device, dtype=torch.float32
    )
    prefetch = LatentPrefetch(
        source=source,
        decoder=decoder,
        plans=plans,
        rank=rank,
        device=device,
        port=args.port,
    )
    module = importlib.import_module(vae.model.__class__.__module__)
    temporal_dtype = getattr(module, "_resolve_temporal_cat_dtype", lambda: None)()
    rev = vae.processor.transform_rev
    options = dict(
        prefetch=prefetch,
        active_vae=vae.model,
        source=source,
        video_parallel_world_size=world,
        temporal_cat_dtype=temporal_dtype,
        temporal_window_microbatch_size=2,
        pixel_postprocess=vae.processor.revert_tensor,
        gpu_rgb24_output=True,
        fused_gpu_rgb24_output=True,
        pixel_denormalize_mean=tuple((float(x) for x in rev.mean)),
        pixel_denormalize_std=tuple((float(x) for x in rev.std)),
    )
    temporal = OnlineTemporalDP(video_parallel_rank=rank, **options)
    options["video_parallel_world_size"] = temporal.num_chunks
    if os.environ.get("H3_PIPELINE_PREWARM", "1") == "1":
        (cpu_rng, gpu_rng) = (torch.get_rng_state(), torch.cuda.get_rng_state())
        warm_options = {k: v for (k, v) in options.items() if k != "prefetch"}
        for _ in range(2):
            for window in range(min(3, temporal.num_chunks)):
                warm = _TemporalWindowDPDecode(video_parallel_rank=window, **warm_options)
                warm.boundary_halo_window_count = 0
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    warm.decode()
                del warm
        torch.cuda.synchronize()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    from taomate_hd.streaming.pipeline_readiness import await_pipeline_ready

    released_at = await_pipeline_ready("vae")
    anchor = torch.cuda.Event(enable_timing=True)
    anchor.record()
    anchor.synchronize()
    epoch = time.time()
    input_ready_at = released_at if isinstance(released_at, (int, float)) else epoch
    audio_result = {}
    audio_future = None
    if rank == 0:
        audio_stream = torch.cuda.Stream(device=device)

        def receive_and_decode_audio():
            torch.cuda.set_device(device)
            parts = prefetch.wait_audio()
            with torch.cuda.stream(audio_stream), torch.inference_mode():
                (a, b) = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                (waveform, sample_rate) = decoder.decode_audio(
                    audio_latents=torch.cat(parts, dim=2), device=device, offload_after_decode=False
                )
                b.record()
                b.synchronize()
            audio_result.update(waveform=waveform, sample_rate=sample_rate, start=a, end=b)

        audio_executor = ThreadPoolExecutor(max_workers=1)
        audio_future = audio_executor.submit(receive_and_decode_audio)
    prefetch.thread.start()
    shared_halo = {}
    (active_decoders, receipts, writes) = ([], [], [])
    writer = ThreadPoolExecutor(max_workers=1)

    def write_window(active, frames, window):
        active.ready_parts[-1][2].synchronize()
        return write_h3_rgb24_shard(
            scratch / f"shard{window}.rgb24", frames=frames, frame_start=active.retained_frame_start
        )

    for window in range(temporal.num_chunks):
        active = SpatialHeadTemporalDP(
            video_parallel_rank=window, head_state=shared_halo, **options
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            (frames, receipt) = active.decode()
        active_decoders.append(active)
        if rank == window % world:
            receipts.append(receipt)
            writes.append(writer.submit(write_window, active, frames, window))
    prefetch.thread.join(timeout=310)
    if prefetch.thread.is_alive() or prefetch.error is not None:
        raise RuntimeError("VAE stream did not finish") from prefetch.error
    if audio_future is not None:
        audio_future.result()
        audio_executor.shutdown()
    shards = [future.result() for future in writes]
    writer.shutdown()
    torch.cuda.synchronize(device)
    (cuda, ready) = ([], [])
    for active in active_decoders:
        cuda.extend(
            (
                dict(
                    kind="cuda_compute",
                    label="video_vae",
                    window=index,
                    start=epoch + anchor.elapsed_time(a) / 1000,
                    end=epoch + anchor.elapsed_time(b) / 1000,
                )
                for (index, a, b) in active.window_events
            )
        )
        if rank == active.video_parallel_rank % world:
            ready.extend(
                (
                    dict(
                        frame_start=start,
                        frames=count,
                        ready_at=epoch + anchor.elapsed_time(event) / 1000,
                    )
                    for (start, count, event) in active.ready_parts
                )
            )
    if rank == 0:
        cuda.append(
            dict(
                kind="cuda_compute",
                label="audio_vae",
                start=epoch + anchor.elapsed_time(audio_result["start"]) / 1000,
                end=epoch + anchor.elapsed_time(audio_result["end"]) / 1000,
            )
        )
    receipt = dict(rank=rank, events=cuda, frames_ready=ready, decoder_shards=receipts)
    (profile / f"vae_rank{rank}.json").write_text(json.dumps(receipt, indent=2))
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, dict(shards=shards, ready=ready, compute=cuda))
    if rank == 0:
        ordered = sorted(
            (part for item in gathered for part in item["ready"]),
            key=lambda part: part["frame_start"],
        )
        rgb_complete = max((part["ready_at"] for part in ordered))
        audio_complete = epoch + anchor.elapsed_time(audio_result["end"]) / 1000
        model_seconds = max(rgb_complete, audio_complete) - input_ready_at
        metrics = dict(
            model_seconds=model_seconds,
            model_fps=args.requests * 120 / model_seconds,
            first_rgb_seconds=ordered[0]["ready_at"] - input_ready_at,
            input_ready_epoch=input_ready_at,
            rgb_complete_epoch=rgb_complete,
            audio_complete_epoch=audio_complete,
            published_frames=args.requests * 120,
        )
        video = case / "video.one_shot.mp4"
        save_h3_video_mp4_from_rgb24_shards(
            video,
            shards=[part for item in gathered for part in item["shards"]],
            fps=24,
            crf=18,
            verify_decode=False,
            video_filter=endpoint_preserving_video_filter(
                native_frames=temporal.output_frames, published_frames=args.requests * 120
            ),
            published_frames=args.requests * 120,
        )
        (waveform, sample_rate) = (audio_result["waveform"], audio_result["sample_rate"])
        pcm = scratch / "audio.f32le"
        waveform[0].transpose(0, 1).contiguous().numpy().tofile(pcm)
        final = case / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-f",
                "f32le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(waveform.shape[1]),
                "-i",
                str(pcm),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-af",
                exact_audio_delivery_filter(
                    native_samples=waveform.shape[2],
                    published_samples=args.requests * 5 * sample_rate,
                    sample_rate=sample_rate,
                ),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                str(args.requests * 5),
                str(final),
            ],
            check=True,
        )
        (case / "result.json").write_text(
            json.dumps(
                dict(
                    video=str(final),
                    requests=args.requests,
                    width=args.width,
                    height=args.height,
                    pipeline_external_vae=True,
                    audio_source="student_final_clean",
                    audio_decode_mode="one_shot_native_timeline",
                    **metrics,
                ),
                indent=2,
            )
        )
        for window in range(temporal.num_chunks):
            (scratch / f"shard{window}.rgb24").unlink()
        pcm.unlink()
        video.unlink()
        print(json.dumps(dict(status="complete", video=str(final), **metrics)), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
