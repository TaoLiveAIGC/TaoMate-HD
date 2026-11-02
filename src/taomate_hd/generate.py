# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""Generate consecutive prompt windows with persistent stage-local AV history."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import torch
from .runtime.direct import H3Runtime
from .streaming.geometry import plan_duration
from .streaming.policy import StreamingInferencePolicy
from .streaming.runtime import install_streaming_runtime
from .inference.acceleration import InferenceAccelerationConfig
from .inference.media_timing import endpoint_preserving_video_filter, exact_audio_delivery_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--ulysses", type=int, default=4)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    pipelined = job["mode"] == "accelerated"
    acceleration = InferenceAccelerationConfig(
        streaming_attention_backend="sage_attn" if pipelined else "sdpa",
        dit_w8a8_policy="interior_fc1_qkv_v1" if pipelined else None,
        dit_w8a8_workspace=pipelined,
        sol_engine_kernels=pipelined,
        fused_qkv_preprocess=pipelined,
        fused_lora_row_reduce=pipelined,
        dit_lora_bf16_buffers=pipelined,
        packed_dense_flash_attn3=pipelined,
        inplace_row_parallel_reduce=pipelined,
        static_timestep_adaln_cache=pipelined,
        video_vae_exact_fusions=pipelined,
        video_vae_batched_tiles=pipelined,
        video_vae_balanced_cpu_output=pipelined,
        record_component_timings=True,
    )
    runtime = H3Runtime(
        model_root=Path(job["model"]),
        adapter_checkpoint=Path(job["adapter"]),
        tp_size=args.tp,
        ulysses_degree=args.ulysses,
        inference_acceleration=acceleration,
    )
    (prompts, seeds) = (job["prompts"], job["seeds"])
    text = runtime.precompute_text_prompts(prompts)
    output = Path(job["work"])
    output.mkdir(parents=True, exist_ok=True)
    policy = StreamingInferencePolicy()
    (width, height) = job["resolution"]
    (low_width, low_height) = job["low_resolution"]
    (video_parts, audio_parts) = ([], [])
    with install_streaming_runtime(
        plan=plan_duration(5),
        server_args=runtime.server_args,
        reuse_across_requests=True,
        policy=policy,
        attention_backend=acceleration.streaming_attention_backend,
    ) as overlay:
        for index, (seed, encoded) in enumerate(zip(seeds, text)):
            runtime.generate(
                seed=seed,
                duration_seconds=5,
                low_resolution=(low_width, low_height),
                high_resolution=(width, height),
                video_noise_seed=(seed + index * 1000003) % 2**63,
                audio_noise_seed=seed,
                precomputed_text=encoded,
                retain_clean_latents_on_device=pipelined,
            )
            execution = overlay.last_execution
            if not pipelined:
                video_parts.append(
                    runtime.last_clean_video_latents[
                        :, :, execution.video_transport_prefix_latents :
                    ].cpu()
                )
                if runtime.is_main_process:
                    audio_parts.append(
                        runtime.last_clean_audio_latents[
                            :, :, execution.audio_transport_prefix_latents_per_channel :
                        ].cpu()
                    )
            if runtime.is_main_process:
                print(f"Completed window {index + 1}/{len(prompts)}", flush=True)
        if pipelined:
            overlay._progressive_pipeline.finish(output.parent / "pipeline_profile" / output.name)
        overlay.release_retained_state()
    if not pipelined:
        video = torch.cat(video_parts, dim=2).contiguous()
        native_frames = 5 + 17 * ((video.shape[2] - 2) // 5)
        full_frames = len(prompts) * 5 * 24
        video_path = output / "video_only.mp4"
        runtime.decode_video_latents(
            video,
            output_path=video_path,
            video_filter=endpoint_preserving_video_filter(
                native_frames=native_frames, published_frames=full_frames
            ),
            published_video_frames=full_frames,
            video_crf=18,
        )
        if runtime.is_main_process:
            (audio, rate) = runtime.decode_audio_latents(torch.cat(audio_parts, dim=2).contiguous())
            pcm = audio[0].float().T.contiguous().cpu().numpy()
            (output / "audio.f32").write_bytes(pcm.tobytes())
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    "-f",
                    "f32le",
                    "-ar",
                    str(rate),
                    "-ac",
                    str(pcm.shape[1]),
                    "-i",
                    str(output / "audio.f32"),
                    "-af",
                    exact_audio_delivery_filter(
                        native_samples=pcm.shape[0], published_samples=len(prompts) * 5 * 32000
                    ),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-t",
                    str(job["duration"]),
                    "-movflags",
                    "+faststart",
                    str(output / "video.mp4"),
                ],
                check=True,
            )
            video_path.unlink()
            (output / "audio.f32").unlink()
    timings = runtime.component_timing_receipt()
    if runtime.is_main_process:
        (
            output / f"execution_{os.environ.get('H3_PROGRESSIVE_PP_STAGE', 'serial')}.json"
        ).write_text(
            json.dumps(
                {
                    "model": runtime.model_resolution,
                    "executions": [e.to_dict() for e in overlay.executions],
                    "timings": timings,
                },
                indent=2,
            )
        )
    runtime.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
