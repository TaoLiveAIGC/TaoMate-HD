# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def canvas(value):
    width, height = (int(v) for v in value.lower().split("x"))
    return width, height


def main():
    parser = argparse.ArgumentParser(description="TaoMate-HD progressive audio-video generation")
    parser.add_argument("--model", type=Path, required=True, help="MiniMax-H3 model directory")
    parser.add_argument("--adapter", type=Path, required=True, help="TaoMate-HD inference adapter")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="One prompt, reused across windows")
    inputs.add_argument("--prompts", type=Path, help="JSON with prompts and optional seeds")
    parser.add_argument("--duration", type=float, default=10, help="Output seconds")
    parser.add_argument("--resolution", type=canvas, default=(768, 1376), metavar="WIDTHxHEIGHT")
    parser.add_argument(
        "--low-resolution",
        type=canvas,
        metavar="WIDTHxHEIGHT",
        help="Default: 480/768 of output, aligned to 32 pixels",
    )
    parser.add_argument("--mode", choices=["serial", "accelerated"], default="serial")
    parser.add_argument("--gpus", type=int, default=8, help="Serial GPU count; accelerated pipeline uses 8 GPUs")
    parser.add_argument("--tp", type=int, default=2, help="Serial tensor parallel degree")
    parser.add_argument("--seed", type=int, default=8301)
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if any((v <= 0 or v % 32 for v in args.resolution)):
        parser.error(
            "Canvas dimensions must be positive multiples of the model’s 32-pixel token grid"
        )
    if args.mode == "accelerated" and args.gpus != 8:
        parser.error("The accelerated pipeline topology uses 1+1+4+2 GPUs")
    count = math.ceil(args.duration / 5)
    data = json.loads(args.prompts.read_text()) if args.prompts else {"prompts": [args.prompt]}
    prompts = data["prompts"]
    seeds = data.get("seeds", [args.seed + i for i in range(count)])
    job = {
        "mode": args.mode,
        "model": str(args.model.expanduser().resolve()),
        "adapter": str(args.adapter.expanduser().resolve()),
        "duration": args.duration,
        "resolution": args.resolution,
        "low_resolution": args.low_resolution
        or tuple((max(32, round(v * 480 / 768 / 32) * 32) for v in args.resolution)),
        "prompts": [prompts[min(i, len(prompts) - 1)] for i in range(count)],
        "seeds": [seeds[min(i, len(seeds) - 1)] for i in range(count)],
        "work": str(args.output.resolve() / "inference/run/case_001"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    job_path = args.output.resolve() / "config.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2))
    env = {k: v for (k, v) in os.environ.items() if not k.startswith("H3_")}
    env.update(
        H3_MODEL_ROOT=job["model"],
        H3_OUTPUT_ROOT=str(args.output.resolve()),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONUNBUFFERED="1",
        H3_PROMPT_START_LOCAL_PREFIX_VIDEO_LATENTS="0",
        H3_RETAIN_INTERMEDIATE_MEDIA="0",
        H3_RELEASE_PRECOMPUTED_TEXT_ENCODER="1",
        H3_RGB_TEMP_ROOT=str(args.output.resolve() / ".decode"),
    )
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus)))).split(
        ","
    )
    base = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    children = []
    logs = []
    try:
        if args.mode == "serial":
            commands = [
                (
                    "serial",
                    devices,
                    args.gpus,
                    [
                        "-m",
                        "taomate_hd.generate",
                        "--job",
                        str(job_path),
                        "--tp",
                        str(args.tp),
                        "--ulysses",
                        str(args.gpus // args.tp),
                    ],
                    {},
                )
            ]
        else:
            for flag in (
                "STAGE_LOCAL_PREDICTED_KV",
                "FUSED_PREDICTED_KV",
                "LOW_EARLY_SEND",
                "STAGE0_FREE_RUN",
                "FUSED_STATIC_PLANS",
                "LOW_REUSE_PREVIOUS_KV",
                "LOW_CONDITION_CACHE",
                "LOW_CONDITION_COMPACT",
                "LOW_FUSED_QKV_RESTORE",
                "LOW_COMPACT_ATTN_OUTPUT",
            ):
                env["H3_PROGRESSIVE_" + flag] = "1"
            env.update(
                H3_PROGRESSIVE_PP_SYNC_START="1",
                H3_PROGRESSIVE_PP_ASYNC_SEND="1",
                H3_PROGRESSIVE_PP_HOSTS="127.0.0.1,127.0.0.1,127.0.0.1",
                H3_PROGRESSIVE_PP_VAE_HOST="127.0.0.1",
                H3_PROGRESSIVE_PP_READY_HOST="127.0.0.1",
                H3_PROGRESSIVE_PP_PORT=str(args.port),
                H3_PROGRESSIVE_PP_VAE_PORT=str(args.port + 30),
                H3_PROGRESSIVE_PP_READY_PORT=str(args.port + 50),
                H3_INFER_PORT_OFFSET=str(args.port - 29940),
                H3_VIDEO_VAE_FUSED_ROPE="1",
                H3_ULYSSES_FUSED_LAYOUT="1",
                H3_SAGE_FUSED_VALUE_PADDING="1",
                H3_HIGH_SKIP_UNUSED_LOW_PREP="1",
                H3_PROGRESSIVE_CONTINUATION_GROUPS="2",
                H3_PROGRESSIVE_LOW_TAIL_MODE="high",
                H3_PROGRESSIVE_STAGE_TIMING="1",
                H3_PIPELINE_PREWARM="1",
                H3_PIPELINE_WARMUP_DIR=str(
                    args.output.resolve() / "inference/run/pipeline_profile/case_001"
                ),
            )
            shared = str(args.output.resolve() / "text.pt")
            Path(shared).unlink(missing_ok=True)
            commands = []
            for stage, indices, tp in (
                (0, devices[:1], 1),
                (1, devices[1:2], 1),
                (2, devices[2:6], 2),
            ):
                text_key = (
                    "H3_PRECOMPUTED_TEXT_EXPORT_PATH" if stage == 2 else "H3_PRECOMPUTED_TEXT_PATH"
                )
                commands.append(
                    (
                        f"stage{stage}",
                        indices,
                        len(indices),
                        [
                            "-m",
                            "taomate_hd.generate",
                            "--job",
                            str(job_path),
                            "--tp",
                            str(tp),
                            "--ulysses",
                            str(len(indices) // tp),
                        ],
                        {"H3_PROGRESSIVE_PP_STAGE": str(stage), text_key: shared},
                    )
                )
            commands.append(
                (
                    "vae",
                    devices[6:8],
                    2,
                    [
                        "-m",
                        "taomate_hd.decoder",
                        "--run-id",
                        "run",
                        "--case-id",
                        "case_001",
                        "--requests",
                        str(count),
                        "--width",
                        str(args.resolution[0]),
                        "--height",
                        str(args.resolution[1]),
                        "--port",
                        str(args.port + 30),
                        "--model-root",
                        job["model"],
                    ],
                    {},
                )
            )
        for name, gpu_ids, nproc, command, extra in commands:
            log = (args.output / f"{name}.log").open("w")
            logs.append(log)
            child_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(gpu_ids), **extra)
            children.append(
                subprocess.Popen(
                    base + [f"--nproc-per-node={nproc}"] + command,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while any((p.poll() is None for p in children)):
            for p in children:
                if p.returncode not in (None, 0):
                    raise RuntimeError(f"Inference failed; see logs in {args.output}")
            time.sleep(1)
        if any((p.returncode for p in children)):
            raise RuntimeError(f"Inference failed; see logs in {args.output}")
        video = Path(job["work"]) / "video.mp4"
        final = args.output / "video.mp4"
        if args.mode == "accelerated" and args.duration != count * 5:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-t",
                    str(args.duration),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(final),
                ],
                check=True,
            )
        else:
            shutil.copyfile(video, final)
        print(final.resolve())
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for log in logs:
            log.close()
