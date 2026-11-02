# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from __future__ import annotations
import importlib.metadata
import json
import os
import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TypeVar
from ..inference.acceleration import (
    InferenceAccelerationConfig,
    InferenceComponentTimer,
    apply_dit_inference_acceleration,
)
from ..denoise_schedule import DISTILLED_STATE_INDICES
from ..paths import H3Paths
from .ppu_torch import execution_torch_facts, load_execution_torch

_TP_SIZE = 2
_DEFAULT_ULYSSES_DEGREE = 2
_T = TypeVar("_T")


class H3RuntimeError(RuntimeError):
    pass


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _native_environment(torch_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "taomate_hd.runtime.native.v1",
        "backend": "taomate_hd_native_torch",
        "execution_torch": dict(torch_facts),
        "distributions": {
            name: _distribution_version(name)
            for name in (
                "diffusers",
                "imageio",
                "imageio-ffmpeg",
                "safetensors",
                "scipy",
                "transformers",
            )
        },
        "external_service_required": False,
    }


class H3Runtime:
    """Progressive T2AV runtime over tensor and sequence parallel groups."""

    def __init__(
        self,
        *,
        model_root: Path | None = None,
        tp_size: int = _TP_SIZE,
        ulysses_degree: int = _DEFAULT_ULYSSES_DEGREE,
        adapter_checkpoint: Path,
        inference_acceleration: InferenceAccelerationConfig | None = None,
    ) -> None:
        self.inference_acceleration = inference_acceleration or InferenceAccelerationConfig()
        load_time_w8a8_policy = self.inference_acceleration.dit_w8a8_policy
        paths = H3Paths.from_env()
        self.source_model_root = (model_root or paths.model_root).resolve(strict=True)
        self.output_root = paths.output_root.resolve()
        self.tp_size = tp_size
        self.ulysses_degree = ulysses_degree
        self.expected_world_size = tp_size * ulysses_degree
        self.server_args = SimpleNamespace(
            backend="native",
            model_variant="fl2va",
            num_gpus=self.expected_world_size,
            tp_size=self.tp_size,
            sp_degree=self.ulysses_degree,
            ulysses_degree=self.ulysses_degree,
            ring_degree=1,
            dp_size=1,
            enable_cfg_parallel=False,
            cfg_parallel_degree=1,
            enable_torch_compile=False,
            warmup_mode="off",
            quantization=None,
        )
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        if self.world_size != self.expected_world_size:
            raise ValueError(
                f"native H3 torchrun world size must equal tp_size * ulysses_degree = {self.expected_world_size}, got {self.world_size}"
            )
        if not 0 <= self.rank < self.world_size or not 0 <= self.local_rank < self.world_size:
            raise ValueError("invalid torchrun rank environment")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            self.torch = load_execution_torch()
            torch_facts = execution_torch_facts(self.torch)
        except (ImportError, RuntimeError, OSError) as exc:
            raise H3RuntimeError(f"PyTorch is unavailable: {exc}") from exc
        if not torch_facts["cuda_available"] or self.local_rank >= torch_facts["cuda_device_count"]:
            raise H3RuntimeError(
                f"CUDA device for LOCAL_RANK={self.local_rank} is unavailable; cuda_device_count={torch_facts['cuda_device_count']} CUDA_VISIBLE_DEVICES={torch_facts['cuda_visible_devices']!r}"
            )
        self.runtime_environment = _native_environment(torch_facts)
        self.torch.cuda.set_device(self.local_rank)
        self.device = self.torch.device("cuda", self.local_rank)
        if not self.torch.distributed.is_initialized():
            self.torch.distributed.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=3)
            )
        if (
            self.torch.distributed.get_world_size() != self.world_size
            or self.torch.distributed.get_rank() != self.rank
        ):
            raise H3RuntimeError("torchrun process group differs from rank environment")
        self.parallel_context = self._build_parallel_context()
        self.model_root = self.source_model_root
        self.model_resolution = {
            "variant": "fl2va",
            "runtime_backend": "native",
            "effective_model_root": str(self.model_root),
            "runtime_environment": self.runtime_environment,
        }
        if self.is_main_process:
            self.output_root.mkdir(mode=493, parents=True, exist_ok=True)
        self.barrier()
        pipeline_error: str | None = None
        self.pipeline: Any | None = None
        self._precomputed_text_path = os.environ.get("H3_PRECOMPUTED_TEXT_PATH")
        try:
            from ..model.pipeline import MiniMaxH3NativePipeline

            self.pipeline = MiniMaxH3NativePipeline(
                model_root=self.model_root,
                device=self.device,
                parallel_context=self.parallel_context,
                rank=self.rank,
                world_size=self.world_size,
                defer_text_encoder=bool(self._precomputed_text_path),
                transformer_load_w8a8_policy=load_time_w8a8_policy,
            )
        except Exception as exc:
            pipeline_error = _exception_text(exc)
        pipeline_error = self._collective_error("native H3 pipeline load", pipeline_error)
        if pipeline_error:
            raise H3RuntimeError(pipeline_error)
        assert self.pipeline is not None
        self._last_clean_video_latents: Any | None = None
        self._last_clean_audio_latents: Any | None = None
        self._output_decoder_preload_thread: threading.Thread | None = None
        self._output_decoder_preload_error: str | None = None
        self._output_decoder_preload_seconds: float | None = None
        self._text_precompute_receipt: dict[str, Any] | None = None
        from ..inference.lora_checkpoint import apply_h3_lora_checkpoint

        receipt = apply_h3_lora_checkpoint(self.pipeline.transformer, adapter_checkpoint)
        self.adapter_resolution = dict(receipt)
        self.model_resolution["adapter"] = self.adapter_resolution
        acceleration_error: str | None = None
        acceleration_receipt: dict[str, Any] | None = None
        try:
            acceleration_receipt = apply_dit_inference_acceleration(
                self.torch, self.pipeline.transformer, self.inference_acceleration
            )
            self.pipeline.configure_inference_video_vae(
                exact_fusions=self.inference_acceleration.video_vae_exact_fusions,
                batched_tiles=self.inference_acceleration.video_vae_batched_tiles,
                balanced_cpu_output=self.inference_acceleration.video_vae_balanced_cpu_output,
                temporal_window_dp=self.inference_acceleration.video_vae_temporal_window_dp,
                temporal_window_microbatch_size=self.inference_acceleration.video_vae_temporal_window_microbatch_size,
                gpu_rgb24_output=self.inference_acceleration.video_vae_gpu_rgb24_output,
                keep_on_device_after_decode=self.inference_acceleration.video_vae_keep_on_device_after_decode,
                decoder_tile_size=self.inference_acceleration.video_vae_decoder_tile_size,
                decoder_tile_overlap=self.inference_acceleration.video_vae_decoder_tile_overlap,
            )
        except Exception as exc:
            acceleration_error = _exception_text(exc)
        acceleration_error = self._collective_error(
            "native H3 inference acceleration", acceleration_error
        )
        if acceleration_error:
            raise H3RuntimeError(acceleration_error)
        self.model_resolution["inference_acceleration"] = acceleration_receipt
        self._component_timer: InferenceComponentTimer | None = None
        if self.inference_acceleration.record_component_timings:
            self._component_timer = InferenceComponentTimer(
                self.torch, rank=self.rank, world_size=self.world_size
            )
            self._component_timer.attach_transformer(self.pipeline.transformer)
            self.pipeline.set_component_timer(self._component_timer)
        sigma_points = len(DISTILLED_STATE_INDICES)
        self.model_resolution["denoise_schedule"] = {
            "base_sigma_points": 50,
            "retained_state_indices": list(DISTILLED_STATE_INDICES),
            "sigma_points": sigma_points,
            "model_forwards": sigma_points - 1,
        }

    def _build_parallel_context(self) -> Any:
        from ..distributed import ParallelContext

        tp_groups: list[Any] = []
        for ulysses_rank in range(self.ulysses_degree):
            ranks = [ulysses_rank * self.tp_size + tp_rank for tp_rank in range(self.tp_size)]
            tp_groups.append(self.torch.distributed.new_group(ranks=ranks, backend="nccl"))
        ulysses_groups: list[Any] = []
        for tp_rank in range(self.tp_size):
            ranks = [
                ulysses_rank * self.tp_size + tp_rank for ulysses_rank in range(self.ulysses_degree)
            ]
            ulysses_groups.append(self.torch.distributed.new_group(ranks=ranks, backend="nccl"))
        return ParallelContext(
            tp_world_size=self.tp_size,
            tp_rank=self.rank % self.tp_size,
            ulysses_world_size=self.ulysses_degree,
            ulysses_rank=self.rank // self.tp_size,
            ring_world_size=1,
            ring_rank=0,
            tp_group=tp_groups[self.rank // self.tp_size],
            ulysses_group=ulysses_groups[self.rank % self.tp_size],
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def modules(self) -> dict[str, Any]:
        assert self.pipeline is not None
        return self.pipeline.modules

    @property
    def last_clean_video_latents(self) -> Any | None:
        return self._last_clean_video_latents

    @property
    def last_clean_audio_latents(self) -> Any | None:
        return self._last_clean_audio_latents

    @property
    def last_video_decode_receipt(self) -> Mapping[str, Any] | None:
        assert self.pipeline is not None
        return self.pipeline.last_video_decode_receipt

    def decode_video_latents(
        self,
        video_latents: Any,
        *,
        output_path: Path,
        video_filter: str | None = None,
        published_video_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        """Collectively decode one complete CPU video-latent timeline."""
        assert self.pipeline is not None
        self.wait_output_decoder_preload()
        resolved_output = output_path.expanduser().resolve()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return self.pipeline._decode_and_save_parallel(
            video_latents=video_latents,
            audio_latents=None,
            output_path=resolved_output,
            fps=24,
            error_prefix="full-sequence Video VAE decode failed",
            decode_audio=False,
            video_filter=video_filter,
            published_video_frames=published_video_frames,
            video_crf=video_crf,
            verify_video_decode=verify_video_decode,
        )

    def preload_output_decoder_async(self) -> None:
        """Load CPU AV-VAE weights while the DiT generates the latent timeline."""
        assert self.pipeline is not None
        if self.pipeline.output_decoder is not None:
            return
        if self._output_decoder_preload_thread is not None:
            return

        def preload() -> None:
            started = time.perf_counter()
            try:
                self.pipeline._load_output_decoder()
            except Exception as exc:
                self._output_decoder_preload_error = _exception_text(exc)
            finally:
                self._output_decoder_preload_seconds = time.perf_counter() - started

        self._output_decoder_preload_thread = threading.Thread(
            target=preload, name=f"h3-output-decoder-preload-rank{self.rank}", daemon=True
        )
        self._output_decoder_preload_thread.start()

    def wait_output_decoder_preload(self) -> None:
        thread = self._output_decoder_preload_thread
        if thread is None:
            return
        thread.join()
        self._output_decoder_preload_thread = None
        collective_error = self._collective_error(
            "output decoder preload", self._output_decoder_preload_error
        )
        if collective_error:
            raise H3RuntimeError(collective_error)
        preload_seconds: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(
            preload_seconds, self._output_decoder_preload_seconds
        )
        if self.is_main_process:
            print(
                json.dumps(
                    {
                        "status": "output_decoder_preload_complete",
                        "seconds_max": max((float(value) for value in preload_seconds)),
                        "seconds_by_rank": [float(value) for value in preload_seconds],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def decode_audio_latents(self, audio_latents: Any) -> tuple[Any, int]:
        """Decode one complete CPU audio-latent timeline on rank zero."""
        if not self.is_main_process:
            raise H3RuntimeError("only rank zero decodes the published audio timeline")
        assert self.pipeline is not None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        decoder = self.pipeline._load_output_decoder()
        timer_context = (
            nullcontext()
            if self._component_timer is None
            else self._component_timer.region("audio_vae")
        )
        with timer_context:
            return decoder.decode_audio(
                audio_latents=audio_latents, device=self.device, offload_after_decode=True
            )

    def component_timing_receipt(self) -> dict[str, Any] | None:
        if self._component_timer is None:
            return None
        return self._component_timer.receipt()

    def inference_acceleration_receipt(self) -> dict[str, Any]:
        receipt = dict(self.model_resolution["inference_acceleration"] or {})
        if self.inference_acceleration.static_timestep_adaln_cache:
            assert self.pipeline is not None
            receipt["static_timestep_adaln_cache_stats"] = (
                self.pipeline.transformer.inference_static_timestep_cache_stats()
            )
        return receipt

    def barrier(self) -> None:
        self.torch.distributed.barrier()

    def broadcast(self, value: Any) -> Any:
        return self._rank_zero_payload(value if self.is_main_process else None)

    def _rank_zero_payload(self, value: _T | None) -> _T:
        payload = [value if self.is_main_process else None]
        self.torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def _collective_error(self, label: str, local_error: str | None) -> str | None:
        gathered: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(
            gathered, {"rank": self.rank, "error": local_error}
        )
        failures = [item for item in gathered if item.get("error")]
        if not failures:
            return None
        return (
            f"{label} failed collectively ("
            + "; ".join((f"rank {item['rank']}: {item['error']}" for item in failures))
            + ")"
        )

    def _collective_request(self, request: Mapping[str, Any]) -> None:
        gathered: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(gathered, dict(request))
        if any((item != request for item in gathered)):
            raise H3RuntimeError("generation request differs across torchrun ranks")

    def precompute_text_prompts(
        self, prompts: Sequence[str], *, parallel: bool = False
    ) -> tuple[tuple[Any, Any], ...]:
        normalized_prompts = tuple(prompts)
        if not normalized_prompts or any(
            (not isinstance(prompt, str) or not prompt.strip() for prompt in normalized_prompts)
        ):
            raise ValueError("precomputed prompts must be non-empty strings")
        self._collective_request(
            {
                "operation": "precompute_text_prompts",
                "prompts": list(normalized_prompts),
                "parallel": parallel,
            }
        )
        assert self.pipeline is not None
        from ..distributed import use_parallel_context

        started = time.perf_counter()
        if self._precomputed_text_path:
            cache_path = Path(self._precomputed_text_path)
            deadline = started + 600.0
            while not cache_path.is_file():
                if time.perf_counter() >= deadline:
                    raise H3RuntimeError(
                        f"timed out waiting for shared text features: {cache_path}"
                    )
                time.sleep(0.25)
            payload = self.torch.load(cache_path, map_location="cpu", weights_only=True)
            if tuple(payload["prompts"]) != normalized_prompts:
                raise H3RuntimeError("shared text features use different prompts")
            encoded = tuple(payload["encoded"])
            if len(encoded) != len(normalized_prompts):
                raise H3RuntimeError("shared text feature count differs")
            self._text_precompute_receipt = {
                "schema": "taomate_hd.inference.text_precompute.v1",
                "mode": "shared_precomputed",
                "prompt_count": len(normalized_prompts),
                "tp_size": self.tp_size,
                "ulysses_degree": self.ulysses_degree,
                "seconds_max": time.perf_counter() - started,
                "source_path": str(cache_path),
                "producer": payload["producer"],
            }
            return encoded
        encoded: list[tuple[Any, Any]] = []
        local_encoded: dict[int, tuple[Any, Any]] = {}
        try:
            with use_parallel_context(self.parallel_context):
                for index, prompt in enumerate(normalized_prompts):
                    if (
                        parallel
                        and index % self.ulysses_degree != self.parallel_context.ulysses_rank
                    ):
                        continue
                    (hidden, tags) = self.pipeline._encode_text(prompt=prompt)
                    local_encoded[index] = (
                        hidden.detach().to(device="cpu").contiguous(),
                        tags.detach().to(device="cpu").contiguous(),
                    )
        finally:
            self.pipeline.release_text_encoder()
        if parallel:
            for index in range(len(normalized_prompts)):
                owner = index % self.ulysses_degree
                source_rank = owner * self.tp_size + self.parallel_context.tp_rank
                payload = [local_encoded.get(index) if self.rank == source_rank else None]
                self.torch.distributed.broadcast_object_list(
                    payload, src=source_rank, group=self.parallel_context.ulysses_group
                )
                item = payload[0]
                if not isinstance(item, tuple) or len(item) != 2:
                    raise H3RuntimeError(f"parallel text prompt {index} broadcast is invalid")
                encoded.append(item)
        else:
            encoded.extend((local_encoded[index] for index in range(len(normalized_prompts))))
        local_timing = {
            "rank": self.rank,
            "tp_rank": self.parallel_context.tp_rank,
            "ulysses_rank": self.parallel_context.ulysses_rank,
            "encoded_prompt_count": len(local_encoded),
            "seconds": time.perf_counter() - started,
        }
        timings: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(timings, local_timing)
        self._text_precompute_receipt = {
            "schema": "taomate_hd.inference.text_precompute.v1",
            "mode": "ulysses_sharded" if parallel else "replicated",
            "prompt_count": len(normalized_prompts),
            "tp_size": self.tp_size,
            "ulysses_degree": self.ulysses_degree,
            "encoded_prompt_count_by_rank": timings,
            "seconds_max": max((float(item["seconds"]) for item in timings)),
            "prompt_sources": [
                {
                    "prompt_index": index,
                    "owner_ulysses_rank": index % self.ulysses_degree,
                    "source_global_rank_by_tp_rank": [
                        index % self.ulysses_degree * self.tp_size + tp_rank
                        for tp_rank in range(self.tp_size)
                    ],
                }
                for index in range(len(normalized_prompts))
            ]
            if parallel
            else None,
        }
        export_path_value = os.environ.get("H3_PRECOMPUTED_TEXT_EXPORT_PATH")
        if export_path_value and self.is_main_process:
            export_path = Path(export_path_value)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = export_path.with_name(export_path.name + ".tmp")
            self.torch.save(
                {
                    "prompts": normalized_prompts,
                    "encoded": tuple(encoded),
                    "producer": dict(self._text_precompute_receipt),
                },
                pending_path,
            )
            pending_path.replace(export_path)
        return tuple(encoded)

    def text_precompute_receipt(self) -> dict[str, Any] | None:
        receipt = getattr(self, "_text_precompute_receipt", None)
        return None if receipt is None else dict(receipt)

    def generate(
        self,
        *,
        seed: int,
        duration_seconds: float,
        low_resolution: tuple[int, int],
        high_resolution: tuple[int, int],
        video_noise_seed: int,
        audio_noise_seed: int,
        precomputed_text: tuple[Any, Any],
        retain_clean_latents_on_device: bool = False,
    ) -> None:
        from ..progressive_resolution import build_progressive_resolution_plan

        plan = build_progressive_resolution_plan(
            low_width=low_resolution[0],
            low_height=low_resolution[1],
            high_width=high_resolution[0],
            high_height=high_resolution[1],
        )
        result = self.pipeline.generate(
            seed=seed,
            duration_seconds=duration_seconds,
            video_noise_seed=video_noise_seed,
            audio_noise_seed=audio_noise_seed,
            precomputed_text=precomputed_text,
            plan=plan,
        )
        if retain_clean_latents_on_device:
            self._last_clean_video_latents = result.video_latents.detach()
            self._last_clean_audio_latents = result.audio_latents.detach()
        else:
            self._last_clean_video_latents = (
                result.video_latents.detach()
                .to(device="cpu", dtype=self.torch.float32)
                .contiguous()
            )
            self._last_clean_audio_latents = (
                result.audio_latents.detach()
                .to(device="cpu", dtype=self.torch.float32)
                .contiguous()
            )
