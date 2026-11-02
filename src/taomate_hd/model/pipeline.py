# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
"""End-to-end in-process MiniMax H3 T2VA/FL2VA pipeline."""

from __future__ import annotations
import contextlib
import gc
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from taomate_hd.distributed import ParallelContext, use_parallel_context
from .components import partition_root
from .denoise import MiniMaxH3DenoiseBranch
from .dit import MiniMaxH3DiT
from ..denoise_schedule import DISTILLED_STATE_INDICES, select_time_shift_sigmas
from .geometry import MiniMaxH3Geometry, resolve_geometry
from .output import (
    H3LatentDecoder,
    save_h3_mp4,
    save_h3_video_mp4,
    save_h3_video_mp4_from_rgb24_shards,
    write_h3_rgb24_shard,
)
from .packed_sequence import minimax_h3_packed_sequence
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import minimax_h3_text_only_ids
from .progressive_denoise import minimax_h3_progressive_denoise_loop
from .text_encoder import MiniMaxH3TextEncoder, load_minimax_h3_text_weights
from .weight_loading import load_minimax_h3_dit_weights
from ..video_vae_receipt import (
    build_parallel_video_vae_temporal_output_receipt,
    require_single_host_temporal_window_dp,
)
from ..progressive_resolution import (
    ProgressiveResolutionPlan,
    coupled_progressive_video_noise_rows,
    progressive_request_video_noise_seed,
)

NATIVE_EXECUTED_STAGES = (
    "NativeRequestValidation",
    "NativeTextEncoding",
    "NativeVisualEncoding",
    "NativeLatentPreparation",
    "NativeTimestepPreparation",
    "NativeDenoising",
)


@dataclass
class H3NativeGenerationResult:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    last_video_condition_rows: torch.Tensor
    geometry: MiniMaxH3Geometry
    output_path: Path | None
    executed_stages: tuple[str, ...]
    progressive_resolution: Mapping[str, Any] | None = None


class MiniMaxH3NativePipeline:
    """One positive CFG-distilled H3 branch over explicit TP/Ulysses groups."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        device: torch.device,
        parallel_context: ParallelContext,
        rank: int,
        world_size: int,
        defer_transformer: bool = False,
        defer_text_encoder: bool = False,
        transformer_load_w8a8_policy: str | None = None,
    ) -> None:
        self.model_root = Path(model_root).resolve(strict=True)
        self.partition_root = partition_root(self.model_root)
        self.device = device
        self.parallel_context = parallel_context
        self.rank = rank
        self.world_size = world_size
        self.transformer_load_w8a8_policy = transformer_load_w8a8_policy
        self.tokenizer = self._load_tokenizer()
        with use_parallel_context(parallel_context):
            self.text_encoder: MiniMaxH3TextEncoder | None = None
            self.text_weight_receipt: Any | None = None
            if not defer_text_encoder:
                self.load_text_encoder()
            self.transformer: MiniMaxH3DiT | None = None
            self.dit_weight_receipt: Any | None = None
            if not defer_transformer:
                self.load_transformer()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self.transformer.eval()
        self.video_vae: Any | None = None
        self.audio_vae: Any | None = None
        self.output_decoder: H3LatentDecoder | None = None
        self._output_decoder_lock = threading.Lock()
        self.last_video_decode_receipt: dict[str, Any] | None = None
        self._component_timer: Any | None = None
        self._video_vae_inference = {
            "exact_fusions": False,
            "batched_tiles": False,
            "balanced_cpu_output": False,
            "temporal_window_dp": False,
            "temporal_window_microbatch_size": 1,
            "gpu_rgb24_output": False,
            "keep_on_device_after_decode": False,
            "decoder_tile_size": None,
            "decoder_tile_overlap": None,
        }

    def configure_inference_video_vae(
        self,
        *,
        exact_fusions: bool = False,
        batched_tiles: bool = False,
        balanced_cpu_output: bool = False,
        temporal_window_dp: bool = False,
        temporal_window_microbatch_size: int = 1,
        gpu_rgb24_output: bool = False,
        keep_on_device_after_decode: bool = False,
        decoder_tile_size: int | None = None,
        decoder_tile_overlap: int | None = None,
    ) -> None:
        if self.output_decoder is not None:
            raise RuntimeError("Video VAE inference options must be set before decode")
        if temporal_window_dp and (not balanced_cpu_output):
            raise ValueError("Video VAE temporal-window DP requires balanced CPU output")
        if temporal_window_dp:
            require_single_host_temporal_window_dp(world_size=self.world_size)
        if temporal_window_microbatch_size < 1:
            raise ValueError("Video VAE temporal-window microbatch size must be positive")
        if temporal_window_microbatch_size > 1 and (not temporal_window_dp):
            raise ValueError("Video VAE temporal-window microbatching requires temporal-window DP")
        if gpu_rgb24_output and (not temporal_window_dp):
            raise ValueError("Video VAE GPU RGB24 output requires temporal-window DP")
        self._video_vae_inference = {
            "exact_fusions": bool(exact_fusions),
            "batched_tiles": bool(batched_tiles),
            "balanced_cpu_output": bool(balanced_cpu_output),
            "temporal_window_dp": bool(temporal_window_dp),
            "temporal_window_microbatch_size": int(temporal_window_microbatch_size),
            "gpu_rgb24_output": bool(gpu_rgb24_output),
            "keep_on_device_after_decode": bool(keep_on_device_after_decode),
            "decoder_tile_size": decoder_tile_size,
            "decoder_tile_overlap": decoder_tile_overlap,
        }

    def set_component_timer(self, timer: Any | None) -> None:
        self._component_timer = timer

    def load_transformer(self) -> MiniMaxH3DiT:
        """Materialize the DiT after prompt encoding when GPU memory is tight."""
        if self.transformer is None:
            with use_parallel_context(self.parallel_context):
                transformer = MiniMaxH3DiT.allocate(
                    self.device,
                    parallel_context=self.parallel_context,
                    load_time_w8a8_policy=self.transformer_load_w8a8_policy,
                )
                receipt = load_minimax_h3_dit_weights(
                    transformer, self.partition_root / "transformer"
                )
            transformer.eval()
            self.transformer = transformer
            self.dit_weight_receipt = receipt
        return self.transformer

    def load_text_encoder(self) -> MiniMaxH3TextEncoder:
        if self.text_encoder is None:
            with use_parallel_context(self.parallel_context):
                self.text_encoder = MiniMaxH3TextEncoder.allocate(
                    self.partition_root / "text_encoder",
                    device=self.device,
                    context=self.parallel_context,
                )
                self.text_weight_receipt = load_minimax_h3_text_weights(
                    self.text_encoder, self.partition_root / "text_encoder"
                )
            self.text_encoder.eval()
        return self.text_encoder

    @property
    def modules(self) -> dict[str, Any]:
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "video_vae": self.video_vae,
            "audio_vae": self.audio_vae,
        }

    def _load_tokenizer(self) -> Any:
        transformers = __import__("transformers", fromlist=["AutoTokenizer"])
        return transformers.AutoTokenizer.from_pretrained(
            str(self.partition_root / "tokenizer"), local_files_only=True
        )

    def _move_text_encoder(self, device: torch.device | str) -> None:
        if self.text_encoder is None:
            raise RuntimeError("native H3 text encoder was released")
        if self.text_encoder.device != torch.device(device):
            self.text_encoder.to(device)

    def _encode_text(self, *, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        self._move_text_encoder(self.device)
        ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
        tags = torch.ones(int(ids.shape[0]), dtype=torch.long)
        hidden = self.text_encoder.encode_ids(ids)
        return hidden, tags

    @staticmethod
    def _initial_video_noise(
        geometry: MiniMaxH3Geometry, *, generator: torch.Generator
    ) -> torch.Tensor:
        raw_video = torch.randn(
            1,
            24,
            geometry.video_latent_t,
            geometry.video_latent_h,
            geometry.video_latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        return minimax_h3_patchify_video_latent(raw_video, patch_size=(1, 2, 2))

    @staticmethod
    def _initial_audio_noise(
        geometry: MiniMaxH3Geometry, *, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.randn(
            geometry.audio_latent_t * 2, 32, generator=generator, dtype=torch.float32, device="cpu"
        )

    def _offload_text(self) -> None:
        if self.text_encoder is None or self.text_encoder.device.type == "cpu":
            return
        self.text_encoder.to("cpu")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _collective_decode_error(self, error: str | None) -> str | None:
        if self.world_size == 1:
            return error
        errors: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(errors, error)
        return next((str(item) for item in errors if item is not None), None)

    def _load_output_decoder(self) -> H3LatentDecoder:
        with self._output_decoder_lock:
            if self.output_decoder is None:
                self.output_decoder = H3LatentDecoder(
                    video_vae_root=self.partition_root / "video_vae",
                    audio_vae_root=self.partition_root / "audio_vae",
                    video_parallel_group=torch.distributed.group.WORLD
                    if self.world_size > 1
                    else None,
                    video_parallel_rank=self.rank,
                    video_parallel_world_size=self.world_size,
                    load_audio=self.rank == 0,
                )
                options = self._video_vae_inference
                if options["batched_tiles"]:
                    self.output_decoder.video_vae.model.stack_tiling = True
                if options["decoder_tile_size"] is not None:
                    self.output_decoder.video_vae.model.decoder_tile_size = int(
                        options["decoder_tile_size"]
                    )
                if options["decoder_tile_overlap"] is not None:
                    self.output_decoder.video_vae.model.decoder_tile_overlap_min = int(
                        options["decoder_tile_overlap"]
                    )
                if options["exact_fusions"]:
                    self.output_decoder.enable_video_vae_exact_fusions()
                if options["temporal_window_dp"]:
                    self.output_decoder.enable_video_vae_temporal_window_dp(
                        microbatch_size=options["temporal_window_microbatch_size"],
                        gpu_rgb24_output=options["gpu_rgb24_output"],
                    )
                self.output_decoder.enable_video_vae_cpu_temporal_output(
                    balanced_across_ranks=options["balanced_cpu_output"]
                )
            return self.output_decoder

    def _save_balanced_video_shards(
        self,
        *,
        frames: torch.Tensor,
        rank_receipts: Sequence[Mapping[str, Any]],
        output_path: Path,
        fps: int,
        video_filter: str | None,
        published_video_frames: int | None,
        video_crf: int,
        verify_video_decode: bool,
    ) -> Path:
        root_payload: list[Any] = [None]
        if self.rank == 0:
            try:
                shared_root = Path("/dev/shm")
                if not shared_root.is_dir():
                    raise RuntimeError("balanced Video VAE output requires local /dev/shm")
                root_payload[0] = {
                    "path": tempfile.mkdtemp(
                        prefix="digitalhuman-h3-balanced-vae-", dir=shared_root
                    ),
                    "error": None,
                }
            except Exception as exc:
                root_payload[0] = {"path": None, "error": f"{type(exc).__name__}: {exc}"}
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(root_payload, src=0)
        root_result = root_payload[0]
        if not isinstance(root_result, dict) or root_result.get("error"):
            detail = (
                root_result.get("error")
                if isinstance(root_result, dict)
                else "invalid shared temporary directory receipt"
            )
            raise RuntimeError(str(detail))
        shard_root = Path(str(root_result["path"]))
        local_shard = None
        local_error = None
        try:
            local_receipt = rank_receipts[self.rank]
            local_shard = write_h3_rgb24_shard(
                shard_root / f"rank_{self.rank:05d}.rgb24",
                frames=frames,
                frame_start=int(local_receipt["retained_frame_start"]),
            )
        except Exception as exc:
            local_error = f"rank {self.rank} Video VAE RGB24 shard: {type(exc).__name__}: {exc}"
        shard_error = self._collective_decode_error(local_error)
        gathered_shards: list[Any] = [None] * self.world_size
        if shard_error is None:
            if self.world_size > 1:
                torch.distributed.all_gather_object(gathered_shards, local_shard)
            else:
                gathered_shards[0] = local_shard
        saved = None
        publication_error = shard_error
        if self.rank == 0:
            try:
                if publication_error is None:
                    saved = save_h3_video_mp4_from_rgb24_shards(
                        output_path,
                        shards=gathered_shards,
                        fps=fps,
                        video_filter=video_filter,
                        published_frames=published_video_frames,
                        crf=video_crf,
                        verify_decode=verify_video_decode,
                    )
            except Exception as exc:
                publication_error = (
                    f"rank 0 balanced Video VAE publication: {type(exc).__name__}: {exc}"
                )
            finally:
                shutil.rmtree(shard_root, ignore_errors=True)
        payload: list[Any] = [
            {"error": publication_error, "path": str(saved) if saved is not None else None}
            if self.rank == 0
            else None
        ]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[0]["error"] is not None:
            raise RuntimeError(str(payload[0]["error"]))
        return Path(str(payload[0]["path"]))

    def _decode_and_save_parallel(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor | None,
        output_path: Path | None,
        fps: int,
        error_prefix: str,
        decode_audio: bool = True,
        video_filter: str | None = None,
        published_video_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        balanced_cpu_output = bool(self._video_vae_inference["balanced_cpu_output"])
        if balanced_cpu_output and decode_audio:
            raise ValueError(
                "balanced Video VAE CPU output is available for video-only one-shot publication"
            )
        if balanced_cpu_output and output_path is None:
            raise ValueError("balanced Video VAE CPU output requires output_path")
        self.last_video_decode_receipt = None
        decoder = None
        local_error = None
        try:
            decoder = self._load_output_decoder()
        except Exception as exc:
            local_error = f"rank {self.rank} VAE initialization: {type(exc).__name__}: {exc}"
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"{error_prefix}: {decode_error}")
        assert decoder is not None
        frames = None
        local_error = None
        try:
            timer_context = (
                contextlib.nullcontext()
                if self._component_timer is None
                else self._component_timer.region("video_vae")
            )
            with timer_context:
                frames = decoder.decode_video(
                    video_latents=video_latents,
                    device=self.device,
                    offload_after_decode=not bool(
                        self._video_vae_inference["keep_on_device_after_decode"]
                    ),
                )
        except Exception as exc:
            local_error = f"rank {self.rank} video VAE: {type(exc).__name__}: {exc}"
        decode_error = self._collective_decode_error(local_error)
        if decode_error is not None:
            raise RuntimeError(f"{error_prefix}: {decode_error}")
        assert frames is not None
        local_receipt = decoder.last_video_decode_receipt
        if self.world_size > 1:
            rank_receipts: list[Any] = [None] * self.world_size
            torch.distributed.all_gather_object(rank_receipts, local_receipt)
        else:
            rank_receipts = [local_receipt]
        self.last_video_decode_receipt = build_parallel_video_vae_temporal_output_receipt(
            rank_receipts
        )
        if balanced_cpu_output:
            assert output_path is not None
            return self._save_balanced_video_shards(
                frames=frames,
                rank_receipts=rank_receipts,
                output_path=output_path,
                fps=fps,
                video_filter=video_filter,
                published_video_frames=published_video_frames,
                video_crf=video_crf,
                verify_video_decode=verify_video_decode,
            )
        saved = None
        local_error = None
        if self.rank == 0:
            try:
                if output_path is None:
                    raise ValueError("decoded inference requires output_path")
                if decode_audio:
                    if audio_latents is None:
                        raise ValueError("audio latents are required when decode_audio is enabled")
                    timer_context = (
                        contextlib.nullcontext()
                        if self._component_timer is None
                        else self._component_timer.region("audio_vae")
                    )
                    with timer_context:
                        (waveform, sample_rate) = decoder.decode_audio(
                            audio_latents=audio_latents,
                            device=self.device,
                            offload_after_decode=True,
                        )
                    saved = save_h3_mp4(
                        output_path, frames=frames, audio=waveform, fps=fps, sample_rate=sample_rate
                    )
                else:
                    saved = save_h3_video_mp4(
                        output_path,
                        frames=frames,
                        fps=fps,
                        video_filter=video_filter,
                        published_frames=published_video_frames,
                        crf=video_crf,
                        verify_decode=verify_video_decode,
                    )
            except Exception as exc:
                local_error = f"rank 0 AV publication: {type(exc).__name__}: {exc}"
        else:
            del frames
        payload: list[Any] = [
            {"error": local_error, "path": str(saved) if saved else None}
            if self.rank == 0
            else None
        ]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        publication_error = payload[0]["error"]
        if publication_error is not None:
            raise RuntimeError(f"{error_prefix}: {publication_error}")
        return Path(payload[0]["path"])

    def release_text_encoder(self) -> None:
        """Release the prompt encoder after all training prompts are cached."""
        self.text_encoder = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _build_denoise_branch(
        self,
        *,
        geometry: MiniMaxH3Geometry,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        refined_prompt_embeddings: torch.Tensor | None = None,
    ) -> tuple[MiniMaxH3DenoiseBranch, torch.Tensor]:
        packed = minimax_h3_packed_sequence(
            text_len=int(text_embeddings.shape[0]),
            latent_t=geometry.video_latent_t,
            latent_h=geometry.video_latent_h,
            latent_w=geometry.video_latent_w,
            audio_t=geometry.audio_latent_t,
            include_keyframe_cond=False,
        )
        token_tags = packed["token_tags"]
        token_tags[packed["text_pos"].view(-1)] = text_tags.view(-1)
        positive = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=token_tags,
            device=self.device,
            parallel_context=self.parallel_context,
        )
        positive._taomate_hd_raw_text_embeddings = text_embeddings
        positive._taomate_hd_raw_text_tags = text_tags
        refined = refined_prompt_embeddings
        if refined is None:
            refined = self.transformer.refine_prompt_embeds(
                positive.static_kwargs["prompt_embeds"],
                positive.static_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"],
                device=self.device,
            )
        positive.static_kwargs["prompt_embeds"] = refined
        positive.static_kwargs["refined_prompt_embeds_length"] = int(refined.shape[0])
        positive.static_kwargs["rope_cache"] = self.transformer.build_rope_cache(
            positive.static_kwargs["img_position_ids"], device=self.device
        )
        return positive, refined

    @torch.no_grad()
    def generate(
        self,
        *,
        seed: int,
        video_noise_seed: int,
        audio_noise_seed: int,
        duration_seconds: float,
        precomputed_text: tuple[torch.Tensor, torch.Tensor],
        plan: ProgressiveResolutionPlan,
    ) -> H3NativeGenerationResult:
        low_geometry = resolve_geometry(
            duration_seconds=duration_seconds, width=plan.low.width, height=plan.low.height
        )
        high_geometry = resolve_geometry(
            duration_seconds=duration_seconds, width=plan.high.width, height=plan.high.height
        )
        coupled_noise_seed = progressive_request_video_noise_seed(
            seed=seed,
            video_noise_seed=video_noise_seed,
            transport_video_latents=low_geometry.video_latent_t,
        )
        video_noise_contract = {
            "authored_seed": seed,
            "provided_video_noise_seed": video_noise_seed,
            "effective_coupled_seed": coupled_noise_seed,
            "transport_video_latents": low_geometry.video_latent_t,
            "request_substream_policy": "full_transport_window_stride",
            "audio_rng_policy": "unchanged_separate_audio_contract",
        }
        with use_parallel_context(self.parallel_context):
            text_embeddings, text_tags = precomputed_text
            self._offload_text()
            low_positive, refined = self._build_denoise_branch(
                geometry=low_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
            )
            (
                initial_video,
                progressive_high_initial_video_target,
                initial_noise_coupling,
            ) = coupled_progressive_video_noise_rows(
                temporal_length=low_geometry.video_latent_t,
                source_canvas=plan.low,
                target_canvas=plan.high,
                noise_seed=coupled_noise_seed,
            )
            audio_generator = torch.Generator(device="cpu").manual_seed(audio_noise_seed)
            self._initial_video_noise(high_geometry, generator=audio_generator)
            initial_audio = self._initial_audio_noise(low_geometry, generator=audio_generator)
            sigmas_video = select_time_shift_sigmas(
                num_steps=50, shift_scale=12.0, state_indices=DISTILLED_STATE_INDICES
            )
            sigmas_audio = select_time_shift_sigmas(
                num_steps=50, shift_scale=3.0, state_indices=DISTILLED_STATE_INDICES
            )
            switch_position = DISTILLED_STATE_INDICES.index(plan.switch_state_index)
            high_positive, _ = self._build_denoise_branch(
                geometry=high_geometry,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                refined_prompt_embeddings=refined,
            )
            high_positive._taomate_hd_coupled_initial_video_target_rows = (
                progressive_high_initial_video_target
            )
            progressive_result = minimax_h3_progressive_denoise_loop(
                model=self.transformer,
                low_positive=low_positive,
                high_positive=high_positive,
                initial_low_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                low_keyframe_cond_rows=None,
                high_keyframe_cond_rows=None,
                sigmas_video=sigmas_video,
                sigmas_audio=sigmas_audio,
                switch_position=switch_position,
                plan=plan,
                device=self.device,
                lift_noise_seed=coupled_noise_seed,
                native_high_initial_video_target_rows=None,
                native_high_initial_audio_rows=None,
                on_low_step=None,
                on_resolution_lift=None,
                on_high_step=None,
            )
            target_video = progressive_result.video_rows[high_positive.video_target_slice]
            video_latents = minimax_h3_unpatchify_video_tokens(
                target_video,
                latent_shape=(
                    high_geometry.video_latent_t,
                    high_geometry.video_latent_h // 2,
                    high_geometry.video_latent_w // 2,
                    24,
                ),
                patch_size=(1, 2, 2),
            )
            target_audio = progressive_result.audio_rows[high_positive.audio_target_slice]
            audio_latents = minimax_h3_unpack_audio_tokens(
                target_audio, audio_t=high_geometry.audio_latent_t * 2, audio_channel=2
            )
        return H3NativeGenerationResult(
            video_latents=video_latents,
            audio_latents=audio_latents,
            last_video_condition_rows=target_video[
                -(high_geometry.video_latent_h // 2 * (high_geometry.video_latent_w // 2)) :
            ]
            .detach()
            .cpu()
            .contiguous(),
            geometry=high_geometry,
            output_path=None,
            executed_stages=NATIVE_EXECUTED_STAGES,
            progressive_resolution={
                **plan.as_dict(),
                "video_noise_contract": video_noise_contract,
                "lift": progressive_result.lift_receipt,
                "high_target_source": progressive_result.high_target_source,
                "low_model_forwards": progressive_result.low_model_forwards,
                "high_model_forwards": progressive_result.high_model_forwards,
                "initial_noise_coupling": initial_noise_coupling,
            },
        )
