# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from __future__ import annotations
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_H3_LORA_TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def canonical_h3_lora_targets() -> tuple[str, ...]:
    """Return the adapter's exact 52-block x 4-projection inventory."""
    blocks = [f"token_refiner.blocks.{index}" for index in range(2)]
    blocks.extend((f"blocks.{index}" for index in range(50)))
    return tuple((f"{block}.{suffix}" for block in blocks for suffix in _H3_LORA_TARGET_SUFFIXES))


class H3LoRACheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedH3LoRA:
    rank: int
    alpha: float
    state: Mapping[str, Any]
    loaded_tensor_count: int

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


def _read_adapter_config(adapter_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    root = adapter_dir.expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    return (root, config)


def load_h3_lora_checkpoint(adapter_dir: str | Path) -> LoadedH3LoRA:
    """Load an inference-only LoRA adapter from safetensors."""
    (root, config) = _read_adapter_config(Path(adapter_dir))
    rank = int(config["rank"])
    alpha = float(config["alpha"])
    targets = canonical_h3_lora_targets()
    leaves = ("lora_a", "lora_b")
    weights_path = root / "adapter_model.safetensors"
    safe_open = importlib.import_module("safetensors").safe_open
    selected: dict[str, Any] = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for target in targets:
            for leaf in leaves:
                name = f"{target}.{leaf}"
                tensor = handle.get_tensor(name)
                selected[name] = tensor.contiguous()
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[adapter-load] H3 LoRA tensors loaded={len(selected)}", flush=True)
    return LoadedH3LoRA(
        rank=rank, alpha=float(alpha), state=selected, loaded_tensor_count=len(selected)
    )


def _tensor_parallel_coordinates(module: Any) -> tuple[int, int]:
    tp_size = int(module.tp_size)
    tp_rank = int(module.tp_rank)
    if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
        raise H3LoRACheckpointError("H3 inference LoRA TP coordinates are invalid")
    return (tp_size, tp_rank)


def _column_local_b(full_b: Any, module: Any) -> Any:
    torch = importlib.import_module("torch")
    (tp_size, tp_rank) = _tensor_parallel_coordinates(module)
    output_sizes = module.output_sizes
    pieces: list[Any] = []
    offset = 0
    for raw_size in output_sizes:
        size = int(raw_size)
        if size % tp_size:
            raise H3LoRACheckpointError("H3 merged-column LoRA output is not TP-divisible")
        shard = size // tp_size
        pieces.append(full_b.narrow(0, offset + tp_rank * shard, shard))
        offset += size
    if offset != int(full_b.shape[0]):
        raise H3LoRACheckpointError("H3 merged-column LoRA output inventory differs")
    return torch.cat(pieces, dim=0).contiguous()


def _row_local_a(full_a: Any, module: Any) -> Any:
    (tp_size, tp_rank) = _tensor_parallel_coordinates(module)
    input_size = int(full_a.shape[1])
    if input_size % tp_size:
        raise H3LoRACheckpointError("H3 row LoRA input is not TP-divisible")
    shard = input_size // tp_size
    return full_a.narrow(1, tp_rank * shard, shard).contiguous()


def apply_h3_lora_checkpoint(transformer: Any, adapter_dir: str | Path) -> Mapping[str, Any]:
    """Install a TP-aware, inference-only LoRA bank on the live H3 DiT."""
    torch = importlib.import_module("torch")
    linear = importlib.import_module("taomate_hd.distributed.linear")
    collectives = importlib.import_module("taomate_hd.distributed.collectives")
    loaded = load_h3_lora_checkpoint(adapter_dir)
    adapter_rank = loaded.rank
    adapter_alpha = loaded.alpha
    adapter_scale = loaded.scale
    loaded_tensor_count = loaded.loaded_tensor_count
    targets = canonical_h3_lora_targets()
    modules = dict(transformer.named_modules())
    supported_column = (linear.MergedColumnParallelLinear,)
    supported_row = (linear.RowParallelLinear,)

    class H3InferenceParallelLinearLoRA(torch.nn.Module):
        def __init__(self, base: Any, *, kind: str, a: Any, b: Any) -> None:
            super().__init__()
            self.base = base
            self.kind = kind
            self.scale = adapter_scale
            self.fused_row_reduce_enabled = False
            self._taomate_hd_inference_lora = True
            self.register_buffer("lora_a", a, persistent=False)
            self.register_buffer("lora_b", b, persistent=False)

        def forward(self, inputs: Any) -> Any:
            if (
                self.kind == "row"
                and self.fused_row_reduce_enabled
                and (int(self.base.tp_size) > 1)
            ):
                base_partial = self.base.forward_local(inputs)
                compute_dtype = base_partial.dtype
                update_partial = torch.nn.functional.linear(
                    torch.nn.functional.linear(
                        inputs.to(compute_dtype), self.lora_a.to(compute_dtype)
                    ),
                    self.lora_b.to(compute_dtype),
                )
                output = self.base.reduce_local_output(base_partial + update_partial * self.scale)
                if self.base.bias is not None:
                    output = output + self.base.bias
                return output
            output = self.base(inputs)
            compute_dtype = output.dtype
            update = torch.nn.functional.linear(
                torch.nn.functional.linear(inputs.to(compute_dtype), self.lora_a.to(compute_dtype)),
                self.lora_b.to(compute_dtype),
            )
            if self.kind == "row" and int(self.base.tp_size) > 1:
                update = collectives.tp_all_reduce(update, context=self.base.parallel_context)
            if tuple(update.shape) != tuple(output.shape):
                raise H3LoRACheckpointError("H3 inference LoRA output shape differs")
            return output + update * self.scale

    replacements: list[tuple[str, Any, Any]] = []
    local_numel = 0
    for target in targets:
        module = modules.get(target)
        if type(module) in supported_column:
            if bool(getattr(module, "gather_output", False)):
                raise H3LoRACheckpointError("gathered H3 column output is not qualified")
            kind = "column"
        elif type(module) in supported_row:
            kind = "row"
        else:
            raise H3LoRACheckpointError(f"H3 inference LoRA target type differs: {target}")
        full_a = loaded.state[f"{target}.lora_a"]
        full_b = loaded.state[f"{target}.lora_b"]
        if int(full_a.shape[1]) != int(module.input_size) or int(full_b.shape[0]) != int(
            module.output_size
        ):
            raise H3LoRACheckpointError(f"H3 inference LoRA full shape differs: {target}")
        device = module.weight.device
        if kind == "column":
            local_a = full_a.to(device=device, dtype=torch.float32)
            local_b = _column_local_b(full_b, module).to(device=device, dtype=torch.float32)
        else:
            local_a = _row_local_a(full_a, module).to(device=device, dtype=torch.float32)
            local_b = full_b.to(device=device, dtype=torch.float32)
        if int(local_a.shape[1]) != int(module.weight.shape[1]) or int(local_b.shape[0]) != int(
            module.weight.shape[0]
        ):
            raise H3LoRACheckpointError(f"H3 inference LoRA local shape differs: {target}")
        replacement = H3InferenceParallelLinearLoRA(module, kind=kind, a=local_a, b=local_b).eval()
        replacements.append((target, module, replacement))
        local_numel += int(local_a.numel() + local_b.numel())
    for target, expected, replacement in replacements:
        parent = transformer
        parts = target.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if getattr(parent, parts[-1]) is not expected:
            raise H3LoRACheckpointError(f"H3 inference LoRA target changed: {target}")
        setattr(parent, parts[-1], replacement)
    return {
        "rank": adapter_rank,
        "alpha": adapter_alpha,
        "scale": adapter_scale,
        "target_count": len(replacements),
        "loaded_tensor_count": loaded_tensor_count,
        "local_adapter_numel": local_numel,
        "tensor_parallel_size": int(replacements[0][1].tp_size),
    }


def materialize_h3_lora_bf16_buffers_(transformer: Any) -> Mapping[str, Any]:
    """Materialize the inference LoRA buffers once in their BF16 compute dtype."""
    torch = importlib.import_module("torch")
    adapters = [
        module
        for module in transformer.modules()
        if getattr(module, "_taomate_hd_inference_lora", False)
    ]
    if not adapters:
        raise H3LoRACheckpointError("BF16 LoRA buffers require a loaded adapter")
    converted_numel = 0
    for module in adapters:
        for name in ("lora_a", "lora_b"):
            value = getattr(module, name)
            if value.dtype != torch.bfloat16:
                converted_numel += int(value.numel())
                setattr(module, name, value.to(dtype=torch.bfloat16))
    return {
        "target_count": len(adapters),
        "buffer_count": 2 * len(adapters),
        "converted_numel": converted_numel,
        "dtype": "bfloat16",
    }


def enable_h3_fused_lora_row_reduce_(transformer: Any) -> Mapping[str, Any]:
    """Add each row-parallel LoRA partial before the existing TP reduction."""
    rows = [
        module
        for module in transformer.modules()
        if getattr(module, "_taomate_hd_inference_lora", False)
        and getattr(module, "kind", None) == "row"
    ]
    if not rows:
        raise H3LoRACheckpointError("fused row reduction requires a loaded adapter")
    for module in rows:
        module.fused_row_reduce_enabled = True
    return {
        "row_parallel_module_count": len(rows),
        "tp_all_reduce_calls_saved_per_forward": len(rows),
    }
