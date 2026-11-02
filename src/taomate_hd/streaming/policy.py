"""Fixed continuation policy for the released progressive model."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StreamingInferencePolicy:
    max_prefix_blocks: int = 2
    cross_chunk_kv_recent_blocks: int = 2

    @property
    def recent_prefix_blocks(self) -> int:
        return min(self.max_prefix_blocks, self.cross_chunk_kv_recent_blocks)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
