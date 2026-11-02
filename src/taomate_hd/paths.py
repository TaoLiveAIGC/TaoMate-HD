# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class H3Paths:
    model_root: Path
    output_root: Path

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.environ["H3_MODEL_ROOT"]).expanduser().resolve(),
            Path(os.environ["H3_OUTPUT_ROOT"]).expanduser().resolve(),
        )

    def output_dir(self, category, run_id):
        return self.output_root / category / run_id
