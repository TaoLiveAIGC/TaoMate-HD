# Adapted for TaoMate-HD inference. See LICENSE and NOTICE.
import torch
import os


def load_execution_torch():
    return torch


def execution_torch_facts(torch):
    return dict(
        source=torch.__file__,
        version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
