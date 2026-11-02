# TaoMate-HD

**Progressive-Resolution Autoregressive Audio-Video Generation**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)]()
  [![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://taoliveaigc.github.io/TaoMate-HD/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/TaoLiveAIGC/TaoMate-HD-preview)

<!-- Demo video -->

TaoMate-HD generates speech and high-resolution video together, one small chunk at a time. It builds each chunk on a coarse video grid before refining it at the output resolution.

- **Three-step progressive generation.** Two evaluations at 480p establish coarse structure, followed by one at 768p for spatial detail. A coupled-noise lift changes the video grid while audio is generated along the same trajectory.
- **Continuous speech and motion.** Each denoising stage retains its own clean audio-video predictions as history for that same stage of the next chunk. An initial video anchor and two recent audio-video chunks carry context across prompt changes.
- **Optional pipeline acceleration.** LOW0, LOW16, HIGH, and decoding run concurrently across eight GPUs. The serial and pipelined modes use the same inference adapter and prompt format.
- **Long-form generation.** Continue a sequence with new prompts without resetting its stage-local history. The paper demonstrates hour-long streams.

Continuation training uses the model's own progressive rollouts. The generator advances through overlapping dialogue windows; the learned fake score keeps the initial extended window, while the frozen teacher score uses the original source-transcript prefix. Speech-completion examples teach the model to finish the remaining words and continue with natural nonspeech behavior.

## News

- **2026-11-03:** 🎉🎉🎉 First release of the [inference code](https://github.com/TaoLiveAIGC/TaoMate-HD) and [preview LoRA weights](https://huggingface.co/TaoLiveAIGC/TaoMate-HD-preview).

## Comparison

| Method | Temporal ↑ | Long-range ↑ | Visual realism ↑ | Audio aesthetics ↑ | CER (%) ↓ | GPUs | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniForcing | 0.786 | 0.278 | 4.85 | 3.571 | — | 1 | 8.32 |
| Lynn | 0.367 | 0.778 | 4.30 | 4.207 | 86.70 | 1 | 1.05 |
| Lynn-Flash | 0.457 | 0.765 | 5.00 | 4.326 | 88.37 | 1 | 1.66 |
| TaoMate | 0.843 | 0.983 | 4.65 | 3.889 | 103.74 | 1 | 11.1 |
| **TaoMate-HD (serial)** | **0.913** | 0.930 | **5.00** | **4.407** | **19.57** | 8 | 4.68 |
| **TaoMate-HD (accelerated)** | **0.913** | 0.928 | **5.00** | 4.367 | 21.28 | 8 | 8.99 |

Quality is evaluated on 100 sequences lasting 1–10 minutes. Temporal measures frame-difference stability; Long-range measures color/brightness continuity. TaoMate-HD timing spans prepared input to decoded video and audio on eight H20 GPUs. Other methods follow the hardware and timing protocols reported in the paper.

Under matched BF16/SDPA execution, progressive resolution provides **1.60×** speedup over direct 768p generation, including audio-video decoding. The optimized deployment provides a further **1.92×** over serial Progressive through stage overlap, LOW-stage fusion, KV reuse, selective W8A8, attention kernels, and codec scheduling. Continuation training reduces CER from **30.75% to 19.57%**, an **11.18 percentage-point** improvement.

## Installation

Use a Linux machine with NVIDIA GPUs, CUDA-compatible PyTorch, and FFmpeg.

```bash
conda create -n taomate-hd python=3.10 -y
conda activate taomate-hd
git clone https://github.com/TaoLiveAIGC/TaoMate-HD.git
cd TaoMate-HD
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
# Ubuntu / Debian
sudo apt-get install ffmpeg
```

Serial mode uses BF16 and PyTorch SDPA. Accelerated mode uses SageAttention 2.2.0, FlashAttention-3 (Hopper, 3.0.0b1), and vLLM's CUTLASS W8A8 kernels. Install these on the GPU server for `--mode accelerated`:

```bash
pip install 'cmake>=3.26.1' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<80' 'setuptools-scm>=8' wheel \
  'jinja2>=3.1.6' regex build
pip install --no-build-isolation sageattention==2.2.0
pip install --no-build-isolation \
  "vllm @ git+https://github.com/vllm-project/vllm.git@c3a722fcb"
git clone https://github.com/Dao-AILab/flash-attention.git
pip install --no-build-isolation ./flash-attention/hopper
```

The inference environment uses PyTorch 2.8.0, Diffusers 0.35.2, Transformers 4.57.6, and Safetensors 0.7.0. The vLLM revision above provides `scaled_int8_quant` and `cutlass_scaled_mm`; LoRA and the remaining DiT projections stay in BF16.

### Download model weights

Install the Hugging Face command-line tool, then download the [MiniMax H3 FL2VA base model](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the [TaoMate-HD preview LoRA](https://huggingface.co/TaoLiveAIGC/TaoMate-HD-preview):

```bash
pip install -U huggingface_hub

hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" "FL2VA/*" \
  --local-dir models/MiniMax-H3

hf download TaoLiveAIGC/TaoMate-HD-preview \
  --include "config.json" "adapter_model.safetensors" \
  --local-dir models/TaoMate-HD
```

For repositories requiring authentication, run `hf auth login` with an account that has access.

The preview adapter contains `config.json` (LoRA rank and alpha) and `adapter_model.safetensors`. The model directories are:

```text
models/
  MiniMax-H3/
    FL2VA/
      transformer/
      text_encoder/
      tokenizer/
      video_vae/
      audio_vae/
  TaoMate-HD/
    config.json
    adapter_model.safetensors
```

## Generate a video

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference.py \
  --model models/MiniMax-H3 \
  --adapter models/TaoMate-HD \
  --prompts examples/serum.json \
  --duration 10 --resolution 768x1376 \
  --mode serial --gpus 8 --output outputs/serum
```

For accelerated inference, change `--mode serial` to `--mode accelerated`. The eight GPUs are assigned to LOW0/LOW16/HIGH/VAE as **1/1/4/2**. Serial mode uses TP2 × Ulysses4 on eight GPUs.

Accelerated mode decodes completed video latents in overlapping temporal windows while later chunks are generated. Both modes decode accumulated audio latents as a complete sequence; serial mode also decodes video as an assembled sequence. First-RGB latency measures the first ordered decoded video segment.

The result is `video.mp4` under the output directory, alongside the effective configuration and execution logs.

The 10-second example has been run on eight H20 96 GB GPUs in serial and accelerated modes, at 768 × 1376 with 24 FPS video and 32 kHz stereo audio.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | required | MiniMax-H3 directory containing FL2VA |
| `--adapter` | required | TaoMate-HD inference-only adapter directory |
| `--prompts` / `--prompt` | required | Prompt JSON or a single prompt |
| `--duration` | `10` | Output duration in seconds |
| `--resolution` | `768x1376` | Output width × height |
| `--low-resolution` | derived | Coarse canvas; default scale is 480/768 |
| `--mode` | `serial` | `serial` or `accelerated` |
| `--gpus` | `8` | Serial GPU count; accelerated mode uses eight GPUs |
| `--tp` | `2` | Serial tensor-parallel degree |
| `--seed` | `8301` | Seed when the prompt file does not supply seeds |
| `--output` | `outputs/demo` | Output directory |
| `--port` | `21000` | Local communication port base for accelerated mode |

Dimensions follow the model's 32-pixel token grid; there is no resolution whitelist. Prompt windows cover five seconds each, with four small streaming chunks per window. A final partial window is trimmed to the requested duration. For durations beyond the supplied prompt list, the last prompt is reused; provide additional windows to continue the dialogue with new content.

### Prompt format

The example contains two consecutive five-second prompts, preserving the first two PromptOpt v2 benchmark windows and their seeds. Each prompt describes the scene, speaker, action, dialogue, and sound-visual alignment. Consecutive prompts include overlapping dialogue; retained audio history determines where speech continues.

```json
{
  "prompts": ["Summary: ...\nNarration 1: ...\nSpeaker 1 (S1) says: <d>[Chinese] ...</d>"],
  "seeds": [8301]
}
```

## Citation

```bibtex
@misc{taomatehd2026,
  title = {TaoMate-HD: Progressive-Resolution Autoregressive Audio-Video Generation},
  year = {2026}
}
```

## License

This release follows the [MiniMax H3 Community License Agreement](LICENSE). See [NOTICE](NOTICE) for attribution. The upstream license terms also apply to use and redistribution of the base model.

## Acknowledgments

We thank MiniMax for H3, and the authors of Self Forcing, distribution-matching distillation, pipeline-parallel generation, SageAttention, and FlashAttention for their research and implementations.
