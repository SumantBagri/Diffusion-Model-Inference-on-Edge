# Diffusion Model Inference on Edge

> Benchmarking and optimizing latent diffusion model inference across heterogeneous edge and desktop hardware using PyTorch, TorchScript JIT, ONNX Runtime, and TensorRT.

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-NVIDIA%20L4T-2496ED?logo=docker)](https://hub.docker.com/r/trapdoor20/csc2231)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](./CSC2231_Project_Report.pdf)
[![Course](https://img.shields.io/badge/Course-CSC2231%20UofT-003E74)](https://web.cs.toronto.edu/)

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Models](#models)
- [Optimization Techniques](#optimization-techniques)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

---

## Overview

This project investigates efficient inference of diffusion models on resource-constrained edge hardware, specifically the **NVIDIA Jetson Nano** (4 GB RAM, Maxwell GPU), and compares results against a desktop **NVIDIA RTX 3070** baseline. The primary goal is to quantify the trade-offs between latency, memory usage, and power consumption under several optimization strategies.

Two diffusion model families are implemented from scratch (without Hugging Face `diffusers` as a runtime dependency):

- **Latent Diffusion Model (LDM)** — unconditional 256x256 face generation on CelebA-HQ, using a UNet2D denoiser, VQVAE decoder, and DDIM scheduler (20 inference steps).
- **Stable Diffusion v1.4** — text-to-image generation via a conditional UNet, CLIP text encoder, and DPM-Solver++ scheduler.

Optimization strategies evaluated include FP16 half-precision casting, TorchScript JIT tracing, ONNX export with graph optimization, and NVIDIA TensorRT engine compilation.

**Key Features:**

- Custom PyTorch implementations of UNet2D, VQVAE, AutoencoderKL, CLIP text encoder, attention blocks, and all schedulers (DDIM, Euler-A, DPM-Solver++).
- Structured profiling harness that records per-run latency (denoising + decoding), GPU memory delta, and power draw (via `nvidia-smi` on RTX / `tegrastats` on Jetson).
- Dockerized environments for both Jetson Nano (L4T base) and desktop GPU, with optional JupyterLab interface.
- Pre-collected benchmark CSVs for all device/optimization combinations, ready for plotting without rerunning hardware experiments.

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.6 |
| **Deep Learning** | PyTorch 1.10, TorchScript (JIT) |
| **Inference Runtimes** | ONNX Runtime 1.11, TensorRT 8.2.2, Torch-TensorRT 1.0 |
| **Model Architecture** | UNet2D, VQVAE, AutoencoderKL, CLIP, Attention, ResNet blocks |
| **Schedulers** | DDIM, Euler Ancestral, DPM-Solver Multistep |
| **Datasets / Checkpoints** | CelebA-HQ 256 (CompVis LDM), Stable Diffusion v1.4 |
| **Profiling** | `nvidia-smi`, `tegrastats`, `memory_profiler` |
| **Containerization** | Docker, NVIDIA Container Runtime, L4T base images |
| **Notebooks** | JupyterLab 3.x |
| **Hardware** | NVIDIA Jetson Nano (aarch64), NVIDIA RTX 3070 (x86\_64) |

---

## Models

### Latent Diffusion Model (Unconditional) — `ldm_uncond/`

Implements `LDMPipeline`, a `torch.nn.Module` wrapping:

- `UNet2DModel` — 2D UNet with attention down/up blocks; configured for 64x64 latent space with channel dimensions `[224, 448, 672, 896]`.
- `VQModel` — VQ-VAE decoder that maps 64x64 latents to 256x256 images.
- `DDIMScheduler` — deterministic DDIM sampler (eta=0); 20 inference steps by default.

Pre-trained weights are fetched from Hugging Face (`CompVis/ldm-celebahq-256`).

### Stable Diffusion v1.4 — `stable_diffusion/`

Implements `StableDiffusionPipeline` wrapping:

- `UNet2DConditionModel` — conditional UNet with cross-attention layers.
- `CLIPTextModel` + `CLIPTokenizer` — text prompt encoding (BPE vocabulary, 77-token context).
- `AutoencoderKL` — KL-regularized VAE for latent encode/decode.
- `DPMSolverMultistepScheduler` — fast DPM-Solver++ sampler.

---

## Optimization Techniques

| Technique | Script | Description |
|---|---|---|
| **Baseline (FP32)** | `eval_ldm.py` | Unmodified PyTorch eager execution |
| **FP16 Half-Precision** | `scripts/model/jit_pipeline.py fp16` | Cast model and inputs to `torch.float16` |
| **TorchScript JIT (FP32/FP16)** | `scripts/model/jit_pipeline.py` | Trace full pipeline with `torch.jit.trace`, save `.ptl` |
| **ONNX Export** | `scripts/model/convert_to_onnx.py` | Export UNet to ONNX (opset 13), constant folding enabled |
| **ONNX + ORT Graph Optimization** | `eval/evaluation.py` (`ONNXEvaluator`) | Load ONNX model under `ORT_ENABLE_EXTENDED` optimization |
| **TensorRT** | Jetson/RTX notebooks | Compile TRT engine via Torch-TensorRT; profiled with `trtexec` |

---

## Getting Started

### Prerequisites

- Docker with NVIDIA Container Runtime
- For Jetson Nano: JetPack 4.6.1 (L4T r32.7.1), CUDA 10.2
- For desktop GPU: CUDA 11.x, NVIDIA driver >= 495

### Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/SumantBagri/Diffusion-Model-Inference-on-Edge.git
cd Diffusion-Model-Inference-on-Edge
```

**Step 2 — Download model weights**

```bash
bash scripts/model/init_celeba.sh
```

This downloads the UNet and VQVAE weights for `CompVis/ldm-celebahq-256` from Hugging Face into `ldm_uncond/ldm-celebahq-256/`.

**Step 3 — Build Docker image**

For the JupyterLab + TensorRT environment (Jetson Nano):

```bash
bash scripts/docker/docker_build.sh jupyterlab v4.0
```

For the ONNX Runtime environment (Jetson Nano):

```bash
bash scripts/docker/docker_build.sh onnxruntime v4.1
```

For the base PyTorch environment:

```bash
bash scripts/docker/docker_build_pytorch.sh
```

**Step 4 — Run the container**

```bash
bash scripts/docker/docker_run.sh v4.0
```

The container mounts `~/project` to `/opt/project` and exposes JupyterLab on port `8888` (password: `jetson`).

### Configuration

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `JETSON_NANO` | Set in `Dockerfile.onnxruntime`; controls which `Reader` class is used |
| `JUPYTER_PASSWORD` | `jetson` | JupyterLab login password |
| `DTYPE` | `torch.float16` | Precision for optimization scripts |
| `num_inference_steps` | `20` | DDIM denoising steps in `LDMPipeline` |

---

## Project Structure

```
Diffusion-Model-Inference-on-Edge/
├── ldm_uncond/                    # LDM pipeline and model components
│   ├── latent_diffusion_uncond.py # LDMPipeline: forward, JIT export, ONNX export
│   ├── model/
│   │   ├── unet.py                # UNet2DModel
│   │   ├── vq.py                  # VQ-VAE decoder
│   │   ├── ddim_scheduler.py      # DDIM sampler
│   │   ├── eulera_scheduler.py    # Euler-A sampler
│   │   ├── attention.py           # Self-attention blocks
│   │   ├── resnet.py              # ResNet blocks
│   │   └── unet_blocks.py         # Down/up blocks
│   ├── ldm-celebahq-256/          # Model config files (weights downloaded separately)
│   │   ├── unet/config.json
│   │   ├── vqvae/config.json
│   │   └── scheduler/scheduler_config.json
│   └── model_jit_*.ptl            # Pre-compiled TorchScript models (FP16/FP32/CUDA/CPU)
│
├── stable_diffusion/              # Stable Diffusion v1.4 pipeline
│   ├── stable_diffusion.py        # StableDiffusionPipeline
│   ├── model/
│   │   ├── unet_conditional.py    # UNet2DConditionModel
│   │   ├── autoencoder.py         # AutoencoderKL (VAE)
│   │   ├── clip.py                # CLIP text encoder
│   │   ├── tokenizer.py           # CLIP tokenizer
│   │   ├── dpmsolver_scheduler.py # DPM-Solver++ sampler
│   │   ├── attention.py
│   │   ├── embeddings.py
│   │   ├── resnet.py
│   │   ├── transformer.py
│   │   └── unet_blocks.py
│   └── stable-diffusion-v1-4/     # Model config files
│
├── eval/                          # Evaluation harness
│   ├── evaluation.py              # BaseEvaluator, ONNXEvaluator
│   ├── readers.py                 # RTXReader, JetsonReader (nvidia-smi / tegrastats)
│   └── run_evaluation.py
│
├── scripts/
│   ├── model/
│   │   ├── init_celeba.sh         # Download LDM weights from HuggingFace
│   │   ├── jit_pipeline.py        # Trace and save TorchScript model
│   │   ├── optim_pipeline.py      # Export full pipeline to ONNX
│   │   ├── optim_unet.py          # Export UNet only to ONNX
│   │   └── convert_to_onnx.py     # Standalone ONNX export
│   └── docker/
│       ├── docker_build.sh
│       ├── docker_build_pytorch.sh
│       ├── docker_run.sh
│       ├── docker_stop.sh
│       └── opencv_install.sh
│
├── dockerfiles/
│   ├── Dockerfile.pytorch         # Base PyTorch on L4T (aarch64)
│   ├── Dockerfile.onnxruntime     # Adds ONNX Runtime 1.11 (aarch64)
│   └── Dockerfile.jupyterlab      # JupyterLab + TensorRT 8.2 + Torch-TRT
│
├── output/
│   ├── eval_data/                 # Benchmark CSVs (latency, memory, power)
│   │   ├── jetson_nano_*/
│   │   └── rtx_3070_*/
│   └── images/                    # Output sample images and result plots
│
├── wheels/                        # Pre-built TensorRT / Torch-TensorRT wheels
├── bin/                           # tegrastats binary (Jetson)
├── lit/                           # Literature review notes and PDFs
├── jetson_nano.ipynb              # Jetson Nano benchmark notebook
├── rtx_3070.ipynb                 # RTX 3070 benchmark notebook
├── run_ldm.ipynb                  # LDM generation demo notebook
├── run_sd.ipynb                   # Stable Diffusion generation demo notebook
├── optim.ipynb                    # Optimization workflow notebook
├── evaluation_flops.ipynb         # FLOPs analysis notebook
├── eval_ldm.py                    # Standalone LDM evaluation script
├── jetson.requirements.txt        # Full pinned dependency list for Jetson
└── CSC2231_Project_Report.pdf     # Full project report (University of Toronto CSC2231)
```

---

## Evaluation

The profiling harness (`eval/evaluation.py`) supports two device profiles:

- **RTXReader** — polls `nvidia-smi --query-gpu=memory.used,power.draw` at 250 ms intervals; records peak and average GPU memory (MiB) and power draw (W).
- **JetsonReader** — runs `tegrastats --interval 250`; parses `POM_5V_IN`, `POM_5V_GPU`, `POM_5V_CPU` (mW) and RAM usage (MB).

Each evaluation run records three CSVs per device/optimization pair: `*_latency.csv`, `*_memory.csv`, `*_power.csv`. Ten iterations are collected after a warmup pass.

**Running an evaluation (PyTorch baseline):**

```python
from eval.evaluation import BaseEvaluator
from ldm_uncond.latent_diffusion_uncond import LDMPipeline
import torch

pipeline = LDMPipeline().to(device='cuda', dtype=torch.float32).eval()
evaluator = BaseEvaluator(dev='rtx_3070', perf_cls='baseline')
noise = torch.randn((1, 3, 64, 64), device='cuda', dtype=torch.float32)
evaluator.evaluate(pipeline, noise, num_iters=10)
```

**Running an ONNX evaluation:**

```python
from eval.evaluation import ONNXEvaluator
evaluator = ONNXEvaluator(mpath='uldm_unet_fp16.onnx', dev='rtx_3070', perf_cls='onnx_optim')
evaluator.evaluate(inp=noise, num_iters=10)
```

---

## Results

All benchmark data is in `output/eval_data/`. Summary of mean total latency per inference (20 DDIM steps):

| Device | Optimization | Mean Latency (s) |
|---|---|---|
| RTX 3070 | Baseline (FP32 eager) | ~1.17 |
| RTX 3070 | FP16 JIT (TorchScript) | ~0.81 |
| RTX 3070 | ONNX + ORT Optimized | ~0.88 |
| RTX 3070 | TensorRT | ~0.25 |
| Jetson Nano | Baseline (FP32 eager) | ~87 |
| Jetson Nano | FP16 JIT (TorchScript) | ~93 |

**Key findings:**

- TensorRT on RTX 3070 achieves approximately **4.7x latency reduction** over the FP32 baseline (1.17 s → 0.25 s).
- FP16 TorchScript JIT reduces latency on the RTX 3070 by approximately **30%** with minimal quality degradation.
- The Jetson Nano baseline runs at ~87 s per inference, roughly **74x slower** than the RTX 3070 baseline, reflecting the Maxwell GPU's architectural limitations.
- FP16 JIT on Jetson Nano does not improve latency over the FP32 baseline, likely due to Maxwell's lack of native FP16 tensor cores.
- ONNX Runtime with graph optimization on the RTX 3070 achieves throughput comparable to FP16 JIT.

Sample generated images are in `output/images/` and `ldm_uncond/ldm-celebahq-256/images/`.

---

## References

- **Project Report:** [CSC2231_Project_Report.pdf](./CSC2231_Project_Report.pdf)
- **Project Poster:** [poster/CSC2231 Project Poster.pptx.pdf](./poster/CSC2231%20Project%20Poster.pptx.pdf)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022.
- Song et al., "Denoising Diffusion Implicit Models," ICLR 2021.
- Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling," NeurIPS 2022.
- NVIDIA TensorRT Documentation: https://developer.nvidia.com/tensorrt
- NVIDIA Jetson Nano Developer Kit: https://developer.nvidia.com/embedded/jetson-nano-developer-kit
- CompVis LDM Weights: https://huggingface.co/CompVis/ldm-celebahq-256
- NVIDIA Jetson Deep Learning course: https://courses.nvidia.com/courses/course-v1:DLI+S-RX-02+V2/

---

## Contributing

This is an academic research project for **CSC2231 — Topics in Computer Systems** at the **University of Toronto**. It is not actively maintained for external contributions. Feel free to open issues for questions.

## Contact

**Sumant Bagri** — University of Toronto
GitHub: [@SumantBagri](https://github.com/SumantBagri)
Docker Hub: [trapdoor20](https://hub.docker.com/u/trapdoor20)
