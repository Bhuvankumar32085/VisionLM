# 👁️ VisionLM: Custom Multimodal Vision-Language Model 

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: UV](https://img.shields.io/badge/package_manager-UV-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0+cu124-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VisionLM** is a modular, transparent, and reproducible **Vision-Language Model (VLM)** built from scratch by coupling independently pretrained vision and language backbones through a custom-engineered Multi-Layer Perceptron (MLP) projection adapter.

---

## 📑 Table of Contents
- [1. Why VisionLM (Built from First Principles)](#1-why-visionlm-built-from-first-principles)
- [2. Complete System Architecture & Dataflow](#2-complete-system-architecture--dataflow)
- [3. Project Directory Structure](#3-project-directory-structure)
- [4. Hardware & Storage Requirements](#4-hardware--storage-requirements)
- [5. Step-by-Step Installation & Quickstart](#5-step-by-step-installation--quickstart)
- [6. Data Pipeline: Download & Preparation](#6-data-pipeline-download--preparation)
- [7. Configuration Guide (`configs/config.yaml`)](#7-configuration-guide-configsconfigyaml)
- [8. Real Training Benchmarks & Results (14% LLaVA Split)](#8-real-training-benchmarks--results-14-llava-split)
- [9. Interactive Streamlit Web UI](#9-interactive-streamlit-web-ui)
- [10. Command-Line Inference](#10-command-line-inference)
- [11. How to Further Improve & Scale Training](#11-how-to-further-improve--scale-training)
- [12. Troubleshooting & FAQ](#12-troubleshooting--faq)

---

## 1. Why VisionLM? (Built from First Principles)

High-level multimodal wrappers (such as monolithic LLaVA or SmolVLM packages) obscure the internal mechanics of vision-language fusion. **VisionLM** builds the entire stack explicitly:

1. **Vision Backbone:** Pretrained Vision Transformer `google/vit-base-patch16-224-in21k` (**FROZEN in Phase 1**)
2. **Trainable Projector:** Custom 2-layer MLP adapter (`Linear(768, 1536) -> GELU -> Linear(1536, 960)`) (**TRAINABLE**)
3. **Language Backbone:** Pretrained decoder-only causal LM `HuggingFaceTB/SmolLM2-360M-Instruct` (**FROZEN in Phase 1**)
4. **True Continuous Multimodal Embedding Fusion:** Visual tokens are projected directly into SmolLM2's continuous embedding space and concatenated into `inputs_embeds`. **No artificial fake image token IDs are created.**

---

## 2. Complete System Architecture & Dataflow

```
                             ┌───────────────────────────────┐
                             │  Input Image [B, 3, 224, 224] │
                             └──────────────┬────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────┐
                    │       ViT-B/16 Pretrained Vision Encoder        │
                    │      (google/vit-base-patch16-224-in21k)        │
                    │                 [FROZEN]                        │
                    └───────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                                Raw Output: [B, 197, 768]
                                            │
                                  (Remove CLS Token)
                                            │
                                            ▼
                           Spatial Patch Tokens: [B, 196, 768]
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────┐
                    │             Trainable MLP Projector             │
                    │   Linear(768 -> 1536) -> GELU -> Linear(1536 -> 960)│
                    │               [TRAINABLE]                       │
                    └───────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                          Projected Visual Tokens: [B, 196, 960]
                                            │
    ┌───────────────────────────────────────┴────────────────────────────────────────┐
    │                                                                                │
    ▼                                                                                ▼
┌───────────────────────────────┐                                ┌───────────────────────────────────────┐
│     User Question Prompt      │                                │          Target Answer Text           │
│  "Question: {question}\nAnswer:"│                                │          "{answer}<|im_end|>"         │
└───────────────┬───────────────┘                                └───────────────────┬───────────────────┘
                │                                                                    │
                └───────────────────────────────┬────────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │        SmolLM2 Tokenizer        │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                                 Text Token IDs: [B, text_len]
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │     SmolLM2 Embedding Layer     │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
                              Text Embeddings: [B, text_len, 960]
                                                │
    ┌───────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│          Combined Multimodal Sequence (`inputs_embeds`): [B, 196 + text_len, 960]  │
│                   [ 196 Visual Embeddings ] + [ Text Embeddings ]                  │
└─────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                    SmolLM2-360M-Instruct Transformer Decoder                       │
│                         (Hidden Size = 960, Vocab = 49152)                         │
│                                      [FROZEN]                                      │
└─────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
                                          ▼
                         Output Logits: [B, 196 + text_len, 49152]
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                            Causal Cross-Entropy Loss                               │
│  Target Labels: [-100 on Visual & Prompt Tokens] | [Active Token IDs on Answers]   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Multimodal Tensor Dimensions Reference

| Stage | Tensor Variable | Shape | Description |
| :--- | :--- | :--- | :--- |
| **Input Image** | `pixel_values` | `[B, 3, 224, 224]` | ViT normalized RGB pixel tensor |
| **ViT Raw Output** | `last_hidden_state` | `[B, 197, 768]` | 1 CLS token + 196 spatial patch tokens |
| **Spatial Patches** | `visual_features` | `[B, 196, 768]` | CLS token removed (`last_hidden_state[:, 1:, :]`) |
| **Projected Visual** | `projected_features` | `[B, 196, 960]` | Visual tokens mapped into LLM dimension |
| **Text Token IDs** | `input_ids` | `[B, text_len]` | Question prompt + answer token IDs |
| **Text Embeddings** | `text_embeds` | `[B, text_len, 960]` | SmolLM2 embedding lookup |
| **Multimodal Input** | `inputs_embeds` | `[B, 196 + text_len, 960]` | Concatenated visual + textual sequence |
| **Attention Mask** | `attention_mask` | `[B, 196 + text_len]` | 1s across visual tokens and valid text tokens |
| **Target Labels** | `labels` | `[B, 196 + text_len]` | `-100` on visual and prompt tokens; true IDs on answer |
| **Output Logits** | `logits` | `[B, 196 + text_len, 49152]` | Vocabulary distribution per sequence position |

---

## 3. Project Directory Structure

```
VisionLM/
├── configs/
│   └── config.yaml             # Central configuration file for models, data, and training
│
├── data/
│   ├── raw/
│   │   └── llava/
│   │       ├── metadata/       # Downloaded raw JSON metadata files
│   │       └── images/         # Downloaded MS-COCO high-resolution JPEG images
│   ├── processed/
│   │   ├── train.jsonl         # Cleaned, validated deterministic training split
│   │   └── validation.jsonl    # Cleaned, validated deterministic validation split
│   ├── subsets/                # Pre-packaged small subsets for quick smoke tests
│   ├── collator.py             # Multimodal batch collator (masks prompt & visual tokens)
│   ├── dataset.py              # PyTorch Dataset with lazy PIL image loading
│   └── preprocessing.py        # Image sanitization & question prompt formatters
│
├── models/
│   ├── vision_encoder.py       # ViT-B/16 wrapper (handles patch extraction & CLS removal)
│   ├── projector.py            # Trainable 2-layer MLP projector
│   ├── language_model.py       # SmolLM2-360M wrapper with LoRA Phase 2 readiness
│   └── vlm.py                  # Integrated VisionLanguageModel orchestrating multimodal flow
│
├── training/
│   ├── trainer.py              # Custom PyTorch training loop (AMP, Grad Accum, Logging)
│   ├── train.py                # Training execution entrypoint
│   ├── evaluate.py             # Standalone quantitative loss & perplexity evaluator
│   ├── checkpoint.py           # Atomic checkpoint save, load, and state restoration
│   └── optimizer.py            # AdamW optimizer & Cosine Annealing scheduler setup
│
├── inference/
│   └── generate.py             # Autoregressive multimodal greedy & top-p generator
│
├── scripts/
│   ├── check_gpu.py            # Hardware, VRAM, and precision capability detector
│   ├── download_dataset.py     # Resumable, multi-threaded LLaVA & COCO image downloader
│   ├── prepare_dataset.py      # Dataset validation, cleaning, and train/val splitting
│   ├── validate_model.py       # Post-training quantitative and qualitative visual QA test
│   ├── inspect_model.py        # Tensor shape flow and parameter breakdown inspector
│   ├── test_forward.py         # Forward pass and loss sanity test
│   ├── test_gradients.py       # Parameter freezing and gradient flow verification
│   ├── test_parameter_update.py# Single-step parameter delta update verification
│   └── test_real_dataset.py    # Batch collator and gradient test on real data samples
│
├── checkpoints/                # Saved model weights (.pt files) [git-ignored]
├── app.py                      # Interactive Streamlit Web Application (Visual QA)
├── main.py                     # Unified CLI entrypoint
├── pyproject.toml              # UV dependency & project configuration
├── uv.lock                     # Deterministic dependency lockfile
└── README.md                   # Complete documentation
```

---

## 4. Hardware & Storage Requirements

### Disk Storage
- **Current Experiment Storage:** ~**97.8 GB** (Total footprint including raw metadata, downloaded COCO image archives, extracted training images, virtual environment, and saved training checkpoints).
- **Recommended Free Disk Space:** **100 GB - 150 GB** SSD (fast NVMe SSD recommended for rapid image batch loading).

### GPU & Compute Allocation
- **Current Tested GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.4).
- **GPU Compute Utilization:** Sustained **80% to 96%** load during training.
- **GPU VRAM Allocation:** **~1.81 GB** out of 6.00 GB (extremely lightweight and safe from Out-Of-Memory errors).

| Hardware Tier | Recommended Settings in `configs/config.yaml` | Suitability |
| :--- | :--- | :--- |
| **6GB VRAM (e.g. RTX 3050 Laptop)** | `batch_size: 1`, `gradient_accumulation_steps: 4`, `precision: bf16` | ✅ **Fully Tested & Verified** |
| **8GB - 12GB VRAM (e.g. RTX 3060/4060/4070)** | `batch_size: 2`, `gradient_accumulation_steps: 2`, `precision: bf16` | 🚀 **Faster Throughput** |
| **16GB - 24GB+ (e.g. RTX 3090/4090/A100)** | `batch_size: 8 - 16`, `gradient_accumulation_steps: 1`, `precision: bf16` | ⚡ **High-Speed Enterprise Training** |
| **CPU-Only Systems** | `precision: fp32` | ⚠️ Inference & inspection only (Training impractical) |

---

## 5. Step-by-Step Installation & Quickstart

> [!IMPORTANT]
> This project uses **UV** as the exclusive package manager. **Do NOT run `pip install`.**

### 1. Clone the Repository
```powershell
git clone https://github.com/<your-username>/VisionLM.git
cd VisionLM
```

### 2. Install Dependencies with UV
```powershell
# Installs PyTorch with CUDA 12.4, Transformers, PEFT, Streamlit, etc.
uv sync
```

### 3. Verify Hardware & CUDA Environment
```powershell
uv run python scripts/check_gpu.py
```
*Expected output: `CUDA Available: True`, `GPU Model: NVIDIA GeForce RTX ...`, `BF16 Support: True`.*

---

## 6. Data Pipeline: Download & Preparation

### Step 1: Download LLaVA-Instruct-150K Metadata & Images

VisionLM includes a robust, multi-threaded, resumable downloader:

```powershell
# Option A: Quick download of 1,000 images (for rapid testing)
uv run python scripts/download_dataset.py --splits detail --subset-size 1000 --workers 16

# Option B: Complete dataset download (Downloads all metadata & COCO images)
uv run python scripts/download_dataset.py --splits all --workers 16
```

- **Features:** Supports `--resume` (skips already existing files), atomic temporary downloads, and automatic PIL image validation.

---

### Step 2: Prepare, Clean, and Split the Dataset

Process raw metadata, filter missing or corrupted images, strip `<image>` tokens, and create deterministic train/validation splits:

```powershell
uv run python scripts/prepare_dataset.py --train-ratio 0.98 --seed 42
```

- **Output:**
  - `data/processed/train.jsonl` (98% Training Set)
  - `data/processed/validation.jsonl` (2% Validation Set)

---

### Step 3: Run Diagnostic Sanity Tests

Run the complete test suite before launching training:

```powershell
# 1. Model architecture and tensor flow breakdown
uv run python scripts/inspect_model.py

# 2. Forward pass and causal loss calculation test
uv run python scripts/test_forward.py

# 3. Parameter isolation test (Asserts ViT and SmolLM2 are frozen, Projector trainable)
uv run python scripts/test_gradients.py

# 4. Parameter update test (Asserts weights update on optimizer step)
uv run python scripts/test_parameter_update.py

# 5. Real dataset collator & gradient test
uv run python scripts/test_real_dataset.py
```

---

## 7. Configuration Guide (`configs/config.yaml`)

All parameters are configured in [`configs/config.yaml`](file:///c:/Users/bhuva/OneDrive/Documents/ComputerVision/VLM_finetune/configs/config.yaml). Edit this file according to your machine:

```yaml
project:
  name: VisionLM              # Project identifier
  seed: 42                    # Random seed for deterministic reproducibility

vision:
  model_name: google/vit-base-patch16-224-in21k  # Official ViT backbone
  image_size: 224             # Input image resolution (224x224)
  patch_size: 16              # Spatial patch size (16x16 -> 196 patches)
  hidden_size: 768            # ViT feature dimension
  use_cls_token: false        # Discard CLS token; keep only spatial patches
  freeze: true                # Keep ViT frozen during Phase 1

language:
  model_name: HuggingFaceTB/SmolLM2-360M-Instruct # Causal LLM backbone
  freeze: true                # Keep SmolLM2 frozen during Phase 1
  use_lora: false             # Set to true for Phase 2 LoRA fine-tuning

projector:
  input_dim: 768              # ViT output dimension
  hidden_dim: 1536            # Intermediate MLP expansion layer
  output_dim: 960             # SmolLM2 hidden dimension
  activation: gelu            # Non-linear activation function
  dropout: 0.0                # Projector dropout rate

training:
  phase: projector            # 'projector' (Phase 1) or 'lora' (Phase 2)
  epochs: 3                   # Total training epochs
  batch_size: 1               # Batch size per GPU step (Keep 1 on 6GB GPUs)
  gradient_accumulation_steps: 4  # Effective batch size = batch_size * accum = 4
  learning_rate: 1.0e-4       # Peak learning rate for AdamW
  weight_decay: 0.01          # Weight decay regularization
  warmup_steps: 100           # Linear learning rate warmup steps
  max_grad_norm: 1.0          # Gradient clipping threshold
  precision: bf16             # Precision mode: 'bf16' (recommended on Ampere+), 'fp16', or 'fp32'
  gradient_checkpointing: false
  num_workers: 0              # DataLoader worker processes
  logging_steps: 10           # Frequency of terminal training logs
  eval_steps: 500             # Validation evaluation frequency

data:
  train_file: data/processed/train.jsonl          # Path to processed training set
  validation_file: data/processed/validation.jsonl# Path to processed validation set
  max_question_length: 128    # Max token length for user query
  max_answer_length: 256      # Max token length for ground truth answer

checkpoint:
  directory: checkpoints      # Directory to save model checkpoints (.pt)
  save_every_steps: 1000      # Checkpoint saving interval
  save_best: true             # Automatically maintain checkpoint_best.pt based on lowest val loss

inference:
  max_new_tokens: 128         # Default max generated tokens
  temperature: 0.7            # Sampling temperature
  top_p: 0.9                  # Nucleus sampling threshold
  do_sample: false            # Default to greedy argmax decoding
```

---

## 8. Real Training Benchmarks & Results (14% LLaVA Split)

The following metrics were achieved during our baseline Phase 1 projector alignment run:

### 🏆 Benchmark Results

| Metric | Result |
| :--- | :--- |
| **Dataset Used** | LLaVA-Instruct-150K (14% downloaded split ~34,000 conversational samples) |
| **Training Steps** | **25,712 optimization steps** across **3 full epochs** |
| **Total Training Time** | **20,586.76 seconds (~5.7 hours)** |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU (6GB VRAM, CUDA 12.4) |
| **GPU Utilization** | **80% - 96% Compute Core Activity** |
| **Peak VRAM Allocated** | **1,811.9 MB (~1.81 GB)** |
| **Initial Loss** | `2.7040` (Step 50) |
| **Final Training Loss** | **`1.5313 - 1.6889`** |
| **Best Validation Loss** | **`1.7223`** (Saved at Step 25,500 in `checkpoints/checkpoint_best.pt`) |
| **Validation Perplexity** | **`5.60`** (Evaluated over 700 unseen validation samples) |

> [!NOTE]
> The close alignment between training loss (`~1.58`) and validation loss (`1.72`) confirms that the MLP projector learned genuine spatial-to-semantic projection without overfitting.

---

## 9. Interactive Streamlit Web UI

VisionLM includes a modern, glassmorphic dark-mode web application for testing custom images:

```powershell
uv run streamlit run app.py
```
*(Or via unified CLI: `uv run python main.py ui`)*

### Key UI Features:
- 🖼️ **Custom Image Upload:** Drag-and-drop your own `.jpg`, `.png`, `.jpeg`, `.webp` images or select built-in demo images.
- 🎯 **Automatic Model Selection:** Automatically detects and loads `checkpoints/checkpoint_best.pt`.
- 💬 **Visual Question Answering:** Enter custom questions or choose from quick preset prompts.
- ⚡ **GPU Resource Caching:** Weights are cached into GPU memory once (`@st.cache_resource`) for fast, repeated inferences.
- 📜 **Session History:** Track previous questions and responses with latency timing.

---

## 10. Command-Line Inference

You can also run autoregressive multimodal generation directly from the terminal:

```powershell
uv run python inference/generate.py \
    --image data/raw/llava/images/000000085564.jpg \
    --question "Describe what is happening in this image in detail." \
    --checkpoint checkpoints/checkpoint_best.pt
```

### Supported Arguments:
- `--image`: Path to input image (`.jpg`, `.png`).
- `--question`: Text prompt or question.
- `--checkpoint`: Path to trained `.pt` checkpoint file.
- `--max_new_tokens`: Number of tokens to generate (default: 64-128).
- `--do_sample`: Enable top-p nucleus sampling (default: greedy).

---

## 11. How to Further Improve & Scale Training

1. **Scale to 100% LLaVA Dataset:**
   - Download the full dataset (`uv run python scripts/download_dataset.py --splits all --workers 16`).
   - Run preparation to generate the full `train.jsonl` (~150,000 samples).
   - Training on 100% data will substantially enrich fine-grained object detection and complex scene reasoning.

2. **Phase 2: Enable LoRA Fine-Tuning on SmolLM2:**
   - In `configs/config.yaml`, set `training.phase: lora` and `language.use_lora: true`.
   - VisionLM is already pre-configured with `peft` LoRA adapters on SmolLM2 attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
   - This allows the language model to adapt its internal reasoning style specifically to visual question answering.

3. **Higher Resolution Input:**
   - You can replace ViT with higher-resolution variants (e.g. ViT-Large or 336px patch models) by updating `vision.model_name` in `config.yaml`.

---

## 12. Troubleshooting & FAQ

### Q1: `CUDA available: False` after cloning?
**Fix:** Run `uv sync`. This ensures the official PyTorch CUDA 12.4 wheel index is downloaded rather than CPU-only wheels.

### Q2: Why use `precision: bf16` instead of `fp16`?
**Fix:** In SmolLM2 attention layers, standard float16 activations can occasionally exceed `65,504`, producing `loss: NaN`. `torch.bfloat16` retains the full 8-bit exponent dynamic range of FP32 while executing with half-precision tensor-core speeds on NVIDIA RTX 30/40 series GPUs.

### Q3: How to resume interrupted training?
**Fix:** Pass the `--resume` flag to continue exactly from the last saved step and optimizer state:
```powershell
uv run python -m training.train --config configs/config.yaml --resume checkpoints/checkpoint_step_20000.pt
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
