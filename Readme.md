# SpectreTrack

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1%2Bcu118-ee4c2c.svg)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-ViT--Base%20(MAE)-orange.svg)](https://huggingface.co/timm/vit_base_patch16_224.mae)
[![Benchmark](https://img.shields.io/badge/Benchmark-GOT--10k-green.svg)](http://got-10k.aitestunion.com/)

**SpectreTrack** is a Vision Transformer-based single-object visual tracker trained with **Teacher–Student Adversarial Patch Masking** and **Feature Reconstruction Distillation** on the [GOT-10k](http://got-10k.aitestunion.com/) benchmark.

---

## Highlights

- **MAE-Pretrained ViT Backbone:** Built on `vit_base_patch16_224.mae` with dynamic image sizing and template-to-search multi-head cross-attention.
- **Adversarial Saliency Masking:** Uses an Exponential Moving Average (EMA) teacher to identify and mask the top-$k$ most salient patches in the search region during training, forcing the student tracker to learn occlusion-resilient representations.
- **Feature Distillation + Box Regression:** Jointly optimizes Generalized IoU ($\text{GIoU}$), Smooth $L_1$, confidence score binary cross-entropy, and normalized feature reconstruction loss.
- **Complete GOT-10k Pipeline:** Includes uninterrupted multi-epoch training, validation evaluation ($\text{AO}$ and $\text{SR}_{0.5}$), official 3-pass test submission packaging (`SpectreTrack.zip`), and an interactive OpenCV sequence inspector.

---

## Methodology & Architecture

### 1. Input Representation
For each tracking pair, image regions are cropped with context padding (`value=(127, 127, 127)`) and resized to fixed resolutions:
- **Template Crop ($z$):** $128 \times 128$ pixels with context factor $0.5$.
- **Search Crop ($x$):** $256 \times 256$ pixels with context factor $2.0$.

### 2. Adversarial Search Masking & EMA Teacher
During training, a momentum-updated teacher network ($\text{momentum} = 0.999$) extracts clean joint template–search features $F_{\text{teacher}}$. Patch importance is computed via the $L_2$ norm across the embedding dimension:

$$
I_i = \| F_{\text{teacher}}^{(i)} \|_2
$$

The top $20\%$ (`mask_ratio = 0.20`) highest-norm search patches ($16 \times 16$ pixels each) are masked out in normalized image space to produce $x_{\text{masked}}$. The student network must then localize the target and reconstruct the teacher's clean feature representation from partial visual evidence.

### 3. Training Objective
SpectreTrack is trained end-to-end with a composite localization and feature reconstruction loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GIoU}}(b, \hat{b}) + 2.0 \, \mathcal{L}_{\text{Smooth-}L_1}(b, \hat{b}) + 0.1 \, \mathcal{L}_{\text{BCE}}(s, \hat{s}) + \lambda_{\text{rec}} \, \mathcal{L}_{\text{rec}}(\tilde{F}_{\text{student}}, \tilde{F}_{\text{teacher}})
$$

where $\tilde{F}$ denotes $L_2$-normalized token embeddings and $\mathcal{L}_{\text{rec}}$ is the Mean Squared Error (MSE) between student and detached teacher representations.

---

## Repository Structure

```text
GOT-10K_SPECTRE/
├── utils/
│   ├── spectreTrack.py        # SpectreTrack ViT-MAE model & cross-attention head
│   └── dataset.py             # GOT-10k dataset loader, jittering, and cropping
├── train_uninterupt.py        # Standalone training script with EMA teacher & checkpoints
├── *.ipynb                    # Interactive notebooks for training, val, test, and inspection
├── requirements.txt           # Python package dependencies
├── evaluation_results/        # Validation bounding box predictions (*.txt)
└── results/
    └── SpectreTrack/          # Official GOT-10k test submission outputs & timing logs
```

---

## Installation

### 1. Clone the Repository & Create Environment
```bash
git clone https://github.com/your-username/GOT-10K_SPECTRE.git
cd GOT-10K_SPECTRE

python3 -m venv spectre-env
source spectre-env/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` targets CUDA 11.8 (`torch==2.7.1+cu118`). Adjust the PyTorch wheel index URL if your system uses a different CUDA version.

### 3. Pretrained Backbone Weights (Offline / Local Setup)
By default, `SpectreTrack` attempts to load local MAE weights before falling back to `timm` / Hugging Face Hub:
```text
/root/victor/vit_base_patch16_224.mae.pth
```
If training on a compute node without internet access, pre-download `vit_base_patch16_224.mae` weights and place them at the configured `checkpoint_path`.

---

## Dataset Preparation

Download the **GOT-10k** dataset (`train`, `val`, and `test` splits) and organize it as follows:

```text
/root/victor/GOT-10K_FULL/
├── train/
│   ├── GOT-10k_Train_000001/
│   │   ├── 00000001.jpg
│   │   ├── ...
│   │   └── groundtruth.txt
│   └── ...
├── val/
│   ├── GOT-10k_Val_000001/
│   └── ...
└── test/
    ├── GOT-10k_Test_000001/
    └── ...
```

Update `data_path`, `val_data_path`, or `test_data_path` in the scripts if your dataset resides in a different directory.

---

## Usage

### 1. Training
Run the standalone training script to fine-tune `SpectreTrack` for 100 epochs on GOT-10k:

```bash
python train_uninterupt.py
```

Checkpoints are saved automatically every 10 epochs (and on the final epoch) as `spectre_got10k_mae_epoch_<N>.pth`.

#### Key Training Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| Backbone | `vit_base_patch16_224.mae` | ViT-B/16 pretrained with Masked Autoencoding |
| Template / Search Size | $128 \times 128$ / $256 \times 256$ | Cropped input resolutions |
| Batch Size | $16$ | Samples per iteration ($60{,}000$ samples/epoch) |
| Optimizer | `AdamW` ($\text{wd} = 10^{-4}$) | Differential LRs: $10^{-5}$ (backbone), $3 \times 10^{-4}$ (head) |
| LR Scheduler | `StepLR` | Decays by $\gamma = 0.1$ every $40$ epochs |
| Adversarial Mask Ratio | $0.20$ | Top $20\%$ salient search patches masked |
| EMA Momentum | $0.999$ | Teacher network parameter decay rate |
| Gradient Clipping | $1.0$ | Max $L_2$ norm clipping across model parameters |

---

### 2. Validation Evaluation
Run the **Validation Evaluation** section of the notebook to evaluate a trained checkpoint across all 180 GOT-10k validation sequences (`GOT-10k_Val_*`).

The validation pipeline features:
- **Temporal Identity Memory (`identity_buffer`):** Initialized on frame 0 and updated via EMA when prediction confidence exceeds $0.6$.
- **Confidence Gating:** Retains the previous bounding box state when confidence drops below $0.2$, and applies confidence-weighted scale smoothing.
- **Metrics Reported:** Average Overlap ($\text{AO}$) and Success Rate at $\text{IoU} > 0.5$ ($\text{SR}_{0.5}$).

---

### 3. Official GOT-10k Test Submission
Run the **Official Test Code** cells in the notebook to generate a submission-ready archive for the GOT-10k evaluation server:
- Evaluates all 180 test sequences (`GOT-10k_Test_*`) across **3 repetitions** (`num_repetitions = 3`).
- Outputs 1-indexed bounding box predictions (`<seq_name>_001.txt` – `_003.txt`) and per-frame inference runtimes (`<seq_name>_time.txt`).
- Automatically compresses `results/SpectreTrack/` into `SpectreTrack.zip`.

---

### 4. Visual Sequence Inspector
Inspect tracking predictions frame-by-frame against ground truth on any validation sequence using the **Sequence Test** cell:
- **Green Box:** Ground Truth (`GT`)
- **Red Box:** SpectreTrack Prediction (`Pred`)
- **Controls:**
  - `Space` — Pause / Resume playback
  - `n` — Step forward one frame (while paused)
  - `Esc` — Close the viewer window