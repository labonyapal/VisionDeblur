# Image Deblurring Arena 🔬

> A deep learning project implementing progressive image deblurring architectures — from simple CNN to GAN-based restoration — with a full-stack interactive comparison interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61dafb?style=flat-square&logo=react)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-76b900?style=flat-square&logo=nvidia)

---

## Overview

This project explores image deblurring through five progressively complex deep learning architectures trained on the [GoPro Large Dataset](https://seungjunnah.github.io/Datasets/gopro). Each architecture builds on the limitations of the previous, demonstrating core concepts in modern image restoration research.

| Model        | Architecture                       | Loss                   | PSNR     | SSIM   |
| ------------ | ---------------------------------- | ---------------------- | -------- | ------ |
| CNN          | Simple 3-layer CNN                 | MSE                    | 28.24 dB | 0.8556 |
| U-Net        | Encoder-Decoder + Skip Connections | MSE                    | 28.36 dB | 0.8571 |
| U-Net + VGG  | U-Net                              | MSE + Perceptual       | 28.03 dB | 0.8519 |
| ResNet + VGG | Residual U-Net                     | MSE + Perceptual       | 27.77 dB | 0.8530 |
| ResNet + GAN | Residual U-Net + PatchGAN          | Adversarial + VGG + L1 | 25.97 dB | 0.8386 |

> **Note:** The GAN model intentionally scores lower on PSNR/SSIM — it optimizes for perceptual sharpness over pixel accuracy, consistent with findings in Ledig et al. (SRGAN) and Isola et al. (Pix2Pix).

---

## Key Features

- **5 model architectures** trained from scratch in PyTorch — no pretrained backbones
- **Progressive complexity** — each phase motivated by failure of the previous model
- **Custom loss functions** — MSE, VGG Perceptual Loss, SSIM, Adversarial Loss
- **Stable GAN training** — label smoothing, separate learning rates, D/G update frequency control
- **Full-stack web interface** — upload one blurry image, compare all 5 model outputs live
- **Blur severity analysis** — Laplacian variance-based blur level classification (Mild / Medium / Heavy)
- **Real-time PSNR/SSIM** — computed per-model on uploaded image
- **Interactive bar chart** — visual PSNR comparison across all models

---

## Architecture Overview

```
Input (Blurry Image)
        │
        ▼
┌───────────────────────────────────────────┐
│  Phase 1: Simple CNN                      │
│  Conv → ReLU → Conv → ReLU → Conv        │
│  Establishes pixel-wise baseline          │
└───────────────────────────────────────────┘
        │ (insufficient — loses spatial detail)
        ▼
┌───────────────────────────────────────────┐
│  Phase 2: U-Net                           │
│  Encoder → Bottleneck → Decoder          │
│  Skip connections preserve fine detail   │
└───────────────────────────────────────────┘
        │ (MSE produces blurry textures)
        ▼
┌───────────────────────────────────────────┐
│  Phase 3: U-Net + VGG Perceptual Loss     │
│  Feature-space loss via VGG16            │
│  Recovers high-frequency texture         │
└───────────────────────────────────────────┘
        │ (needs deeper feature extraction)
        ▼
┌───────────────────────────────────────────┐
│  Phase 4: ResNet U-Net                    │
│  Residual blocks in encoder + bottleneck │
│  Stable training of deeper networks      │
└───────────────────────────────────────────┘
        │ (pixel loss ceiling reached)
        ▼
┌───────────────────────────────────────────┐
│  Phase 5: GAN (ResNet + PatchGAN)         │
│  Adversarial training for perceptual      │
│  realism — discriminator as adaptive loss │
└───────────────────────────────────────────┘
        │
        ▼
Output (Deblurred Image)
```

---

## Project Structure

```
image_deblur/
├── models/
│   ├── model.py          # SimpleCNN, UNet, ResNetUNet, PatchGANDiscriminator
│   └── loss.py           # VGGPerceptualLoss
├── utils/
│   └── dataset.py        # GoProDataset, get_loaders()
├── frontend/
│   ├── src/
│   │   └── App.jsx       # React comparison UI
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── train.py              # Train CNN / UNet / ResNet with MSE or VGG loss
├── train_gan.py          # Train GAN with adversarial + VGG + L1 loss
├── evaluate.py           # PSNR / SSIM evaluation on test set
├── infer.py              # Single image inference
├── main.py               # FastAPI backend server
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/labonyapal/VisionDeblur.git
cd image_deblur
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the GoPro Dataset

Download from [Kaggle — GoPro Large Dataset](https://www.kaggle.com/datasets/rahulbhalley/gopro-large-dataset) and organize as:

```
data/
├── train/
│   ├── blur/
│   └── sharp/
└── test/
    ├── blur/
    └── sharp/
```

---

## Training

### Train standard models

```bash
python train.py --model cnn --loss mse
python train.py --model unet --loss mse
python train.py --model unet --loss vgg
python train.py --model resnet --loss vgg
```

### Train GAN

```bash
python train_gan.py
```

Checkpoints saved to `checkpoints/`.

> **Hardware:** Trained on NVIDIA RTX 3060 12GB. Estimated training time: 1–4 hours per phase, 6–8 hours for GAN.

---

## Evaluation

```bash
python evaluate.py
```

Reports PSNR and SSIM for all saved checkpoints on the test set.

---

## Inference

```bash
python infer.py --model cnn
python infer.py --model unet
```

Saves deblurred output as `deblurred_result_<model>.png`.

---

## Running The Web Interface

### Start backend

```bash
python main.py
```

Backend runs at `http://localhost:8000`

### Start frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Frontend runs at `http://localhost:5173`

### Usage

1. Open `http://localhost:5173`
2. Upload any blurry image
3. All 5 models run simultaneously
4. Compare outputs with PSNR/SSIM scores per model
5. Blur severity (Mild/Medium/Heavy) shown automatically

---

## Results

### Quantitative (GoPro Test Set)

| Model        | PSNR ↑   | SSIM ↑ |
| ------------ | -------- | ------ |
| CNN (MSE)    | 28.24 dB | 0.8556 |
| U-Net (MSE)  | 28.36 dB | 0.8571 |
| U-Net (VGG)  | 28.03 dB | 0.8519 |
| ResNet (VGG) | 27.77 dB | 0.8530 |
| ResNet (GAN) | 25.97 dB | 0.8386 |

### Key Findings

- **Skip connections** in U-Net improve PSNR by +0.12 dB over plain CNN while preserving edge detail invisible to PSNR
- **Perceptual loss** shifts optimization from pixel accuracy to feature-space similarity, producing sharper textures
- **GAN training** sacrifices PSNR for perceptual realism — lower PSNR but visually sharper outputs, consistent with published literature
- **PSNR is a limited metric** — does not correlate with human perceptual quality, particularly for GAN outputs

---

## Concepts Demonstrated

| Concept                                          | Where Used                                       |
| ------------------------------------------------ | ------------------------------------------------ |
| Convolutional filters as learnable features      | SimpleCNN                                        |
| Encoder-decoder architecture                     | UNet                                             |
| Skip connections for spatial detail preservation | UNet                                             |
| Residual learning for deep network stability     | ResNetUNet                                       |
| Feature-space vs pixel-space loss                | VGGPerceptualLoss                                |
| Adversarial training dynamics                    | train_gan.py                                     |
| GAN stabilization techniques                     | Label smoothing, LR scheduling, D/G update ratio |
| Quantitative evaluation                          | PSNR, SSIM                                       |

---

## References

- Nah et al. — _Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring_ (GoPro Dataset)
- Isola et al. — _Image-to-Image Translation with Conditional Adversarial Networks_ (Pix2Pix / PatchGAN)
- Johnson et al. — _Perceptual Losses for Real-Time Style Transfer and Super-Resolution_
- Ledig et al. — _Photo-Realistic Single Image Super-Resolution Using a GAN_ (SRGAN)
- Ronneberger et al. — _U-Net: Convolutional Networks for Biomedical Image Segmentation_

---

## License

MIT License — free to use for academic and personal projects.

---

_Built with PyTorch · FastAPI · React · Trained on RTX 3060_
