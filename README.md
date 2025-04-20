# 🦋 Fine-Tuning Pretrained CNNs on the iNaturalist Dataset

This project explores two main approaches to image classification using the iNaturalist dataset:
- **Part A:** Building and training a simple CNN from scratch.
- **Part B:** Fine-tuning a large pretrained model (ResNet50) with various strategies and hyperparameter tuning using Weights & Biases (W&B).

---

## 📁 File Overview

- `parta.ipynb` – Implements a basic convolutional neural network trained from scratch.
- `Partb.ipynb` – Fine-tunes a pretrained ResNet50 using transfer learning and W&B sweeps.
- `README.md` – This documentation file.

---

## 📦 Dependencies

Make sure to install the following before running the notebooks:

```bash
pip install torch torchvision matplotlib numpy wandb
