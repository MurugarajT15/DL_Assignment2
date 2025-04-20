# Fine-Tuning Pretrained CNNs on the iNaturalist Dataset

This repository contains two parts:

- **Part A:** Training a simple CNN from scratch on the iNaturalist dataset.
- **Part B:** Fine-tuning a large pretrained model (e.g., ResNet50) using various strategies.

---

##  Project Overview

The main goal of this project is to explore the differences between training a model from scratch and fine-tuning large pretrained models using transfer learning techniques.

###  Part A - Training from Scratch
- A custom CNN is built and trained on a subset of the iNaturalist dataset.
- Basic data augmentation and standard training procedures are used.
- Shows the limitations of training from scratch, especially with limited data and compute.

###  Part B - Fine-Tuning Pretrained Models
- Used pretrained ResNet50 from torchvision models.
- Introduced the ability to **freeze a percentage of layers** (e.g., 90%) for efficient fine-tuning.
- Conducted a **W&B sweep** to find the best hyperparameters:
  - Learning Rate: `1e-4`
  - Batch Size: `64`
  - Freeze Percent: `0.9`
  - L2 Regularization: `0.0005`
  - Epochs: `10`

---

##  Key Takeaways

Here are some things I learned while working with pretrained models compared to training from scratch:

- **Less compute needed:** Fine-tuning is faster and requires less hardware than training from scratch.
- **Needs less labeled data:** Pretrained models can work well even with small datasets.
- **Better generalization:** Pretrained models are less likely to overfit and perform better on new data.
- **Supports transfer learning:** Unlike models trained from scratch, pretrained models can be reused for other tasks by adjusting the top layers.

---

##  Files

- `parta.ipynb` – Custom CNN trained from scratch.
- `Partb.ipynb` – Pretrained ResNet50 fine-tuned with W&B hyperparameter sweep.
- `README.md` – You're here :)

---

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/finetune-inaturalist.git
   cd finetune-inaturalist

#  Fine-Tuning Pretrained CNNs on the iNaturalist Dataset

This project explores two main approaches to image classification using the iNaturalist dataset:
- **Part A:** Building and training a simple CNN from scratch.
- **Part B:** Fine-tuning a large pretrained model (ResNet50) with various strategies and hyperparameter tuning using Weights & Biases (W&B).

---

##  File Overview

- `parta.ipynb` – Implements a basic convolutional neural network trained from scratch.
- `Partb.ipynb` – Fine-tunes a pretrained ResNet50 using transfer learning and W&B sweeps.
- `README.md` – This documentation file.

---

##  Dependencies

Make sure to install the following before running the notebooks:

```bash
pip install torch torchvision matplotlib numpy wandb

[WandB Report](https://wandb.ai/iitm-ma23m015/DA6401-Assignment_2/reports/MA23M015-DA6401-Assignment-2--VmlldzoxMjA4NTIyMw)

