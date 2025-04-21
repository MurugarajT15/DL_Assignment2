# Pretrained CNN Fine-Tuning on iNaturalist

This project demonstrates how to fine-tune a pretrained model like ResNet50 on the iNaturalist dataset using PyTorch. We use modular code for transformations, model freezing, and training.

## Folder Structure
```
project_folder/
│
├── models/
│   └── pretrained_cnn.py
├── utils/
│   └── transform.py
├── train.py
└── README.md
```

## Features
- Fine-tuning a ResNet50 pretrained model.
- Option to freeze layers (controlled via `--freeze_percent`).
- Easily configurable via `argparse`.

## How to Run
1. Install dependencies:
```bash
pip install torch torchvision
```

2. Run training:
```bash
python train.py --lr 1e-4 --batch_size 64 --epochs 10 --freeze_percent 0.9 --l2 0.0005 --data_path ./data
```

Place your dataset in `./data/train` and `./data/val` directories.

## Insights
- Pretrained models perform better with less data and fewer resources.
- You can tune how many layers to freeze to balance training cost and accuracy.
- Generalizes better than models trained from scratch.
