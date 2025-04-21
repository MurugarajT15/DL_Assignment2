
#  Project: CNN Image Classifier

This project implements a configurable **Convolutional Neural Network (CNN)** from scratch using PyTorch for image classification on the [iNaturalist 12K dataset](https://www.kaggle.com/datasets/ambityga/inaturalist-12k).

---

##  Folder Structure

```
cnn_project/
├── train.py                    # Main training script with manual arguments
├── models/
│   └── cnn.py                  # CNN architecture definition
├── data/
│   └── transforms.py           # Data loading and augmentation logic
└── data/inaturalist_12k/train/ # Place your dataset here manually
```

---

##  Dependencies

Before running, install the required packages:

```bash
pip install torch torchvision opencv-python
```

---

##  Dataset

Download the dataset from:  
🔗 [iNaturalist 12K on Kaggle](https://www.kaggle.com/datasets/ambityga/inaturalist-12k)

Then extract the `train/` folder into:

```
cnn_project/data/inaturalist_12k/train/
```

The dataset should be in the format:
```
data/inaturalist_12k/train/
    ├── ants/
    ├── butterflies/
    ├── ...
```

---

##  Hyperparameters

The following command-line arguments can be passed to `train.py`:

| Argument         | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `--batch_size`   | `int`   | `64`    | Batch size for training and validation |
| `--num_filters`  | `int`   | `32`    | Number of filters in the first conv layer |
| `--filter_org`   | `str`   | `double`| Filter progression strategy: `same`, `double` |
| `--kernel_size`  | `int*5` | `[3,3,3,3,3]` | List of kernel sizes for 5 conv layers |
| `--act_fn`       | `str`   | `gelu`  | Activation function: `relu`, `gelu` |
| `--num_neurons`  | `int`   | `128`   | Number of neurons in the fully connected layer |
| `--batch_norm`   | `flag`  | `False` | Use batch normalization if set |
| `--dropout_rate` | `float` | `0.0`   | Dropout rate (0 = no dropout) |
| `--lr`           | `float` | `1e-3`  | Learning rate for Adam optimizer |
| `--epochs`       | `int`   | `15`    | Number of training epochs |
| `--data_dir`     | `str`   | `./data/inaturalist_12k/train` | Path to training dataset |

---

##  How to Run the Code

###  Step-by-step from Command Prompt

1. **Navigate to the project folder:**

```bash
cd path\to\CNN
```

>  If there's a space in folder name (`Deep learning`), wrap the path in quotes.

2. **Run training with default settings:**

```bash
python train.py
```

3. **Run with custom hyperparameters:**

```bash
python train.py --batch_size 32 --num_filters 64 --filter_org same --kernel_size 5 5 5 3 3 --act_fn relu --num_neurons 256 --dropout_rate 0.2 --lr 0.0005 --epochs 10 --batch_norm
```

4. **Set custom dataset location:**

```bash
python train.py --data_dir "./my_dataset_path/"
```

---

##  Notes

- Model training progress (loss and accuracy) will be printed after each epoch.
- You can enable/disable batch normalization using `--batch_norm`.
- `--kernel_size` must have **exactly 5 integers** for 5 convolutional layers.
