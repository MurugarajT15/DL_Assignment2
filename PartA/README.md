 Project: CNN Image Classifier
This project implements a configurable Convolutional Neural Network (CNN) from scratch using PyTorch for image classification on the iNaturalist 12K dataset.

##Folder Structure
cnn_project/
├── train.py                    # Main training script with manual arguments
├── models/
│   └── cnn.py                  # CNN architecture definition
├── data/
│   └── transforms.py           # Data loading and augmentation logic
└── data/inaturalist_12k/train/ # Place your dataset here manually

📦 Dependencies
Before running, install the required packages:

bash
Copy
Edit
pip install torch torchvision opencv-python
📥 Dataset
Download the dataset from:
🔗 iNaturalist 12K on Kaggle

Then extract the train/ folder into:

bash
Copy
Edit
cnn_project/data/inaturalist_12k/train/
The dataset should be in the format:

bash
Copy
Edit
data/inaturalist_12k/train/
    ├── ants/
    ├── butterflies/
    ├── ...
🧠 Hyperparameters
The following command-line arguments can be passed to train.py:


Argument	Type	Default	Description
--batch_size	int	64	Batch size for training and validation
--num_filters	int	32	Number of filters in the first conv layer
--filter_org	str	double	Filter progression strategy: same, double
--kernel_size	int*5	[3,3,3,3,3]	List of kernel sizes for 5 conv layers
--act_fn	str	gelu	Activation function: relu, gelu
--num_neurons	int	128	Number of neurons in the fully connected layer
--batch_norm	flag	False	Use batch normalization if set
--dropout_rate	float	0.0	Dropout rate (0 = no dropout)
--lr	float	1e-3	Learning rate for Adam optimizer
--epochs	int	15	Number of training epochs
--data_dir	str	./data/inaturalist_12k/train	Path to training dataset
▶️ How to Run the Code
✅ Step-by-step from Command Prompt
Navigate to the project folder:

bash
Copy
Edit
cd "C:\Users\rajum\OneDrive\Desktop\Deep learning\CNN"
⚠️ If there's a space in folder name (Deep learning), wrap the path in quotes.

Run training with default settings:

bash
Copy
Edit
python train.py
Run with custom hyperparameters:

bash
Copy
Edit
python train.py --batch_size 32 --num_filters 64 --filter_org same --kernel_size 5 5 5 3 3 --act_fn relu --num_neurons 256 --dropout_rate 0.2 --lr 0.0005 --epochs 10 --batch_norm
Set custom dataset location:

bash
Copy
Edit
python train.py --data_dir "./my_dataset_path/"
