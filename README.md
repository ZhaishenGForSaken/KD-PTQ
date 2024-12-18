# PyTorch Model Training and Evaluation

This repository contains Python scripts for training and evaluating deep learning models using PyTorch. It includes a teacher model training script (`teachermodeltrain.py`), a student model training script (`studentmodeltrain.py`), and a knowledge distillation script (`knowledgedis.py`) that demonstrates model training with knowledge distillation techniques.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- torchvision 0.11 or higher

Ensure that your system has CUDA-compatible GPUs for hardware acceleration.

## Setup

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-repository/pytorch-models.git
cd pytorch-models
pip install torch torchvision
```
## Usage
### Training the Teacher Model
The teacher model uses a ResNet34 architecture pretrained on ImageNet and is fine-tuned on a custom dataset. The dataset should be organized into a directory structure suitable for (`torchvision.datasets.ImageFolder`).

```bash
python teachermodeltrain.py
```
### Training the Student Model with Knowledge Distillation
The student model training script demonstrates two approaches:

Using MobileNetV2 or a simple CNN as a student model with knowledge distillation from the teacher model.
Run the student model training with knowledge distillation:
```bash
python studentmodeltrain.py
python python knowledgedis.py
```
The script (`knowledgedis.py`) provides functions to test different alpha and temperature values for the knowledge distillation loss function. Adjust the alpha and temperature values in the script to see their impact on the student model's performance.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with any enhancements or bug fixes.

```
This README can be tailored further based on the specific needs of your project or any additional details you might want to include.
```
