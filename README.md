# Exploring-Vision-Transformers-for-CIFAR-10-Classification-Deep-Learning-Model

A complete implementation of Vision Transformer (ViT) for image classification on CIFAR-10 dataset, developed as part of Deep Learning course assignment.
![image alt](https://github.com/annastudent2003/Exploring-Vision-Transformers-for-CIFAR-10-Classification-Deep-Learning-Model-/blob/d9a3f45cf5ec56602a288aa8bc702cdb8bca4db3/sample_images.png)

![image alt](https://github.com/annastudent2003/Exploring-Vision-Transformers-for-CIFAR-10-Classification-Deep-Learning-Model-/blob/57bc37f32a1745f80e56da87c7b34ee1dc1f548e/training_results.png)

## 📋 Project Overview
- **Task**: Image Classification using Vision Transformers
- **Dataset**: CIFAR-10 (50,000 training, 10,000 test images)
- **Model**: Custom Vision Transformer built from scratch
- **Features**: Manual patch embedding, multi-head attention, transformer blocks


## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/vit-cifar10.git
cd vit-cifar10
```

## Install dependencies
```bash
pip install torch torchvision matplotlib numpy tqdm seaborn scikit-learn Pillow
```

## Run the Project
```bash
python main.py
```

## 📊 Results
### Performance Metrics
- **Final Test Accuracy:** 45.45%
- **Training Time:** 53.50 minutes
- **Model Parameters:** 4,854,026
- **Best Validation Accuracy:** 45.59%

### Generated Outputs
- **sample_images.png -** Sample CIFAR-10 images
- **training_results.png -** Training curves & confusion matrix
- **results_summary.txt -** Complete performance metrics
- **vit_cifar10_model.pth -** Trained model weights


## Package Installation Errors
```bash
# Use virtual environment
python -m venv vit_env
source vit_env/bin/activate  # Linux/Mac
vit_env\Scripts\activate    # Windows
pip install -r requirements.txt
```


## 🏗️ Model Architecture
### Vision Transformer Components
- **Patch Embedding:** Convert 32x32 images to 12x12 patches (49 patches total)
- **Positional Encoding:** Learnable positional embeddings
- **Transformer Encoder:** 6 layers with multi-head self-attention
- **Classification Head:** Linear layer for 10-class prediction

### Technical Specifications
- **Hidden Dimension:** 256
- **Number of Heads:** 6
- **Patch Size:** 12×12
- **Transformer Layers:** 6
- **Parameters:** 4.85 million

## ⚙️ Training Configuration
### Hyperparameters
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.03)
- **Loss Function:** Cross Entropy
- **Batch Size:** 128
- **Epochs:** 14
- **Scheduler:** Cosine Annealing

### Dataset
- **CIFAR-10:** Automatically downloaded via PyTorch
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size:** 32×32 pixels


## 📄 License
This project is created for educational purposes as part of academic assignment.

## 👨‍💻 Author
Name: [Ananya Saikia]

Course: Deep Learning 
